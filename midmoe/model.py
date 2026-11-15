"""MarketMoE architecture.

Copied verbatim from notebook `midmoe-vn-transfer`, Phase 2 ("Rebuild MarketMoE architecture").
DO NOT modify the logic in this file: the parity test (tests/test_parity.py) relies on it.
Hooks needed for explanation (ablating a branch) live in midmoe/explain.py as wrappers.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketConvFlatten1D(nn.Module):
    def __init__(self, N, D, F_dim, hidden=192):
        super().__init__()
        self.N = N
        self.conv = nn.Conv1d(N * D, hidden, kernel_size=1)
        self.proj = nn.Linear(hidden, F_dim)
    def forward(self, x):
        B, _, T, N, D = x.shape
        xt = x.squeeze(1).reshape(B, T, N * D).permute(0, 2, 1)
        feat = torch.relu(self.conv(xt)).mean(dim=2)
        emb = self.proj(feat)
        return emb.unsqueeze(1).expand(-1, self.N, -1)

class MarketConv2D(nn.Module):
    def __init__(self, N, D, F_dim, kernel_n=5, kernel_d=3, hidden=32):
        super().__init__()
        pad_n, pad_d = kernel_n // 2, kernel_d // 2
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=(kernel_n, kernel_d), padding=(pad_n, pad_d))
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=(kernel_n, kernel_d), padding=(pad_n, pad_d))
        self.proj = nn.Linear(hidden * D, F_dim)
    def forward(self, x):
        B, _, T, N, D = x.shape
        xt = x.squeeze(1).reshape(B * T, 1, N, D)
        h = torch.relu(self.conv1(xt)); h = torch.relu(self.conv2(h))
        h = h.reshape(B, T, -1, N, D).mean(dim=1)
        h = h.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.proj(h)

class MarketConv3D(nn.Module):
    def __init__(self, N, D, F_dim, kernel_t=3, kernel_n=5, kernel_d=3, hidden=32):
        super().__init__()
        pad = (kernel_t // 2, kernel_n // 2, kernel_d // 2)
        self.conv1 = nn.Conv3d(1, hidden, kernel_size=(kernel_t, kernel_n, kernel_d), padding=pad)
        self.conv2 = nn.Conv3d(hidden, hidden, kernel_size=(kernel_t, kernel_n, kernel_d), padding=pad)
        self.proj = nn.Linear(hidden * D, F_dim)
    def forward(self, x):
        B, _, T, N, D = x.shape
        h = torch.relu(self.conv1(x)); h = torch.relu(self.conv2(h))
        h = h.mean(dim=2)
        h = h.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.proj(h)

class ZeroMarketConv(nn.Module):
    def __init__(self, N, F_dim):
        super().__init__()
        self.N, self.F_dim = N, F_dim
    def forward(self, x):
        B = x.shape[0]
        return torch.zeros(B, self.N, self.F_dim, device=x.device, dtype=x.dtype)

def build_market_conv(market_conv_type, N, D, F_dim):
    if market_conv_type == 'flatten1d': return MarketConvFlatten1D(N, D, F_dim)
    elif market_conv_type == 'conv2d': return MarketConv2D(N, D, F_dim)
    elif market_conv_type == 'conv3d': return MarketConv3D(N, D, F_dim)
    elif market_conv_type == 'no_market': return ZeroMarketConv(N, F_dim)
    else: raise ValueError(market_conv_type)

class GraphEmbed(nn.Module):
    def __init__(self, in_dim, F_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, F_dim)
    def forward(self, x_flat, A_norm):
        h_prop = torch.einsum('nm,bmf->bnf', A_norm, x_flat)
        return self.proj(h_prop)

class NewsProj(nn.Module):
    def __init__(self, emb_dim, F_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(emb_dim, F_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, news_emb, news_mask):
        out = self.dropout(F.leaky_relu(self.fc(news_emb)))
        return out * news_mask.unsqueeze(-1)

class ConditionEncoder(nn.Module):
    def __init__(self, F_dim, dropout=0.1):
        super().__init__()
        self.ln_market = nn.LayerNorm(F_dim); self.ln_ts = nn.LayerNorm(F_dim)
        self.ln_news = nn.LayerNorm(F_dim); self.ln_graph = nn.LayerNorm(F_dim)
        self.fuse = nn.Sequential(nn.Linear(F_dim * 4, F_dim), nn.LeakyReLU(), nn.Dropout(dropout))
    def forward(self, market_emb, ts_emb, news_emb, graph_emb):
        parts = [self.ln_market(market_emb), self.ln_ts(ts_emb), self.ln_news(news_emb), self.ln_graph(graph_emb)]
        return self.fuse(torch.cat(parts, dim=-1))

def make_beta_schedule(K, beta1=1e-4, betaK=0.2):
    k = torch.arange(1, K + 1, dtype=torch.float32)
    betas = ((K - k) / (K - 1) * math.sqrt(beta1) + (k - 1) / (K - 1) * math.sqrt(betaK)) ** 2
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

class StepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, k):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=k.device) / half)
        args = k.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
        self.ln2 = nn.LayerNorm(dim)
        self.con_proj = nn.Linear(dim, dim)
    def forward(self, e, context):
        attn_out, _ = self.mha(e, e, e)
        e = self.ln1(attn_out + e)
        e = self.ln2(self.ffn(e) + self.con_proj(context) + e)
        return e

class DIT(nn.Module):
    def __init__(self, in_dim, dim, heads, num_blocks, num_middle, G_dim, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim)
        self.step_emb = StepEmbedding(dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(dim, in_dim)
        self.param_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_middle * G_dim),
        )
        self.num_middle, self.G_dim = num_middle, G_dim
    def forward(self, x_k, k_idx, context):
        e = self.in_proj(x_k) + self.step_emb(k_idx).unsqueeze(1)
        for blk in self.blocks: e = blk(e, context)
        eps_theta = self.out_proj(e)
        e_pooled = e.mean(dim=1)
        G = self.param_mlp(e_pooled).view(-1, self.num_middle, self.G_dim)
        return eps_theta, G

class MiddleExpert(nn.Module):
    def __init__(self, num_bottom, G_dim):
        super().__init__()
        self.phi = nn.Parameter(torch.randn(num_bottom, G_dim) * 0.1)
    def forward(self, G_i):
        return G_i.unsqueeze(1) * self.phi.unsqueeze(0)

class FourierMLPExpert(nn.Module):
    def __init__(self, in_dim, F_dim, hidden=64, num_fourier=16, dropout=0.1):
        super().__init__()
        self.register_buffer('Bf', torch.randn(in_dim, num_fourier) * 2.0)
        self.fc1 = nn.Linear(num_fourier * 2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.film1 = nn.Linear(F_dim * 2, hidden * 2)
        self.film2 = nn.Linear(F_dim * 2, hidden * 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_in, G_shared, local_context):
        N_ = x_in.shape[1]
        G_bc = G_shared.unsqueeze(1).expand(-1, N_, -1)
        combined = torch.cat([G_bc, local_context], dim=-1)
        proj = 2 * math.pi * (x_in @ self.Bf)
        fou = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        h = self.fc1(fou)
        scale1, shift1 = self.film1(combined).chunk(2, dim=-1)
        h = self.dropout(torch.relu(h * (1 + scale1) + shift1))
        h = self.fc2(h)
        scale2, shift2 = self.film2(combined).chunk(2, dim=-1)
        h = self.dropout(torch.relu(h * (1 + scale2) + shift2))
        return self.fc3(h).squeeze(-1)

def sparsemax(z, dim=-1):
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = -1
    k = k.view(*shape)
    support = (1 + k * z_sorted) > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (z_cumsum.gather(dim, k_z - 1) - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0)

class SSAMLevel(nn.Module):
    def __init__(self, N, F_dim, noise_std=0.05):
        super().__init__()
        self.pre_norm = nn.LayerNorm(N)        
        self.Wq = nn.Linear(N, F_dim, bias=False)
        self.Wk = nn.Linear(N, F_dim, bias=False)
        self.scale = math.sqrt(F_dim)            
        self.noise_std = noise_std
    def forward(self, O):
        B, E, _ = O.shape
        O_norm = self.pre_norm(O)                
        Q, K = self.Wq(O_norm), self.Wk(O_norm)  
        scores = (Q @ K.transpose(-1, -2) / self.scale).reshape(B, -1)
        attn = torch.softmax(scores, dim=-1).reshape(B, E, E)
        importance = attn.sum(dim=-1)
        if self.training:
            importance = importance + torch.randn_like(importance) * self.noise_std
        weights = sparsemax(importance, dim=-1)
        M = torch.einsum('be,ben->bn', weights, O)   
        return M, weights

class MarketMoE(nn.Module):
    def __init__(self, N, T, D, news_emb_dim, A_norm, market_conv_type,
                 F_dim=20, dit_dim=32, dit_heads=4, dit_blocks=4,
                 num_middle=9, num_bottom=6, diffusion_K=100, beta1=1e-4, betaK=0.2):
        super().__init__()
        self.N, self.T, self.D = N, T, D
        self.num_middle, self.num_bottom, self.K = num_middle, num_bottom, diffusion_K
        self.register_buffer('A_norm', A_norm)
        _, _, alpha_bars = make_beta_schedule(diffusion_K, beta1, betaK)
        self.register_buffer('alpha_bars', alpha_bars)
        self.market_conv = build_market_conv(market_conv_type, N, D, F_dim)
        self.ts_proj = nn.Sequential(nn.Linear(T * D, F_dim), nn.LeakyReLU())
        self.news_proj = NewsProj(news_emb_dim, F_dim)
        self.graph_embed = GraphEmbed(T * D, F_dim)
        self.cond_encoder = ConditionEncoder(F_dim)
        self.context_proj = nn.Linear(F_dim, dit_dim)
        self.dit = DIT(in_dim=T*D, dim=dit_dim, heads=dit_heads, num_blocks=dit_blocks, num_middle=num_middle, G_dim=F_dim)
        self.middle_experts = nn.ModuleList([MiddleExpert(num_bottom, F_dim) for _ in range(num_middle)])
        self.bottom_experts = nn.ModuleList([
            nn.ModuleList([FourierMLPExpert(in_dim=T*D, F_dim=F_dim) for _ in range(num_bottom)])
            for _ in range(num_middle)
        ])
        self.ssam_b2m = SSAMLevel(N=N, F_dim=F_dim)
        self.ssam_m2t = SSAMLevel(N=N, F_dim=F_dim)
    def forward(self, x, news_emb, news_mask):
        B, _, T, N, D = x.shape
        x_flat = x.squeeze(1).permute(0, 2, 1, 3).reshape(B, N, T * D)
        m_emb = self.market_conv(x)
        ts_emb = self.ts_proj(x_flat)
        n_emb = self.news_proj(news_emb, news_mask)
        g_emb = self.graph_embed(x_flat, self.A_norm)
        context = self.cond_encoder(m_emb, ts_emb, n_emb, g_emb)
        context_dit = self.context_proj(context)
        if self.training:
            k_idx = torch.randint(0, self.K, (B,), device=x.device)
        else:
            k_idx = torch.zeros(B, dtype=torch.long, device=x.device)
        ab = self.alpha_bars[k_idx].view(-1, 1, 1)
        eps = torch.randn_like(x_flat)
        x_k = torch.sqrt(ab) * x_flat + torch.sqrt(1 - ab) * eps
        eps_theta, G = self.dit(x_k, k_idx, context_dit)
        M_list = []
        for i in range(self.num_middle):
            G_bottom_i = self.middle_experts[i](G[:, i, :])
            O_i = torch.stack([self.bottom_experts[i][k](x_flat, G_bottom_i[:, k, :], context) for k in range(self.num_bottom)], dim=1)
            M_i, _ = self.ssam_b2m(O_i)
            M_list.append(M_i)
        M_all = torch.stack(M_list, dim=1)
        R_final, _ = self.ssam_m2t(M_all)
        return R_final, eps, eps_theta

class BaselineMLP(nn.Module):
    def __init__(self, T, D, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(T * D, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x, news_emb=None, news_mask=None):
        B, _, T, N, D = x.shape
        x_flat = x.squeeze(1).permute(0, 2, 1, 3).reshape(B, N, T * D)
        r = self.net(x_flat).squeeze(-1)
        return r, None, None
