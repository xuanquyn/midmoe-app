"""Ensemble inference for the deployed MarketMoE (transfer_minimal, conv3d, 10 seeds).

Note on determinism: `MarketMoE.forward` draws `eps = torch.randn_like(x_flat)` even in eval mode
(k_idx=0, so the DiT input gets ~0.01*eps of noise). To make predictions reproducible we fix the
RNG state around every forward pass (`seeded_forward`), without touching the global RNG.
The Kaggle export cell (notebooks/export_vn_weights.py) uses the same convention so the parity
test can compare outputs exactly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from midmoe.artifacts import Artifacts, load_artifacts
from midmoe.model import MarketMoE

log = logging.getLogger(__name__)

MARKET_CONV_TYPE = "conv3d"
DEFAULT_NOISE_SEED = 0
TOP_K = 5


def seeded_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    news_emb: torch.Tensor,
    news_mask: torch.Tensor,
    noise_seed: int = DEFAULT_NOISE_SEED,
) -> torch.Tensor:
    """Run `model` with a fixed RNG state and return R_final (B, N). Global RNG is left untouched."""
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(noise_seed)
        r_final, _, _ = model(x, news_emb, news_mask)
    return r_final


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score. Works on (..., N)."""
    order = np.argsort(-scores, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, scores.shape[-1] + 1), axis=-1)
    return ranks


@dataclass
class PredictionResult:
    tickers: list[str]
    score_mean: np.ndarray  # (N,)
    score_std: np.ndarray  # (N,)
    rank: np.ndarray  # (N,) int, 1 = best
    top5_votes: np.ndarray  # (N,) int, number of seeds that put the ticker in their top-5
    per_seed: np.ndarray  # (n_seeds, N)
    seeds: list[int]

    def to_records(self) -> list[dict]:
        return [
            {
                "ticker": t,
                "score": float(self.score_mean[i]),
                "score_std": float(self.score_std[i]),
                "rank": int(self.rank[i]),
                "top5_votes": int(self.top5_votes[i]),
            }
            for i, t in enumerate(self.tickers)
        ]


def build_model(art: Artifacts, state_dict: dict) -> MarketMoE:
    """Instantiate MarketMoE with shapes inferred from artifacts + weights (no hard-coded T/D)."""
    cfg = art.model_config
    d = art.n_features
    t_window = state_dict["ts_proj.0.weight"].shape[1] // d
    news_emb_dim = state_dict["news_proj.fc.weight"].shape[1]
    model = MarketMoE(
        N=art.n_tickers,
        T=t_window,
        D=d,
        news_emb_dim=news_emb_dim,
        A_norm=art.A_norm.clone(),
        market_conv_type=MARKET_CONV_TYPE,
        F_dim=cfg["F_dim"],
        dit_dim=cfg["dit_dim"],
        dit_heads=cfg["dit_heads"],
        dit_blocks=cfg["dit_blocks"],
        num_middle=cfg["num_middle"],
        num_bottom=cfg["num_bottom"],
        diffusion_K=cfg["diffusion_K"],
    )
    model.load_state_dict(state_dict, strict=True)
    if not torch.allclose(model.A_norm, art.A_norm, atol=1e-6):
        log.warning("A_norm stored in weights differs from the one built from vn_sector_graph.npy")
    model.eval()
    return model


class Predictor:
    def __init__(self, artifact_dir: str | Path, device: str = "cpu", noise_seed: int = DEFAULT_NOISE_SEED):
        self.art = load_artifacts(artifact_dir)
        self.device = torch.device(device)
        self.noise_seed = noise_seed
        weight_paths = self.art.weight_paths()
        if not weight_paths:
            raise FileNotFoundError(f"No weights/model_seed*.pt in {self.art.root}")
        self.seeds: list[int] = []
        self.models: list[MarketMoE] = []
        for seed, path in weight_paths:
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.models.append(build_model(self.art, state).to(self.device))
            self.seeds.append(seed)
        first = self.models[0]
        self.T = first.T
        self.N = first.N
        self.D = first.D
        log.info("Loaded %d models (seeds=%s), T=%d N=%d D=%d", len(self.models), self.seeds, self.T, self.N, self.D)

    @property
    def tickers(self) -> list[str]:
        return self.art.tickers

    def _to_tensors(self, x, news_emb, news_mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def t(a) -> torch.Tensor:
            return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

        x, news_emb, news_mask = t(x), t(news_emb), t(news_mask)
        expected = (1, 1, self.T, self.N, self.D)
        if tuple(x.shape) != expected:
            raise ValueError(f"x shape {tuple(x.shape)} != {expected} (predict one day at a time)")
        if news_emb.shape[:2] != (1, self.N) or tuple(news_mask.shape) != (1, self.N):
            raise ValueError(f"news_emb {tuple(news_emb.shape)} / news_mask {tuple(news_mask.shape)} mismatch")
        return x, news_emb, news_mask

    def predict_per_seed(self, x, news_emb, news_mask, models: list[MarketMoE] | None = None) -> np.ndarray:
        """Raw scores (n_seeds, N) for one day."""
        x, news_emb, news_mask = self._to_tensors(x, news_emb, news_mask)
        outs = [seeded_forward(m, x, news_emb, news_mask, self.noise_seed)[0] for m in (models or self.models)]
        return torch.stack(outs).cpu().numpy().astype(np.float64)

    def predict(self, x, news_emb, news_mask) -> PredictionResult:
        per_seed = self.predict_per_seed(x, news_emb, news_mask)
        return aggregate(per_seed, self.tickers, self.seeds)


def aggregate(per_seed: np.ndarray, tickers: list[str], seeds: list[int], top_k: int = TOP_K) -> PredictionResult:
    """Ensemble = mean score over seeds (decision D3). Also std and top-k votes."""
    score_mean = per_seed.mean(axis=0)
    score_std = per_seed.std(axis=0)
    seed_ranks = ranks_from_scores(per_seed)
    top5_votes = (seed_ranks <= top_k).sum(axis=0)
    return PredictionResult(
        tickers=list(tickers),
        score_mean=score_mean,
        score_std=score_std,
        rank=ranks_from_scores(score_mean),
        top5_votes=top5_votes,
        per_seed=per_seed,
        seeds=list(seeds),
    )
