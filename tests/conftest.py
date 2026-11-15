from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from midmoe.artifacts import normalize_adjacency  # noqa: E402
from midmoe.model import MarketMoE  # noqa: E402

FEATURES = ["ret_1d", "ret_5d", "ret_10d", "rsi", "macd", "close_open_ratio", "high_low_ratio"]


def real_artifact_dir() -> Path:
    return Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts" / "vn"))


@pytest.fixture(scope="session")
def fake_artifacts(tmp_path_factory) -> Path:
    """Small random artifacts (N=12, T=16, 3 seeds) so the pipeline can be tested without Kaggle weights."""
    root = tmp_path_factory.mktemp("artifacts")
    n, t, d, emb = 12, 16, len(FEATURES), 768
    tickers = [f"T{i:02d}" for i in range(n)]
    sectors = np.repeat(np.arange(4), 3)
    graph = (sectors[:, None] == sectors[None, :]).astype(np.float64)
    cfg = {"F_dim": 20, "dit_dim": 32, "dit_heads": 4, "dit_blocks": 2, "num_middle": 3, "num_bottom": 3, "diffusion_K": 100}

    (root / "weights").mkdir()
    (root / "model_config.json").write_text(json.dumps(cfg))
    (root / "vn_tickers_sorted.json").write_text(json.dumps(tickers))
    (root / "vn_feature_order.json").write_text(json.dumps(FEATURES))
    (root / "vn_norm_params.json").write_text(
        json.dumps(
            {
                "feature_cols": FEATURES,
                "clip_min": [-0.06, -0.12, -0.18, 20, -2, 0.95, 1.0],
                "clip_max": [0.06, 0.12, 0.18, 80, 2, 1.05, 1.1],
                "mins": [-0.06, -0.12, -0.18, 20, -2, 0.95, 1.0],
                "maxs": [0.06, 0.12, 0.18, 80, 2, 1.05, 1.1],
            }
        )
    )
    np.save(root / "vn_sector_graph.npy", graph)
    a_norm = normalize_adjacency(graph)
    for seed in range(3):
        torch.manual_seed(seed)
        m = MarketMoE(N=n, T=t, D=d, news_emb_dim=emb, A_norm=a_norm, market_conv_type="conv3d", **cfg)
        torch.save(m.state_dict(), root / "weights" / f"model_seed{seed}.pt")
    return root
