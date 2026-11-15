"""Load deployment artifacts (configs, ticker order, normalisation params, sector graph).

All constants (ticker order, feature order, window length, normalisation) come from
`artifacts/vn/`; nothing here is hard-coded. See artifacts/README.md for file formats.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

WEIGHT_PATTERN = re.compile(r"model_seed(\d+)\.pt$")


@dataclass(frozen=True)
class NormParams:
    """Winsorize (clip_min/clip_max) then min-max (mins/maxs), per feature, in feature order."""

    feature_cols: list[str]
    clip_min: np.ndarray  # (D,)
    clip_max: np.ndarray  # (D,)
    mins: np.ndarray  # (D,)
    maxs: np.ndarray  # (D,)

    @classmethod
    def from_json(cls, path: Path) -> "NormParams":
        raw = json.loads(Path(path).read_text())

        def arr(key: str) -> np.ndarray:
            return np.asarray(raw[key], dtype=np.float64).reshape(-1)

        params = cls(list(raw["feature_cols"]), arr("clip_min"), arr("clip_max"), arr("mins"), arr("maxs"))
        d = len(params.feature_cols)
        for name in ("clip_min", "clip_max", "mins", "maxs"):
            if getattr(params, name).shape != (d,):
                raise ValueError(f"vn_norm_params.json: '{name}' must have {d} values")
        return params


def normalize_adjacency(sector_graph: np.ndarray) -> "torch.Tensor":
    """SGC-style symmetric normalisation, identical to `A_norm_vn` in midmoe-vn-transfer Phase 1."""
    import torch

    n = sector_graph.shape[0]
    a_hat = sector_graph + np.eye(n, dtype=np.float32) * (sector_graph.diagonal() == 0)
    deg = a_hat.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0
    a_norm = deg_inv_sqrt[:, None] * a_hat * deg_inv_sqrt[None, :]
    return torch.from_numpy(a_norm.astype(np.float32))


@dataclass
class Artifacts:
    root: Path
    model_config: dict
    tickers: list[str]
    feature_order: list[str]
    norm: NormParams
    sector_graph: np.ndarray
    A_norm: "torch.Tensor"

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)

    @property
    def n_features(self) -> int:
        return len(self.feature_order)

    def weight_paths(self) -> list[tuple[int, Path]]:
        """(seed, path) sorted by seed."""
        out = []
        for p in (self.root / "weights").glob("model_seed*.pt"):
            m = WEIGHT_PATTERN.search(p.name)
            if m:
                out.append((int(m.group(1)), p))
        return sorted(out)

    def icb_industry(self):
        """Ticker metadata table (ticker, name, exchange, icb_sector, ...) if present."""
        import pandas as pd

        path = self.root / "icb_industry.csv"
        return pd.read_csv(path) if path.exists() else None


def load_artifacts(artifact_dir: str | Path) -> Artifacts:
    root = Path(artifact_dir)
    model_config = json.loads((root / "model_config.json").read_text())
    tickers = json.loads((root / "vn_tickers_sorted.json").read_text())
    feature_order = json.loads((root / "vn_feature_order.json").read_text())
    norm = NormParams.from_json(root / "vn_norm_params.json")
    sector_graph = np.load(root / "vn_sector_graph.npy")

    if norm.feature_cols != feature_order:
        raise ValueError(
            f"Feature order mismatch: vn_norm_params.json={norm.feature_cols} vs vn_feature_order.json={feature_order}"
        )
    if sector_graph.shape != (len(tickers), len(tickers)):
        raise ValueError(f"vn_sector_graph.npy shape {sector_graph.shape} != ({len(tickers)}, {len(tickers)})")

    return Artifacts(
        root=root,
        model_config=model_config,
        tickers=list(tickers),
        feature_order=list(feature_order),
        norm=norm,
        sector_graph=sector_graph,
        A_norm=normalize_adjacency(sector_graph),
    )
