"""Model inputs for one day (M2.5): x (1,1,T,N,D), news_emb (1,N,E), news_mask (1,N).

    inputs = build_inputs(prices, as_of, art, t_window=predictor.T, news_emb_by_ticker=...)

`prices`: long OHLCV frame [date, ticker, open, high, low, close, volume] with enough history before `as_of`
(features.MIN_HISTORY_SESSIONS) so RSI/MACD match training. Indicators are computed per ticker, normalised with
the frozen training parameters, and the last T sessions ending at `as_of` form the window (same layout as the
cube notebooks: window_end = as_of, ticker order = vn_tickers_sorted.json, feature order = vn_feature_order.json).

Missing data: a ticker's missing session is forward-filled for at most MAX_FFILL sessions (warning). Anything
still missing inside the window marks the run `degraded`; those cells are filled with the ticker's last known
value, or 0.5 (middle of the normalised range) when the ticker has no data at all, so the model can still run.
News embeddings are computed elsewhere (midmoe/news.py, M2.4); tickers without news get a zero vector, mask 0.
"""
from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from midmoe.artifacts import Artifacts
from midmoe.features import add_indicators_all, normalize_array

log = logging.getLogger(__name__)

MAX_FFILL = 3
NEUTRAL_VALUE = 0.5
NEWS_EMB_DIM = 768


class NoSessionError(Exception):
    """`as_of` is not a trading day in the price data (weekend / holiday / data not published yet)."""


@dataclass
class CubeInputs:
    x: np.ndarray  # (1, 1, T, N, D) float32, normalised
    news_emb: np.ndarray  # (1, N, E) float32
    news_mask: np.ndarray  # (1, N) float32
    window_dates: list[dt.date]
    raw_today: pd.DataFrame  # index ticker, raw indicator values on as_of (for explain.feature_snapshot)
    degraded: bool = False
    issues: list[str] = field(default_factory=list)


def build_inputs(
    prices: pd.DataFrame,
    as_of: dt.date,
    art: Artifacts,
    t_window: int,
    news_emb_by_ticker: Mapping[str, np.ndarray] | None = None,
    emb_dim: int = NEWS_EMB_DIM,
) -> CubeInputs:
    tickers, feats = art.tickers, art.feature_order
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= as_of]
    unknown = sorted(set(df["ticker"]) - set(tickers))
    if unknown:
        df = df[df["ticker"].isin(tickers)]

    calendar = sorted(df["date"].unique())
    if not calendar or calendar[-1] != as_of:
        raise NoSessionError(f"Không có phiên giao dịch ngày {as_of.isoformat()} trong dữ liệu giá.")
    if len(calendar) < t_window:
        raise ValueError(f"Cần ít nhất {t_window} phiên, chỉ có {len(calendar)}.")

    ind = add_indicators_all(df)
    issues: list[str] = []
    degraded = False
    window = calendar[-t_window:]

    # (dates x tickers) per feature, aligned to the full calendar so ffill can look back before the window
    raw = {f: ind.pivot(index="date", columns="ticker", values=f).reindex(index=calendar, columns=tickers) for f in feats}
    present = ind.pivot(index="date", columns="ticker", values="close").reindex(index=calendar, columns=tickers).notna()

    missing_win = ~present.loc[window]
    filled_counts = missing_win.sum()
    for t in tickers:
        n_miss = int(filled_counts[t])
        if n_miss:
            run = 0  # longest run of consecutive missing sessions inside the window
            longest = 0
            for miss in missing_win[t]:
                run = run + 1 if miss else 0
                longest = max(longest, run)
            if longest > MAX_FFILL or not present[t].any():
                degraded = True
                issues.append(f"{t}: thiếu {n_miss} phiên trong cửa sổ (liên tiếp tối đa {longest}) -> degraded")
            else:
                issues.append(f"{t}: forward-fill {n_miss} phiên")
                log.warning("%s: forward-filled %d missing session(s)", t, n_miss)

    for f in feats:
        limited = raw[f].ffill(limit=MAX_FFILL)
        raw[f] = limited.where(limited.notna(), raw[f].ffill())  # beyond the limit: last known value (degraded)

    stacked = np.stack([raw[f].loc[window].to_numpy(dtype=np.float64) for f in feats], axis=-1)  # (T, N, D)
    norm = normalize_array(stacked, art.norm)
    nan_cells = np.isnan(norm)
    if nan_cells.any():
        degraded = True
        bad = sorted({tickers[j] for j in np.argwhere(nan_cells)[:, 1]})
        issues.append(f"Không có dữ liệu chỉ báo, dùng giá trị trung tính {NEUTRAL_VALUE}: {', '.join(bad)}")
        norm = np.where(nan_cells, NEUTRAL_VALUE, norm).astype(np.float32)

    news_emb = np.zeros((1, len(tickers), emb_dim), dtype=np.float32)
    news_mask = np.zeros((1, len(tickers)), dtype=np.float32)
    for j, t in enumerate(tickers):
        v = (news_emb_by_ticker or {}).get(t)
        if v is not None:
            news_emb[0, j] = np.asarray(v, dtype=np.float32)
            news_mask[0, j] = 1.0

    raw_today = pd.DataFrame({f: raw[f].loc[as_of] for f in feats}).reindex(tickers)
    if unknown:
        issues.append(f"Bỏ qua mã ngoài danh sách: {', '.join(unknown[:10])}")
    return CubeInputs(
        x=norm[None, None].astype(np.float32),
        news_emb=news_emb,
        news_mask=news_mask,
        window_dates=list(window),
        raw_today=raw_today,
        degraded=degraded,
        issues=issues,
    )
