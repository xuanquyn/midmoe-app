"""The 7 technical indicators + normalisation, identical to the training notebooks.

`add_indicators` reproduces notebook `midmoe-pipeline` (Phase 5):
    ret_Nd = close.pct_change(N); rsi = ta.rsi(close, 14); macd = ta.macd(close)['MACD_12_26_9'];
    close_open_ratio = close/open; high_low_ratio = high/low
RSI and MACD are re-implemented in pandas following pandas-ta 0.4.x (non-TA-Lib path). That is the version
the Kaggle notebook actually ran: its printed output has RSI defined from the 2nd row, which only the 0.4.x
RMA (ewm adjust=False, no min_periods) produces; 0.3.14b0 is no longer on PyPI anyway. tests/test_features.py
checks against the rows printed in the notebook, and against pandas-ta itself when it is installed.

Both RSI (Wilder RMA) and MACD (EMA seeded with an SMA) are recursive, so their value depends slightly on
where the history starts. Use at least `MIN_HISTORY_SESSIONS` sessions of history before the first date
you need (the effect decays as 0.926^n for MACD).

`normalize` reproduces notebook `midmoe-data-processed` (Phase 5):
    x_norm = (clip(x, clip_min, clip_max) - mins) / (maxs - mins + 1e-8), then clipped to [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from midmoe.artifacts import NormParams

RSI_LENGTH = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MIN_HISTORY_SESSIONS = 250
EPS = 1e-8


def _rma(s: pd.Series, length: int) -> pd.Series:
    # pandas_ta 0.4.x rma: close.ewm(alpha=1/length, adjust=False).mean()  (no min_periods)
    return s.ewm(alpha=1.0 / length, adjust=False).mean()


def rsi(close: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    if len(close) < length + 1:  # pandas_ta returns None -> column of NaN
        return pd.Series(np.nan, index=close.index)
    negative = close.diff(1)
    positive = negative.copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0
    pos_avg = _rma(positive, length)
    neg_avg = _rma(negative, length)
    return 100 * pos_avg / (pos_avg + neg_avg.abs())


def _ema(close: pd.Series, length: int) -> pd.Series:
    # pandas_ta.ema with defaults sma=True, adjust=False
    c = close.copy()
    if len(c) >= length:
        sma_nth = c.iloc[0:length].mean()
        c.iloc[: length - 1] = np.nan
        c.iloc[length - 1] = sma_nth
    return c.ewm(span=length, adjust=False).mean()


def macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.Series:
    close = close.astype(float)
    if len(close) < slow + signal - 1:  # pandas_ta returns None
        return pd.Series(np.nan, index=close.index)
    return _ema(close, fast) - _ema(close, slow)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """One ticker's OHLCV (columns date, open, high, low, close, ...) -> + 7 indicator columns."""
    df = df.sort_values("date").copy()
    close = df["close"].astype(float)
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["rsi"] = rsi(close, RSI_LENGTH)
    df["macd"] = macd(close)
    df["close_open_ratio"] = close / df["open"].astype(float)
    df["high_low_ratio"] = df["high"].astype(float) / df["low"].astype(float)
    return df


def add_indicators_all(prices: pd.DataFrame) -> pd.DataFrame:
    """Long frame [date, ticker, open, high, low, close, volume] -> same + indicators (per ticker)."""
    parts = [add_indicators(g) for _, g in prices.groupby("ticker", sort=False)]
    if not parts:
        return prices.copy()
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"], ignore_index=True)


def normalize_array(values: np.ndarray, norm: NormParams) -> np.ndarray:
    """values (..., D) in `norm.feature_cols` order -> normalised float32 in [0, 1]. NaN stays NaN."""
    clipped = np.clip(values, norm.clip_min, norm.clip_max)
    out = (clipped - norm.mins) / (norm.maxs - norm.mins + EPS)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def normalize(df: pd.DataFrame, norm: NormParams) -> pd.DataFrame:
    """Return a copy of df with the feature columns replaced by their normalised values."""
    out = df.copy()
    out[norm.feature_cols] = normalize_array(df[norm.feature_cols].to_numpy(dtype=np.float64), norm)
    return out


def percentile_rank(values: pd.Series) -> pd.Series:
    """Cross-sectional percentile in [0, 1] (used by explain.feature_snapshot)."""
    return values.rank(pct=True, method="average")
