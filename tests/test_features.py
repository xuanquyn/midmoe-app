import numpy as np
import pandas as pd
import pytest

from midmoe.artifacts import NormParams
from midmoe.features import add_indicators, macd, normalize_array, rsi

# First rows of NFLX printed by notebook `midmoe-pipeline` (cell 30), i.e. the exact output of training-time
# add_indicators: (open, close, rsi).
NFLX = [
    (4.915143, 4.984857, np.nan),
    (4.925857, 4.731143, 0.000000),
    (4.734714, 4.650143, 0.000000),
    (4.734714, 4.674286, 0.763516),
    (4.712000, 4.778000, 4.149221),
    (4.763143, 4.704143, 4.043426),
    (4.709000, 4.554714, 3.830614),
    (4.602143, 4.625571, 6.347683),
    (4.707143, 4.632000, 6.586590),
    (4.654714, 4.625143, 6.567347),
    (4.685714, 4.819143, 14.204281),
    (4.857143, 4.982857, 20.136778),
    (5.923429, 5.846857, 42.667839),
    (5.857143, 6.120571, 47.701587),
    (6.098857, 6.249429, 49.930366),
    (6.257143, 6.379429, 52.146066),
    (6.308714, 6.488143, 53.980114),
    (6.474143, 6.320857, 50.756501),
]


def test_rsi_matches_notebook_output():
    close = pd.Series([r[1] for r in NFLX])
    expected = np.array([r[2] for r in NFLX])
    got = rsi(close).to_numpy()
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], expected[1:], atol=2e-3)


def test_add_indicators_columns_and_ratios():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2015-01-02", periods=len(NFLX)),
            "open": [r[0] for r in NFLX],
            "close": [r[1] for r in NFLX],
            "high": [r[1] * 1.01 for r in NFLX],
            "low": [r[1] * 0.99 for r in NFLX],
        }
    ).iloc[::-1]  # unsorted input must be sorted by date
    out = add_indicators(df)
    assert out["date"].is_monotonic_increasing
    np.testing.assert_allclose(out["ret_1d"].iloc[1], -0.050897, atol=1e-6)
    np.testing.assert_allclose(out["ret_5d"].iloc[5], -0.056313, atol=1e-6)
    np.testing.assert_allclose(out["close_open_ratio"].iloc[0], 1.014184, atol=1e-6)
    assert out["macd"].isna().all()  # < 34 rows: pandas_ta returns None


def test_macd_ema_presma():
    close = pd.Series(np.linspace(10, 20, 60) + np.sin(np.arange(60)))
    m = macd(close)
    assert m.iloc[:25].isna().all() and m.iloc[25:].notna().all()
    # EMA seeded with SMA: at index slow-1 the slow EMA equals the SMA of the first 26 closes
    fast = close.copy()
    fast.iloc[:11] = np.nan
    fast.iloc[11] = close.iloc[:12].mean()
    fast_ema = fast.ewm(span=12, adjust=False).mean()
    np.testing.assert_allclose(m.iloc[25], fast_ema.iloc[25] - close.iloc[:26].mean())


def test_macd_against_pandas_ta_if_available():
    ta = pytest.importorskip("pandas_ta")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    np.testing.assert_allclose(macd(close), ta.macd(close)["MACD_12_26_9"], equal_nan=True)
    np.testing.assert_allclose(rsi(close), ta.rsi(close, 14), equal_nan=True)


def test_normalize_winsorize_then_minmax():
    norm = NormParams(
        feature_cols=["a", "b"],
        clip_min=np.array([-1.0, 0.0]),
        clip_max=np.array([1.0, 10.0]),
        mins=np.array([-1.0, 0.0]),
        maxs=np.array([1.0, 10.0]),
    )
    x = np.array([[-5.0, 5.0], [0.0, 20.0], [1.0, np.nan]])
    out = normalize_array(x, norm)
    np.testing.assert_allclose(out[0], [0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.5, 1.0], atol=1e-6)
    assert np.isnan(out[2, 1]) and out.dtype == np.float32
