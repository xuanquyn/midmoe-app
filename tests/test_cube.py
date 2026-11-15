import datetime as dt

import numpy as np
import pandas as pd
import pytest

from midmoe.artifacts import load_artifacts
from midmoe.cube import NoSessionError, build_inputs
from midmoe.features import add_indicators_all, normalize_array
from midmoe.predict import Predictor


def _prices(tickers, n_days=80, start="2026-06-01", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days).date
    rows = []
    for t in tickers:
        close = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        rows.append(
            pd.DataFrame(
                {"date": dates, "ticker": t, "open": close * 0.99, "high": close * 1.02, "low": close * 0.97, "close": close, "volume": 1000}
            )
        )
    return pd.concat(rows, ignore_index=True), list(dates)


@pytest.fixture(scope="module")
def art(fake_artifacts):
    return load_artifacts(fake_artifacts)


def test_shapes_order_and_values_match_features(art):
    prices, dates = _prices(art.tickers)
    as_of = dates[-1]
    out = build_inputs(prices.sample(frac=1, random_state=0), as_of, art, t_window=16)  # input order must not matter
    assert out.x.shape == (1, 1, 16, len(art.tickers), 7) and out.x.dtype == np.float32
    assert out.window_dates == dates[-16:] and not out.degraded and out.issues == []
    # cell (last day, ticker j) equals normalize(add_indicators) computed directly
    ind = add_indicators_all(prices)
    j = 5
    row = ind[(ind.ticker == art.tickers[j]) & (ind.date == as_of)][art.feature_order].to_numpy(dtype=float)
    np.testing.assert_allclose(out.x[0, 0, -1, j], normalize_array(row, art.norm)[0], atol=1e-6)
    assert out.raw_today.index.tolist() == art.tickers
    assert out.news_mask.sum() == 0 and out.news_emb.shape == (1, len(art.tickers), 768)


def test_runs_through_predictor(art, fake_artifacts):
    prices, dates = _prices(art.tickers)
    out = build_inputs(prices, dates[-1], art, t_window=16, news_emb_by_ticker={art.tickers[0]: np.ones(768)})
    assert out.news_mask[0, 0] == 1 and out.news_emb[0, 0, 0] == 1
    res = Predictor(str(fake_artifacts)).predict(out.x, out.news_emb, out.news_mask)
    assert np.isfinite(res.score_mean).all()


def test_no_session_and_short_history(art):
    prices, dates = _prices(art.tickers)
    saturday = dates[-1] + dt.timedelta(days=(5 - dates[-1].weekday()) % 7 or 7)
    with pytest.raises(NoSessionError):
        build_inputs(prices, saturday, art, t_window=16)
    short, sd = _prices(art.tickers, n_days=10)
    with pytest.raises(ValueError):
        build_inputs(short, sd[-1], art, t_window=16)


def test_short_gap_is_forward_filled(art):
    prices, dates = _prices(art.tickers)
    t = art.tickers[3]
    gap = dates[-3:-1]  # 2 missing sessions inside the window
    prices = prices[~((prices.ticker == t) & prices.date.isin(gap))]
    out = build_inputs(prices, dates[-1], art, t_window=16)
    assert not out.degraded and any("forward-fill 2" in s for s in out.issues)
    j = art.tickers.index(t)
    np.testing.assert_array_equal(out.x[0, 0, -2, j], out.x[0, 0, -4, j])  # filled from the last real session


def test_long_gap_and_missing_ticker_are_degraded(art):
    prices, dates = _prices(art.tickers)
    t_gap, t_none = art.tickers[1], art.tickers[2]
    prices = prices[~((prices.ticker == t_gap) & prices.date.isin(dates[-6:-1]))]
    prices = prices[prices.ticker != t_none]
    prices = pd.concat([prices, prices[prices.ticker == art.tickers[0]].assign(ticker="ZZZ")])
    out = build_inputs(prices, dates[-1], art, t_window=16)
    assert out.degraded
    assert any(s.startswith(f"{t_gap}:") and "degraded" in s for s in out.issues)
    assert np.all(out.x[0, 0, :, art.tickers.index(t_none)] == 0.5)
    assert any("ZZZ" in s for s in out.issues) and not np.isnan(out.x).any()
