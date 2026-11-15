import datetime as dt

import pandas as pd
import pytest

from backend.db.models import MarketIndex
from backend.services.market_index import index_returns, upsert_index_bars


def _bars(dates, closes):
    return pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes, "volume": 1000})


def _rets(db):
    return {r.date: r.ret_1d for r in db.query(MarketIndex).order_by(MarketIndex.date)}


D = [dt.date(2026, 11, d) for d in (10, 11, 12, 13, 14)]


def test_upsert_computes_ret_and_is_idempotent(db):
    assert upsert_index_bars(db, _bars(D[:3], [100.0, 110.0, 99.0])) == 3
    upsert_index_bars(db, _bars(D[:3], [100.0, 110.0, 99.0]))
    assert db.query(MarketIndex).count() == 3
    r = _rets(db)
    assert r[D[0]] is None and r[D[1]] == pytest.approx(0.10) and r[D[2]] == pytest.approx(-0.10)


def test_daily_append_uses_previous_close(db):
    upsert_index_bars(db, _bars(D[:2], [100.0, 110.0]))
    upsert_index_bars(db, _bars([D[2]], [121.0]))  # next day's pipeline run
    assert _rets(db)[D[2]] == pytest.approx(0.10)


def test_rerun_same_day_overwrites_and_updates_following_day(db):
    upsert_index_bars(db, _bars(D[:3], [100.0, 110.0, 121.0]))
    upsert_index_bars(db, _bars([D[1]], [105.0]))  # corrected close for D1
    r = _rets(db)
    assert r[D[1]] == pytest.approx(0.05) and r[D[2]] == pytest.approx(121 / 105 - 1)


def test_backfill_older_data_fixes_first_return(db):
    upsert_index_bars(db, _bars(D[2:], [100.0, 101.0, 102.0]))
    assert _rets(db)[D[2]] is None
    upsert_index_bars(db, _bars(D[:2], [90.0, 95.0]))
    r = _rets(db)
    assert r[D[1]] == pytest.approx(95 / 90 - 1) and r[D[2]] == pytest.approx(100 / 95 - 1)
    assert r[D[3]] == pytest.approx(0.01)


def test_rejects_null_close_and_lookup(db):
    with pytest.raises(ValueError):
        upsert_index_bars(db, _bars([D[0]], [float("nan")]))
    db.rollback()
    upsert_index_bars(db, _bars(D, [100.0, 101.0, 102.0, 103.0, 104.0]))
    got = index_returns(db, [D[0], D[2], dt.date(2026, 12, 1)])
    assert set(got) == {D[2]}  # D0 has no return (first bar), Dec 1 has no data
