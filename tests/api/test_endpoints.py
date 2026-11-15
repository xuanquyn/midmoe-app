import datetime as dt

import numpy as np
import pytest

from backend.db.models import Explanation
from midmoe.metrics import compute_metrics, daily_ret, max_drawdown
from backend.db.demo import seed_demo


@pytest.fixture
def demo(db):
    return seed_demo(db)


# ------------------------------------------------------------------ recommendations / rankings
def test_recommendations_latest(client, demo):
    r = client.get("/api/recommendations?k=5")
    assert r.status_code == 200
    body = r.json()
    last, prev = demo["pred_dates"][-1], demo["pred_dates"][-2]
    assert body["date"] == last.isoformat() and body["k"] == 5
    assert [it["rank"] for it in body["items"]] == [1, 2, 3, 4, 5]
    top = body["items"][0]
    assert top["reason"].startswith("Xếp hạng 1/12")
    assert top["name"] and top["icb_sector"] in demo["sector_of"].values()
    # rank_change = previous rank - current rank
    prev_rank = client.get(f"/api/rankings?date={prev}").json()["items"]
    prev_map = {it["ticker"]: it["rank"] for it in prev_rank}
    assert top["rank_change"] == prev_map[top["ticker"]] - 1


def test_recommendations_weekend_falls_back_to_friday(client, demo):
    d = next(d for d in demo["pred_dates"] if d.weekday() == 4)  # a Friday
    sat = d + dt.timedelta(days=1)
    assert client.get(f"/api/recommendations?date={sat}").json()["date"] == d.isoformat()


def test_recommendations_errors(client, demo):
    before = demo["pred_dates"][0] - dt.timedelta(days=1)
    r = client.get(f"/api/recommendations?date={before}")
    assert r.status_code == 404 and "Không có dữ liệu" in r.json()["detail"]
    assert client.get("/api/recommendations?k=79").status_code == 422


def test_recommendations_empty_db(client):
    r = client.get("/api/recommendations")
    assert r.status_code == 404 and r.json()["detail"] == "Chưa có dữ liệu dự đoán nào."


def test_rankings_all_and_sector(client, demo):
    body = client.get("/api/rankings").json()
    assert len(body["items"]) == 12
    assert [it["rank"] for it in body["items"]] == list(range(1, 13))
    first_day = client.get(f"/api/rankings?date={demo['pred_dates'][0]}").json()
    assert all(it["rank_change"] is None for it in first_day["items"])

    sector = "Công nghệ"
    sub = client.get("/api/rankings", params={"sector": sector}).json()["items"]
    assert sub and all(it["icb_sector"] == sector for it in sub)
    assert len(sub) == sum(1 for s in demo["sector_of"].values() if s == sector)
    assert client.get("/api/rankings", params={"sector": "Không tồn tại"}).status_code == 404
    assert client.get("/api/sectors").json() == sorted(set(demo["sector_of"].values()))


# ------------------------------------------------------------------ stocks
def test_stock_detail(client, demo):
    t = demo["tickers"][3]
    r = client.get(f"/api/stocks/{t.lower()}")  # case-insensitive
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == t and body["exchange"] == "HOSE"
    dates = [p["date"] for p in body["prices"]]
    assert dates == sorted(dates) and dates[-1] == demo["prices"]["date"].max().isoformat()
    last_bar = body["prices"][-1]
    expected = demo["prices"][(demo["prices"].ticker == t)].iloc[-1]
    assert last_bar["rsi"] == pytest.approx(expected["rsi"]) and last_bar["macd"] == pytest.approx(expected["macd"])
    assert len(body["rank_history"]) == len(demo["pred_dates"])


def test_stock_range_and_errors(client, demo):
    t = demo["tickers"][0]
    d0, d1 = demo["pred_dates"][5], demo["pred_dates"][9]
    body = client.get(f"/api/stocks/{t}?from={d0}&to={d1}").json()
    assert body["prices"][0]["date"] == d0.isoformat() and body["prices"][-1]["date"] == d1.isoformat()
    assert len(body["rank_history"]) == 5
    assert client.get(f"/api/stocks/{t}?from={d1}&to={d0}").status_code == 400
    r = client.get("/api/stocks/AAPL")
    assert r.status_code == 404 and "không nằm trong danh sách" in r.json()["detail"]


def test_explanation(client, db, demo):
    top = client.get("/api/recommendations?k=1").json()["items"][0]["ticker"]
    r = client.get(f"/api/stocks/{top}/explanation")
    assert r.status_code == 200
    p = r.json()["payload"]
    assert p["rank"] == 1 and set(p["sources"]) == {"market", "news", "graph", "timeseries"}

    bottom = client.get("/api/rankings").json()["items"][-1]["ticker"]
    assert client.get(f"/api/stocks/{bottom}/explanation").status_code == 404

    row = db.get(Explanation, (demo["pred_dates"][-1], top))
    row.payload = {"rank": "bad"}
    db.commit()
    assert client.get(f"/api/stocks/{top}/explanation").status_code == 500


def test_news_window(client, demo):
    t = demo["tickers"][0]
    news7 = client.get(f"/api/stocks/{t}/news?days=7").json()
    assert [n["title"] for n in news7] == ["Tin 0", "Tin 2", "Tin 4", "Tin 6"]
    assert len(client.get(f"/api/stocks/{t}/news?days=30").json()) == 6
    assert client.get(f"/api/stocks/{demo['tickers'][1]}/news").json() == []


# ------------------------------------------------------------------ backtest
def _expected_backtest(demo, k=5):
    prices = demo["prices"]
    ret = prices.pivot(index="date", columns="ticker", values="ret_1d")[demo["tickers"]]
    sessions = list(ret.index)
    P, Y = [], []
    for d in demo["pred_dates"]:
        nxt = sessions[sessions.index(d) + 1] if sessions.index(d) + 1 < len(sessions) else None
        if nxt is None:
            continue
        P.append(demo["scores"][d])
        Y.append(ret.loc[nxt].to_numpy())
    return np.stack(P), np.stack(Y)


def test_backtest_matches_paper_functions(client, demo):
    body = client.get("/api/backtest?k=5").json()
    P, Y = _expected_backtest(demo)
    assert body["n_days"] == len(Y) == len(demo["pred_dates"])  # factory adds one session after the last prediction
    topk = daily_ret(P, Y, 5)
    m = compute_metrics(list(P), list(Y), k=5)
    got = body["metrics"]["topk"]
    assert got["SR"] == pytest.approx(m["SR"]) and got["IRR"] == pytest.approx(m["IRR"])
    assert got["nDCG5"] == pytest.approx(m["nDCG@5"]) and got["MaxDD"] == pytest.approx(max_drawdown(topk))
    assert [p["topk"] for p in body["series"]] == pytest.approx(list(topk))
    assert body["series"][-1]["topk_cum"] == pytest.approx(topk.sum())
    assert set(body["metrics"]) == {"topk", "random", "vnindex"}


def test_backtest_vnindex_is_k_times_next_session_return(client, demo):
    k = 5
    body = client.get(f"/api/backtest?k={k}").json()
    bars = demo["index_bars"].set_index("date")["close"]
    sessions = list(bars.index)
    expected = []
    for d in demo["pred_dates"]:
        j = sessions.index(d) + 1
        expected.append(k * (bars.iloc[j] / bars.iloc[j - 1] - 1))
    expected = np.array(expected)
    assert body["vnindex_coverage"] == 1.0
    assert [p["vnindex"] for p in body["series"]] == pytest.approx(list(expected))
    assert body["series"][-1]["vnindex_cum"] == pytest.approx(expected.sum())
    vm = body["metrics"]["vnindex"]
    assert vm["IRR"] == pytest.approx(expected.sum())
    assert vm["SR"] == pytest.approx(expected.mean() / (expected.std() + 1e-8))
    assert vm["MaxDD"] == pytest.approx(max_drawdown(expected)) and vm["nDCG5"] is None


def test_backtest_partial_vnindex_coverage_returns_nulls(client, db, demo):
    from backend.db.models import MarketIndex

    assert client.get("/api/backtest").json()["vnindex_coverage"] == 1.0  # fills the cache
    label_day = demo["pred_dates"][5 + 1]  # label session of pred_dates[5]
    db.delete(db.get(MarketIndex, ("VNINDEX", label_day)))
    db.commit()
    body = client.get("/api/backtest").json()  # cache must be invalidated by the index change
    assert body["vnindex_coverage"] == pytest.approx(1 - 1 / body["n_days"])
    assert all(p["vnindex"] is None and p["vnindex_cum"] is None for p in body["series"])
    assert "vnindex" not in body["metrics"]


def test_backtest_without_index_data(client, db):
    seed_demo(db, with_index=False)
    body = client.get("/api/backtest").json()
    assert body["vnindex_coverage"] == 0.0 and set(body["metrics"]) == {"topk", "random"}


def test_backtest_range_and_errors(client, demo):
    d0, d1 = demo["pred_dates"][10], demo["pred_dates"][19]
    body = client.get(f"/api/backtest?from={d0}&to={d1}&k=3").json()
    assert body["from_date"] == d0.isoformat() and body["to_date"] == d1.isoformat() and body["n_days"] == 10
    assert client.get(f"/api/backtest?from={d1}&to={d0}").status_code == 400
    assert client.get("/api/backtest?k=20").status_code == 400  # only 12 tickers
    assert client.get("/api/backtest?from=2030-01-01").status_code == 404


def test_backtest_cache_invalidated_by_new_pipeline_run(client, db, demo):
    from backend.db.models import PipelineRun, Prediction, RunStatus

    before = client.get("/api/backtest?k=1").json()
    d = demo["pred_dates"][0]
    top = db.query(Prediction).filter_by(date=d, rank=1).one()
    worst = db.query(Prediction).filter_by(date=d, rank=12).one()
    top.score, worst.score = worst.score, top.score + 1.0  # re-run of the same day changes scores only
    db.add(PipelineRun(run_date=d, status=RunStatus.success))
    db.commit()
    after = client.get("/api/backtest?k=1").json()
    assert after["series"][0]["topk"] != before["series"][0]["topk"]


def test_max_drawdown():
    assert max_drawdown(np.array([0.1, -0.05, -0.1, 0.2])) == pytest.approx(0.15)
    assert max_drawdown(np.array([-0.02, 0.01])) == pytest.approx(0.02)
    assert max_drawdown(np.array([])) == 0.0
