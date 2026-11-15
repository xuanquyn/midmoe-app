import json

import pytest

from assistant.tools import TOOLS, PageContext, execute_tool, tool_specs
from backend.db.demo import seed_demo


@pytest.fixture
def demo(db):
    return seed_demo(db)


CTX = PageContext()


def test_specs_are_clean_json_schema():
    specs = {s.name: s for s in tool_specs()}
    assert set(specs) == {
        "get_recommendations",
        "get_stock_explanation",
        "get_stock_prices",
        "get_stock_news",
        "compare_stocks",
        "get_model_performance",
        "get_model_info",
    }
    for s in specs.values():
        assert s.parameters["type"] == "object" and s.parameters["additionalProperties"] is False
        assert "title" not in json.dumps(s.parameters)
        assert s.description
    assert specs["get_stock_explanation"].parameters["required"] == ["ticker"]


def test_recommendations(db, demo):
    res, ok = execute_tool(db, "get_recommendations", {"k": 3}, CTX)
    assert ok and res["date"] == demo["pred_dates"][-1].isoformat()
    assert [i["rank"] for i in res["items"]] == [1, 2, 3] and res["n_tickers_total"] == 12
    assert res["items"][0]["confidence"].endswith("mô hình đồng thuận xếp vào top-5")


def test_page_date_is_default(db, demo):
    d = demo["pred_dates"][3]
    res, _ = execute_tool(db, "get_recommendations", {}, PageContext(date=d))
    assert res["date"] == d.isoformat()


def test_explanation_order_and_units(db, demo):
    top = execute_tool(db, "get_recommendations", {"k": 1}, CTX)[0]["items"][0]["ticker"]
    res, ok = execute_tool(db, "get_stock_explanation", {"ticker": top.lower()}, CTX)
    assert ok and res["rank"] == 1
    deltas = [s["delta_score"] for s in res["source_contributions"]]
    assert deltas == sorted(deltas, reverse=True)  # largest contributor first
    assert res["source_contributions"][0]["source"] == "thị trường (market cube)"
    assert res["features"]["rsi"]["percentile_among_tickers"] == 93  # 0.93 -> percent


def test_errors_are_returned_not_raised(db, demo):
    res, ok = execute_tool(db, "get_stock_explanation", {"ticker": "AAPL"}, CTX)
    assert not ok and "không nằm trong danh sách" in res["error"]
    res, ok = execute_tool(db, "get_recommendations", {"k": 0}, CTX)
    assert not ok and res["error"] == "Tham số không hợp lệ"
    res, ok = execute_tool(db, "get_recommendations", {"bogus": 1}, CTX)
    assert not ok
    res, ok = execute_tool(db, "nope", {}, CTX)
    assert not ok and "Không có công cụ" in res["error"]


def test_prices_and_news(db, demo):
    t = demo["tickers"][0]
    res, ok = execute_tool(db, "get_stock_prices", {"ticker": t, "days": 10}, CTX)
    assert ok and res["n_sessions"] == 10 and len(res["sessions"]) == 10
    last = demo["prices"][demo["prices"].ticker == t].iloc[-1]
    assert res["summary"]["last_close"] == pytest.approx(round(last["close"], 2))
    assert res["sessions"][-1]["ret_1d_pct"] == pytest.approx(round(last["ret_1d"] * 100, 2))
    # page date caps the window
    d = demo["pred_dates"][5]
    res, _ = execute_tool(db, "get_stock_prices", {"ticker": t, "days": 5}, PageContext(date=d))
    assert res["to"] == d.isoformat()

    res, ok = execute_tool(db, "get_stock_news", {"ticker": t, "days": 7}, CTX)
    assert ok and res["n_articles"] == 4 and res["articles"][0]["url"].startswith("https://cafef.vn/")


def test_compare(db, demo):
    a, b = demo["tickers"][0], demo["tickers"][1]
    res, ok = execute_tool(db, "compare_stocks", {"tickers": [b, a.lower(), "AAPL"]}, CTX)
    assert ok and {s["ticker"] for s in res["stocks"]} == {a, b}
    assert res["not_in_universe"] == ["AAPL"]
    ranks = [s["rank"] for s in res["stocks"]]
    assert ranks == sorted(ranks)
    res, ok = execute_tool(db, "compare_stocks", {"tickers": ["AAPL", "TSLA"]}, CTX)
    assert not ok


def test_performance_and_info(db, demo):
    res, ok = execute_tool(db, "get_model_performance", {"period": "2025"}, CTX)
    assert ok and res["from"].startswith("2025") and res["to"].startswith("2025")
    assert set(res["top_k"]) == {"sharpe_daily", "total_return_pct", "max_drawdown_pct", "ndcg_at_5"}
    assert res["vnindex"] is not None and "TỔNG lợi suất" in res["conventions"]
    res, ok = execute_tool(db, "get_model_performance", {"period": "live"}, CTX)
    assert not ok  # demo data is all in 2025 -> no live predictions yet
    res, ok = execute_tool(db, "get_model_info", {}, CTX)
    assert ok and res["paper_results_vn_2025"]["sharpe_daily"] == 0.041
    assert any("T+2" in x for x in res["limitations"])


def test_every_tool_has_a_label():
    assert all(t.label.startswith("Đang") for t in TOOLS.values())
