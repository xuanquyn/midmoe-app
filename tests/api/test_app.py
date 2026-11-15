from datetime import date, datetime, timezone

import pytest

from backend.db.models import PipelineRun, Prediction, RunStatus, Ticker


def test_health_empty_db(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "db": "ok", "db_error": None, "last_run": None}
    assert r.headers["X-Request-ID"]


def test_health_returns_latest_run(client, db):
    t0 = datetime(2026, 11, 13, 9, 30, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 14, 9, 30, tzinfo=timezone.utc)
    db.add_all(
        [
            PipelineRun(run_date=date(2026, 11, 13), status=RunStatus.success, started_at=t0, n_tickers=78),
            PipelineRun(run_date=date(2026, 11, 14), status=RunStatus.degraded, started_at=t1, error="CafeF timeout"),
        ]
    )
    db.commit()
    run = client.get("/api/health").json()["last_run"]
    assert run["run_date"] == "2026-11-14" and run["status"] == "degraded"


def test_dates(client, db):
    assert client.get("/api/dates").json() == {"dates": [], "latest": None}
    db.add(Ticker(ticker="FPT", sort_order=0))
    db.flush()  # tickers must exist before FK rows (no ORM relationship to order the flush)
    for d in (date(2026, 11, 14), date(2026, 11, 13)):
        db.add(Prediction(date=d, ticker="FPT", score=0.1, score_std=0.01, rank=1, top5_votes=9, model_version="v1"))
    db.commit()
    assert client.get("/api/dates").json() == {"dates": ["2026-11-13", "2026-11-14"], "latest": "2026-11-14"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/recommendations?k=5",
        "/api/rankings",
        "/api/stocks/FPT?from=2026-01-01",
        "/api/stocks/FPT/explanation",
        "/api/stocks/FPT/news?days=7",
        "/api/backtest?from=2025-01-01&to=2025-12-31&k=5",
    ],
)
def test_empty_db_returns_404_with_message(client, path):
    r = client.get(path)
    assert r.status_code == 404
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_chat_validates_body(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    r = client.post("/api/chat", json={"message": "top 5 hôm nay?", "context": {"ticker": "FPT"}})
    assert r.status_code == 200 and "chưa được cấu hình" in r.text  # no LLM configured in tests by default


def test_query_validation(client):
    assert client.get("/api/recommendations?k=0").status_code == 422
    assert client.get("/api/recommendations?date=not-a-date").status_code == 422


def test_openapi_and_cors(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in [
        "/api/health",
        "/api/dates",
        "/api/recommendations",
        "/api/rankings",
        "/api/stocks/{ticker}",
        "/api/stocks/{ticker}/explanation",
        "/api/stocks/{ticker}/news",
        "/api/backtest",
        "/api/chat",
    ]:
        assert p in paths, p
    assert client.get("/docs").status_code == 200
    r = client.options(
        "/api/health", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
