"""M4 check: p95 latency of GET endpoints < 300 ms on a backfill-sized DB (78 tickers x ~470 sessions,
i.e. 2025-01 -> 2026-11). Runs on SQLite; Postgres with the same indexes should be at least as fast."""
import time

import numpy as np
import pytest

from backend.db.demo import seed_demo

N_TICKERS, N_DAYS, N_REQ = 78, 470, 20
P95_MS = 300


@pytest.fixture
def big(db):
    return seed_demo(db, n_tickers=N_TICKERS, n_days=N_DAYS, seed=1)


def test_get_p95_under_300ms(client, big):
    t = big["tickers"][10]
    mid = big["pred_dates"][N_DAYS // 2]
    paths = [
        "/api/health",
        "/api/dates",
        "/api/recommendations?k=5",
        f"/api/recommendations?date={mid}",
        "/api/rankings",
        f"/api/rankings?sector={big['sector_of'][t]}",
        f"/api/stocks/{t}",
        f"/api/stocks/{t}?from=2025-01-01",
        f"/api/stocks/{t}/explanation?date={mid}",
        f"/api/stocks/{big['tickers'][0]}/news?days=7",
        "/api/backtest?k=5",
        "/api/backtest?from=2025-01-01&to=2025-12-31&k=5",
    ]
    report = {}
    for p in paths:
        assert client.get(p).status_code in (200, 404), p  # warm-up
        times = []
        for _ in range(N_REQ):
            start = time.perf_counter()
            client.get(p)
            times.append((time.perf_counter() - start) * 1000)
        report[p] = float(np.percentile(times, 95))
    print("\n" + "\n".join(f"{v:7.1f} ms  {k}" for k, v in sorted(report.items(), key=lambda x: -x[1])))
    slow = {k: v for k, v in report.items() if v >= P95_MS}
    assert not slow, f"p95 >= {P95_MS} ms: {slow}"
