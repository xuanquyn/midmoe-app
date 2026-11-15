"""SYNTHETIC demo data (tickers M00..Mxx, random-walk prices, random scores) - internally consistent, used by the
API tests, the p95 latency check and local UI development before the real pipeline has run.

    python -m backend.db.demo --tickers 78 --days 480      # into DATABASE_URL (refuses a DB that has tickers)

Never load this into a database used for the demo / report: none of these numbers are model output."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import insert
from sqlalchemy.orm import Session

from backend.db.models import Explanation, News, PipelineRun, Prediction, Price, RunStatus, Ticker
from backend.services.market_index import upsert_index_bars
from midmoe.features import add_indicators_all
from midmoe.predict import ranks_from_scores

SECTORS = ["Ngân hàng", "Bất động sản", "Công nghệ"]
INDICATORS = ["ret_1d", "ret_5d", "ret_10d", "rsi", "macd", "close_open_ratio", "high_low_ratio"]


def make_payload(rank: int, score: float, votes: int, n: int, sector: str) -> dict:
    return {
        "rank": rank,
        "score": score,
        "top5_votes": votes,
        "score_std": 0.003,
        "sources": {
            "market": {"delta": 0.004, "rank_without": min(n, rank + 5)},
            "news": {"delta": 0.002, "rank_without": min(n, rank + 2)},
            "graph": {"delta": -0.0005, "rank_without": rank},
            "timeseries": {"delta": 0.003, "rank_without": min(n, rank + 4)},
        },
        "features": {"rsi": {"value": 72.1, "pct": 0.93, "label": "quá mua"}},
        "sector": {"name": sector, "rank_in_sector": 1, "n_in_sector": 4, "top": []},
        "news": [],
        "summary": f"Xếp hạng {rank}/{n}, {votes}/10 mô hình đồng thuận; yếu tố thị trường đóng góp nhiều nhất.",
    }


def seed_demo(
    db: Session, n_tickers: int = 12, n_days: int = 40, start: str = "2025-01-02", seed: int = 0, with_index: bool = True,
    explain_all: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    tickers = [f"M{i:02d}" for i in range(n_tickers)]
    sector_of = {t: SECTORS[i * len(SECTORS) // n_tickers] for i, t in enumerate(tickers)}
    db.execute(
        insert(Ticker),
        [
            {"ticker": t, "name": f"Công ty {t}", "exchange": "HOSE", "icb_sector": sector_of[t], "sort_order": i}
            for i, t in enumerate(tickers)
        ],
    )
    db.flush()

    # prices: 60 warm-up sessions + n_days prediction sessions + 1 extra session (label of the last-but-one day)
    dates = pd.bdate_range(start, periods=n_days + 61)
    rows = []
    for t in tickers:
        close = 20 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
        open_ = close * (1 + rng.normal(0, 0.005, len(dates)))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates.date,
                    "ticker": t,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.01,
                    "low": np.minimum(open_, close) * 0.99,
                    "close": close,
                    "volume": rng.integers(1e5, 1e7, len(dates)),
                }
            )
        )
    prices = add_indicators_all(pd.concat(rows))
    price_records = prices.astype(object).where(prices.notna(), None).to_dict("records")
    db.execute(insert(Price), price_records)

    index_bars = None
    if with_index:
        idx_close = 1200 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        index_bars = pd.DataFrame({"date": dates.date, "open": idx_close, "high": idx_close, "low": idx_close, "close": idx_close})
        upsert_index_bars(db, index_bars, commit=False)

    pred_dates = list(dates.date[60 : 60 + n_days])
    pred_rows, expl_rows = [], []
    scores_by_date = {}
    for d in pred_dates:
        per_seed = rng.normal(0, 0.01, (10, n_tickers))
        score = per_seed.mean(0)
        rank = ranks_from_scores(score)
        votes = (ranks_from_scores(per_seed) <= 5).sum(0)
        scores_by_date[d] = score
        for i, t in enumerate(tickers):
            pred_rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "score": float(score[i]),
                    "score_std": float(per_seed[:, i].std()),
                    "rank": int(rank[i]),
                    "top5_votes": int(votes[i]),
                    "model_version": "test",
                }
            )
            if explain_all or rank[i] <= 5:
                expl_rows.append(
                    {
                        "date": d,
                        "ticker": t,
                        "payload": make_payload(int(rank[i]), float(score[i]), int(votes[i]), n_tickers, sector_of[t]),
                    }
                )
    db.execute(insert(Prediction), pred_rows)
    db.execute(insert(Explanation), expl_rows)

    last = pred_dates[-1]
    db.execute(
        insert(News),
        [
            {"ticker": tickers[0], "date": last - dt.timedelta(days=k), "title": f"Tin {k}", "url": f"https://cafef.vn/{k}", "source": "cafef"}
            for k in range(0, 12, 2)
        ],
    )
    db.add(PipelineRun(run_date=last, status=RunStatus.success, n_tickers=n_tickers, n_news=6))
    db.commit()
    return {
        "tickers": tickers,
        "sector_of": sector_of,
        "pred_dates": pred_dates,
        "prices": prices,
        "scores": scores_by_date,
        "index_bars": index_bars,
    }


def main() -> None:
    import argparse

    from sqlalchemy import func, select

    from backend.db.session import get_sessionmaker

    ap = argparse.ArgumentParser(description="Load SYNTHETIC demo data (for local UI development only).")
    ap.add_argument("--tickers", type=int, default=78)
    ap.add_argument("--days", type=int, default=480, help="prediction sessions (plus 60 warm-up sessions)")
    ap.add_argument("--start", default="2024-10-01", help="first price date (warm-up starts here)")
    args = ap.parse_args()
    with get_sessionmaker()() as db:
        if db.scalar(select(func.count()).select_from(Ticker)):
            raise SystemExit("DB already has tickers: refusing to mix synthetic data with real data.")
        info = seed_demo(db, n_tickers=args.tickers, n_days=args.days, start=args.start, explain_all=True)
    print(f"Synthetic data: {len(info['tickers'])} tickers, {info['pred_dates'][0]} -> {info['pred_dates'][-1]}")


if __name__ == "__main__":
    main()
