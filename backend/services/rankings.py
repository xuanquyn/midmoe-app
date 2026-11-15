"""Recommendations (top-k) and full rankings for a day."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Explanation, Prediction, Ticker
from backend.schemas import RankingItem, RankingsOut, RecommendationItem, RecommendationsOut
from backend.services.common import NotFoundError, previous_prediction_date, resolve_prediction_date


def _ranking_rows(db: Session, date: dt.date, sector: str | None = None, limit: int | None = None):
    q = (
        select(Prediction, Ticker)
        .join(Ticker, Ticker.ticker == Prediction.ticker)
        .where(Prediction.date == date)
        .order_by(Prediction.rank)
    )
    if sector:
        q = q.where(Ticker.icb_sector == sector)
    if limit:
        q = q.limit(limit)
    return db.execute(q).all()


def _prev_ranks(db: Session, date: dt.date, tickers: list[str]) -> dict[str, int]:
    prev = previous_prediction_date(db, date)
    if prev is None or not tickers:
        return {}
    rows = db.execute(
        select(Prediction.ticker, Prediction.rank).where(Prediction.date == prev, Prediction.ticker.in_(tickers))
    )
    return dict(rows.tuples().all())


def _item(pred: Prediction, tick: Ticker, prev_rank: int | None) -> dict:
    return {
        "ticker": pred.ticker,
        "name": tick.name,
        "icb_sector": tick.icb_sector,
        "score": pred.score,
        "score_std": pred.score_std,
        "rank": pred.rank,
        "top5_votes": pred.top5_votes,
        "rank_change": None if prev_rank is None else prev_rank - pred.rank,
    }


def get_rankings(db: Session, date: dt.date | None = None, sector: str | None = None) -> RankingsOut:
    d = resolve_prediction_date(db, date)
    rows = _ranking_rows(db, d, sector)
    if sector and not rows:
        raise NotFoundError(f"Không có mã nào thuộc ngành '{sector}'.")
    prev = _prev_ranks(db, d, [p.ticker for p, _ in rows])
    return RankingsOut(
        date=d, sector=sector, items=[RankingItem(**_item(p, t, prev.get(p.ticker))) for p, t in rows]
    )


def get_recommendations(db: Session, date: dt.date | None = None, k: int = 5) -> RecommendationsOut:
    d = resolve_prediction_date(db, date)
    rows = _ranking_rows(db, d, limit=k)
    tickers = [p.ticker for p, _ in rows]
    prev = _prev_ranks(db, d, tickers)
    summaries = dict(
        db.execute(
            select(Explanation.ticker, Explanation.payload).where(Explanation.date == d, Explanation.ticker.in_(tickers))
        )
        .tuples()
        .all()
    )
    items = [
        RecommendationItem(
            **_item(p, t, prev.get(p.ticker)),
            reason=(summaries.get(p.ticker) or {}).get("summary"),
        )
        for p, t in rows
    ]
    return RecommendationsOut(date=d, k=k, items=items)


def list_sectors(db: Session) -> list[str]:
    return [s for s in db.scalars(select(Ticker.icb_sector).distinct().order_by(Ticker.icb_sector)) if s]
