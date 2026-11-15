"""Per-ticker data: prices + indicators, rank history, explanation, news."""
from __future__ import annotations

import datetime as dt

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.db.models import Explanation, News, Prediction, Price
from backend.schemas import ExplanationOut, ExplanationPayload, NewsItem, PriceBar, RankPoint, StockOut
from backend.services.common import (
    BadRequestError,
    NotFoundError,
    DataIntegrityError,
    get_ticker,
    latest_prediction_date,
    resolve_prediction_date,
)

DEFAULT_LOOKBACK_DAYS = 180  # calendar days when `from` is omitted


def get_stock(db: Session, ticker: str, date_from: dt.date | None = None, date_to: dt.date | None = None) -> StockOut:
    t = get_ticker(db, ticker)
    if date_to is None:
        date_to = db.scalar(select(func.max(Price.date)).where(Price.ticker == t.ticker)) or dt.date.today()
    if date_from is None:
        date_from = date_to - dt.timedelta(days=DEFAULT_LOOKBACK_DAYS)
    if date_from > date_to:
        raise BadRequestError("Ngày bắt đầu phải trước ngày kết thúc.")

    prices = db.scalars(
        select(Price).where(Price.ticker == t.ticker, Price.date.between(date_from, date_to)).order_by(Price.date)
    ).all()
    ranks = db.execute(
        select(Prediction.date, Prediction.rank, Prediction.score)
        .where(Prediction.ticker == t.ticker, Prediction.date.between(date_from, date_to))
        .order_by(Prediction.date)
    ).all()
    return StockOut(
        ticker=t.ticker,
        name=t.name,
        exchange=t.exchange,
        icb_sector=t.icb_sector,
        prices=[PriceBar.model_validate(p) for p in prices],
        rank_history=[RankPoint(date=d, rank=r, score=s) for d, r, s in ranks],
    )


def get_explanation(db: Session, ticker: str, date: dt.date | None = None) -> ExplanationOut:
    t = get_ticker(db, ticker)
    d = resolve_prediction_date(db, date)
    row = db.get(Explanation, (d, t.ticker))
    if row is None:
        raise NotFoundError(f"Chưa có giải thích cho mã {t.ticker} ngày {d.isoformat()}.")
    try:
        payload = ExplanationPayload.model_validate(row.payload)
    except ValidationError as exc:
        raise DataIntegrityError(f"Dữ liệu giải thích của {t.ticker} ngày {d.isoformat()} không hợp lệ.") from exc
    return ExplanationOut(ticker=t.ticker, date=d, payload=payload)


def get_news(db: Session, ticker: str, days: int = 7, as_of: dt.date | None = None) -> list[NewsItem]:
    """News of the last `days` calendar days up to `as_of` (default: latest prediction date, else today)."""
    t = get_ticker(db, ticker)
    end = as_of or latest_prediction_date(db) or dt.date.today()
    start = end - dt.timedelta(days=days - 1)
    rows = db.scalars(
        select(News)
        .where(News.ticker == t.ticker, News.date.between(start, end))
        .order_by(News.date.desc(), News.published_at.desc(), News.id.desc())
    ).all()
    return [NewsItem.model_validate(n) for n in rows]
