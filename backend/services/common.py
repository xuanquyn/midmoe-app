"""Shared helpers for services: domain errors and date resolution."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.db.models import Prediction, Ticker


class ServiceError(Exception):
    """Base class; `message` is user-facing (Vietnamese)."""

    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFoundError(ServiceError):
    status_code = 404


class BadRequestError(ServiceError):
    status_code = 400


class DataIntegrityError(ServiceError):
    """Stored data is malformed (server-side problem)."""

    status_code = 500


def latest_prediction_date(db: Session) -> dt.date | None:
    return db.scalar(select(func.max(Prediction.date)))


def resolve_prediction_date(db: Session, date: dt.date | None) -> dt.date:
    """None -> latest date with predictions. A date without predictions (weekend, holiday) falls back to the
    closest earlier date that has them."""
    if date is None:
        resolved = latest_prediction_date(db)
        if resolved is None:
            raise NotFoundError("Chưa có dữ liệu dự đoán nào.")
        return resolved
    resolved = db.scalar(select(func.max(Prediction.date)).where(Prediction.date <= date))
    if resolved is None:
        raise NotFoundError(f"Không có dữ liệu dự đoán vào hoặc trước ngày {date.isoformat()}.")
    return resolved


def previous_prediction_date(db: Session, date: dt.date) -> dt.date | None:
    return db.scalar(select(func.max(Prediction.date)).where(Prediction.date < date))


def get_ticker(db: Session, ticker: str) -> Ticker:
    obj = db.get(Ticker, ticker.strip().upper())
    if obj is None:
        raise NotFoundError(f"Mã {ticker.upper()} không nằm trong danh sách 78 mã mô hình theo dõi.")
    return obj
