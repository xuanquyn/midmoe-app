"""SQLAlchemy 2 models — PLAN.md section 5.

Portable between PostgreSQL 16 (prod) and SQLite (dev/tests): JSON columns become JSONB on Postgres,
UUIDs use the native type on Postgres and CHAR(32) on SQLite.
"""
from __future__ import annotations

import enum
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")

# Deterministic constraint names so Alembic migrations are stable across backends.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class RunStatus(str, enum.Enum):
    success = "success"
    degraded = "degraded"
    failed = "failed"
    skipped = "skipped"


class Ticker(Base):
    __tablename__ = "tickers"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    exchange: Mapped[str | None] = mapped_column(String(16))
    icb_sector: Mapped[str | None] = mapped_column(String(128), index=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False)


class Price(Base):
    __tablename__ = "prices"

    ticker: Mapped[str] = mapped_column(ForeignKey("tickers.ticker"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    # raw (un-normalised) indicators
    ret_1d: Mapped[float | None] = mapped_column(Float)
    ret_5d: Mapped[float | None] = mapped_column(Float)
    ret_10d: Mapped[float | None] = mapped_column(Float)
    rsi: Mapped[float | None] = mapped_column(Float)
    macd: Mapped[float | None] = mapped_column(Float)
    close_open_ratio: Mapped[float | None] = mapped_column(Float)
    high_low_ratio: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_prices_date", "date"),)


class MarketIndex(Base):
    """Daily bars of market indices (VNINDEX; VN30 etc. can share the table). ret_1d = close.pct_change(1),
    computed on write by backend.services.market_index.upsert_index_bars."""

    __tablename__ = "market_index"

    index_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    ret_1d: Mapped[float | None] = mapped_column(Float)


class News(Base):
    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("tickers.ticker"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    source: Mapped[str | None] = mapped_column(String(32))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_news_ticker_date", "ticker", "date"),)


class Prediction(Base):
    __tablename__ = "predictions"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("tickers.ticker"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    score_std: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    top5_votes: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_predictions_date_rank", "date", "rank"),
        Index("ix_predictions_ticker_date", "ticker", "date"),
    )


class Explanation(Base):
    __tablename__ = "explanations"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("tickers.ticker"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONType, nullable=False)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus, name="run_status"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    n_tickers: Mapped[int | None] = mapped_column(Integer)
    n_news: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.id"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool_calls: Mapped[list | dict | None] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    session: Mapped[ChatSession] = relationship(back_populates="messages")
