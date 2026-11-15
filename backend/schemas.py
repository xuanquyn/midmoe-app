"""Pydantic v2 schemas for the API (and the `explanations.payload` contract shared with midmoe/explain.py)."""
from __future__ import annotations

import uuid
import datetime as dt
from typing import Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from typing_extensions import Annotated


def _ensure_utc(v: dt.datetime) -> dt.datetime:
    # SQLite drops tzinfo; every timestamp is written in UTC (models.utcnow), so naive means UTC.
    return v.replace(tzinfo=dt.timezone.utc) if v.tzinfo is None else v


UTCDateTime = Annotated[dt.datetime, AfterValidator(_ensure_utc)]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------- health / dates
class PipelineRunOut(ORMModel):
    id: int
    run_date: dt.date
    status: Literal["success", "degraded", "failed", "skipped"]
    started_at: UTCDateTime
    finished_at: UTCDateTime | None = None
    n_tickers: int | None = None
    n_news: int | None = None
    error: str | None = None


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    db: Literal["ok", "error"]
    db_error: str | None = None
    last_run: PipelineRunOut | None = None


class DatesOut(BaseModel):
    dates: list[dt.date]
    latest: dt.date | None


# ---------------------------------------------------------------- explanation payload (PLAN.md §5)
SourceName = Literal["market", "news", "graph", "timeseries"]


class SourceContribution(BaseModel):
    delta: float = Field(description="score_full - score_ablated, averaged over seeds")
    rank_without: int = Field(ge=1)


class FeatureSnapshot(BaseModel):
    value: float | None
    pct: float | None = Field(default=None, ge=0, le=1, description="percentile among the 78 tickers")
    label: str | None = None


class SectorContext(BaseModel):
    name: str
    rank_in_sector: int = Field(ge=1)
    n_in_sector: int = Field(ge=1)
    top: list[str] = Field(default_factory=list, description="top-3 tickers in the sector")


class NewsRef(BaseModel):
    date: dt.date
    title: str
    url: str


class ExplanationPayload(BaseModel):
    rank: int = Field(ge=1)
    score: float
    top5_votes: int = Field(ge=0)
    score_std: float = Field(ge=0)
    n_models: int = 10
    sources: dict[SourceName, SourceContribution]
    features: dict[str, FeatureSnapshot]
    sector: SectorContext
    news: list[NewsRef] = Field(default_factory=list)
    summary: str


# ---------------------------------------------------------------- recommendations / rankings
class RankingItem(BaseModel):
    ticker: str
    name: str | None = None
    icb_sector: str | None = None
    score: float
    score_std: float
    rank: int
    top5_votes: int
    rank_change: int | None = Field(default=None, description="previous rank - current rank (positive = up)")


class RecommendationItem(RankingItem):
    reason: str | None = Field(default=None, description="one-line reason (explanation summary)")


class RecommendationsOut(BaseModel):
    date: dt.date
    k: int
    items: list[RecommendationItem]


class RankingsOut(BaseModel):
    date: dt.date
    sector: str | None = None
    items: list[RankingItem]


# ---------------------------------------------------------------- stocks
class PriceBar(ORMModel):
    date: dt.date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int | None
    ret_1d: float | None
    ret_5d: float | None
    ret_10d: float | None
    rsi: float | None
    macd: float | None
    close_open_ratio: float | None
    high_low_ratio: float | None


class RankPoint(BaseModel):
    date: dt.date
    rank: int
    score: float


class StockOut(BaseModel):
    ticker: str
    name: str | None
    exchange: str | None
    icb_sector: str | None
    prices: list[PriceBar]
    rank_history: list[RankPoint]


class ExplanationOut(BaseModel):
    ticker: str
    date: dt.date
    payload: ExplanationPayload


class NewsItem(ORMModel):
    id: int
    ticker: str
    date: dt.date
    title: str
    url: str
    source: str | None
    published_at: UTCDateTime | None


# ---------------------------------------------------------------- backtest
class BacktestPoint(BaseModel):
    """Daily returns are SUMS over k positions (paper convention); vnindex = k x VN-Index return."""

    date: dt.date
    topk: float
    random: float
    vnindex: float | None
    topk_cum: float
    random_cum: float
    vnindex_cum: float | None


class BacktestMetrics(BaseModel):
    SR: float
    IRR: float
    MaxDD: float
    nDCG5: float | None = None


class BacktestOut(BaseModel):
    from_date: dt.date
    to_date: dt.date
    k: int
    n_days: int
    vnindex_coverage: float = Field(
        ge=0, le=1, description="share of label sessions with VN-Index data; VN-Index fields are null unless 1"
    )
    series: list[BacktestPoint]
    metrics: dict[Literal["topk", "random", "vnindex"], BacktestMetrics]


# ---------------------------------------------------------------- chat
class ChatContext(BaseModel):
    ticker: str | None = None
    date: dt.date | None = None


class ChatMessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    tools: list[str] = Field(default_factory=list, description="tools called while producing this answer")
    created_at: UTCDateTime


class ChatRequest(BaseModel):
    session_id: uuid.UUID | None = None
    message: str = Field(min_length=1, max_length=4000)
    context: ChatContext = Field(default_factory=ChatContext)
