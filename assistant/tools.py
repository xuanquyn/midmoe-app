"""Assistant tools (M5.2). Each tool validates its arguments with Pydantic, calls the backend service layer
directly (no HTTP) and returns a compact JSON-able dict. Returns are expressed in percent (`*_pct`) so the model
does not have to convert units; every number the assistant cites should be copied from these outputs.
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from assistant.llm import ToolSpec
from assistant.prompts import MODEL_INFO
from backend.config import get_settings
from backend.db.models import Price
from backend.services import backtest, rankings, stocks
from backend.services.common import NotFoundError, ServiceError, get_ticker, resolve_prediction_date

SOURCE_LABELS = {"market": "thị trường (market cube)", "news": "tin tức", "graph": "quan hệ ngành", "timeseries": "chuỗi giá"}


@dataclass
class PageContext:
    ticker: str | None = None
    date: dt.date | None = None


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def _pct(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(float(x) * 100, nd)


def _confidence(votes: int, n_models: int = 10) -> str:
    return f"{votes}/{n_models} mô hình đồng thuận xếp vào top-5"


# ---------------------------------------------------------------------------------------------- arguments
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecommendationsArgs(_Args):
    date: dt.date | None = Field(None, description="Ngày (YYYY-MM-DD). Bỏ trống = ngày đang xem hoặc ngày mới nhất.")
    k: int = Field(5, ge=1, le=20, description="Số mã muốn lấy (mặc định 5)")


class TickerDateArgs(_Args):
    ticker: str = Field(..., min_length=1, max_length=16, description="Mã cổ phiếu, ví dụ FPT")
    date: dt.date | None = Field(None, description="Ngày (YYYY-MM-DD). Bỏ trống = ngày đang xem hoặc mới nhất.")


class PricesArgs(_Args):
    ticker: str = Field(..., min_length=1, max_length=16)
    days: int = Field(30, ge=1, le=250, description="Số phiên gần nhất")


class NewsArgs(_Args):
    ticker: str = Field(..., min_length=1, max_length=16)
    days: int = Field(7, ge=1, le=90, description="Số ngày gần nhất")


class CompareArgs(_Args):
    tickers: list[str] = Field(..., min_length=2, max_length=5, description="2–5 mã cần so sánh")
    date: dt.date | None = None


class PerformanceArgs(_Args):
    period: Literal["2025", "live", "all"] = Field(
        "2025", description="'2025' = năm kiểm thử trong paper; 'live' = từ ngày triển khai; 'all' = toàn bộ"
    )
    k: int = Field(5, ge=1, le=20)


class NoArgs(_Args):
    pass


# ---------------------------------------------------------------------------------------------- handlers
def _date(args_date: dt.date | None, ctx: PageContext) -> dt.date | None:
    return args_date or ctx.date


def get_recommendations(db: Session, a: RecommendationsArgs, ctx: PageContext) -> dict:
    rec = rankings.get_recommendations(db, _date(a.date, ctx), a.k)
    return {
        "date": rec.date.isoformat(),
        "n_tickers_total": len(rankings.get_rankings(db, rec.date).items),
        "items": [
            {
                "rank": it.rank,
                "ticker": it.ticker,
                "name": it.name,
                "sector": it.icb_sector,
                "score": _r(it.score, 5),
                "top5_votes": it.top5_votes,
                "confidence": _confidence(it.top5_votes),
                "rank_change_vs_prev_session": it.rank_change,
                "reason": it.reason,
            }
            for it in rec.items
        ],
    }


def _feature_view(features: dict) -> dict:
    out = {}
    for name, f in features.items():
        out[name] = {
            "value": _r(f.value, 4),
            "percentile_among_tickers": _pct(f.pct, 0),
            "label": f.label,
        }
    return out


def get_stock_explanation(db: Session, a: TickerDateArgs, ctx: PageContext) -> dict:
    ticker = a.ticker
    exp = stocks.get_explanation(db, ticker, _date(a.date, ctx))
    p = exp.payload
    n_total = len(rankings.get_rankings(db, exp.date).items)
    srcs = sorted(p.sources.items(), key=lambda kv: kv[1].delta, reverse=True)
    return {
        "ticker": exp.ticker,
        "date": exp.date.isoformat(),
        "rank": p.rank,
        "n_tickers_total": n_total,
        "score": _r(p.score, 5),
        "score_std": _r(p.score_std, 5),
        "top5_votes": p.top5_votes,
        "confidence": _confidence(p.top5_votes, p.n_models),
        "source_contributions": [
            {
                "source": SOURCE_LABELS.get(name, name),
                "delta_score": _r(c.delta, 5),
                "rank_if_removed": c.rank_without,
            }
            for name, c in srcs
        ],
        "features": _feature_view(p.features),
        "sector": p.sector.model_dump(),
        "news": [n.model_dump(mode="json") for n in p.news],
        "summary": p.summary,
    }


def get_stock_prices(db: Session, a: PricesArgs, ctx: PageContext) -> dict:
    t = get_ticker(db, a.ticker)
    q = select(Price).where(Price.ticker == t.ticker)
    if ctx.date:
        q = q.where(Price.date <= ctx.date)
    rows = list(reversed(db.scalars(q.order_by(Price.date.desc()).limit(a.days)).all()))
    if not rows:
        raise NotFoundError(f"Chưa có dữ liệu giá cho mã {t.ticker}.")
    closes = [r.close for r in rows if r.close is not None]
    return {
        "ticker": t.ticker,
        "name": t.name,
        "from": rows[0].date.isoformat(),
        "to": rows[-1].date.isoformat(),
        "n_sessions": len(rows),
        "summary": {
            "last_close": _r(rows[-1].close, 2),
            "change_over_period_pct": _pct(closes[-1] / closes[0] - 1) if len(closes) > 1 and closes[0] else None,
            "highest_close": _r(max(closes), 2),
            "lowest_close": _r(min(closes), 2),
            "last_rsi": _r(rows[-1].rsi, 1),
            "last_macd": _r(rows[-1].macd, 4),
        },
        "sessions": [
            {
                "date": r.date.isoformat(),
                "close": _r(r.close, 2),
                "volume": r.volume,
                "ret_1d_pct": _pct(r.ret_1d),
                "rsi": _r(r.rsi, 1),
            }
            for r in rows
        ],
    }


def get_stock_news(db: Session, a: NewsArgs, ctx: PageContext) -> dict:
    items = stocks.get_news(db, a.ticker, a.days, as_of=ctx.date)
    return {
        "ticker": a.ticker.upper(),
        "days": a.days,
        "n_articles": len(items),
        "articles": [{"date": n.date.isoformat(), "title": n.title, "url": n.url, "source": n.source} for n in items],
    }


def compare_stocks(db: Session, a: CompareArgs, ctx: PageContext) -> dict:
    d = resolve_prediction_date(db, _date(a.date, ctx))
    ranking = {it.ticker: it for it in rankings.get_rankings(db, d).items}
    out, missing = [], []
    for raw in dict.fromkeys(t.strip().upper() for t in a.tickers):
        it = ranking.get(raw)
        if it is None:
            missing.append(raw)
            continue
        entry: dict[str, Any] = {
            "ticker": raw,
            "name": it.name,
            "sector": it.icb_sector,
            "rank": it.rank,
            "score": _r(it.score, 5),
            "top5_votes": it.top5_votes,
            "confidence": _confidence(it.top5_votes),
            "rank_change_vs_prev_session": it.rank_change,
        }
        try:
            exp = stocks.get_explanation(db, raw, d).payload
            top = max(exp.sources.items(), key=lambda kv: kv[1].delta)
            entry["largest_source"] = SOURCE_LABELS.get(top[0], top[0])
            entry["features"] = _feature_view(exp.features)
        except ServiceError:
            price = db.get(Price, (raw, d))
            if price:
                entry["features"] = {
                    "rsi": {"value": _r(price.rsi, 1)},
                    "ret_5d_pct": _pct(price.ret_5d),
                    "ret_1d_pct": _pct(price.ret_1d),
                }
        out.append(entry)
    if not out:
        raise NotFoundError("Không mã nào trong danh sách nằm trong 78 mã mô hình theo dõi.")
    return {
        "date": d.isoformat(),
        "n_tickers_total": len(ranking),
        "stocks": sorted(out, key=lambda e: e["rank"]),
        "not_in_universe": missing,
    }


def get_model_performance(db: Session, a: PerformanceArgs, ctx: PageContext) -> dict:
    live_start = get_settings().live_start
    if a.period == "2025":
        lo, hi, label = dt.date(2025, 1, 1), dt.date(2025, 12, 31), "Năm kiểm thử 2025"
    elif a.period == "live":
        lo, hi, label = live_start, None, f"Live từ {live_start.isoformat()}"
    else:
        lo, hi, label = None, None, "Toàn bộ dữ liệu"
    bt = backtest.run_backtest(db, lo, hi, a.k)

    def m(key: str) -> dict | None:
        x = bt.metrics.get(key)
        if x is None:
            return None
        return {
            "sharpe_daily": _r(x.SR, 3),
            "total_return_pct": _pct(x.IRR),
            "max_drawdown_pct": _pct(x.MaxDD),
            "ndcg_at_5": _r(x.nDCG5, 3),
        }

    return {
        "period": label,
        "from": bt.from_date.isoformat(),
        "to": bt.to_date.isoformat(),
        "n_days": bt.n_days,
        "k": bt.k,
        "top_k": m("topk"),
        "random": m("random"),
        "vnindex": m("vnindex"),
        "conventions": (
            f"Lợi nhuận ngày = TỔNG lợi suất phiên kế tiếp của {bt.k} mã (quy ước của paper); VN-Index được nhân {bt.k} "
            "để cùng thang. Sharpe tính theo ngày, không annualize. Chưa trừ phí giao dịch, chưa mô hình hoá T+2 và biên độ."
        ),
    }


def get_model_info(db: Session, a: NoArgs, ctx: PageContext) -> dict:
    return MODEL_INFO


# ---------------------------------------------------------------------------------------------- registry
@dataclass
class Tool:
    name: str
    description: str
    args_model: type[_Args]
    handler: Callable[[Session, Any, PageContext], dict]
    label: str  # shown in the UI while running ("đang tra cứu ...")

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        return ToolSpec(self.name, self.description, schema)


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "get_recommendations",
            "Lấy top-k mã được mô hình xếp hạng cao nhất trong một ngày, kèm độ đồng thuận và lý do một dòng.",
            RecommendationsArgs,
            get_recommendations,
            "Đang lấy danh sách khuyến nghị",
        ),
        Tool(
            "get_stock_explanation",
            "Giải thích vì sao một mã được xếp hạng như vậy: hạng, độ đồng thuận, đóng góp của 4 nguồn thông tin, "
            "chỉ báo nổi bật, vị trí trong ngành và tin liên quan.",
            TickerDateArgs,
            get_stock_explanation,
            "Đang tra cứu giải thích khuyến nghị",
        ),
        Tool(
            "get_stock_prices",
            "Lấy giá đóng cửa, khối lượng, lợi suất ngày và RSI của N phiên gần nhất cho một mã.",
            PricesArgs,
            get_stock_prices,
            "Đang tra cứu dữ liệu giá",
        ),
        Tool(
            "get_stock_news",
            "Lấy các tin tức (tiêu đề, ngày, link CafeF) của một mã trong N ngày gần nhất.",
            NewsArgs,
            get_stock_news,
            "Đang tìm tin tức",
        ),
        Tool(
            "compare_stocks",
            "So sánh 2–5 mã trong cùng một ngày: hạng, điểm, độ đồng thuận, nguồn đóng góp lớn nhất, chỉ báo.",
            CompareArgs,
            compare_stocks,
            "Đang so sánh các mã",
        ),
        Tool(
            "get_model_performance",
            "Hiệu quả của chiến lược top-k so với chọn ngẫu nhiên và VN-Index: Sharpe, tổng lợi nhuận, "
            "sụt giảm tối đa, nDCG@5, cho năm kiểm thử 2025 hoặc giai đoạn live.",
            PerformanceArgs,
            get_model_performance,
            "Đang tính hiệu quả mô hình",
        ),
        Tool(
            "get_model_info",
            "Mô tả mô hình MIDMoE, dữ liệu huấn luyện, kết quả trong paper và các hạn chế. Dùng khi được hỏi về "
            "mô hình, độ tin cậy hoặc hạn chế.",
            NoArgs,
            get_model_info,
            "Đang đọc thông tin mô hình",
        ),
    ]
}


def tool_specs() -> list[ToolSpec]:
    return [t.spec() for t in TOOLS.values()]


def execute_tool(db: Session, name: str, arguments: dict, ctx: PageContext) -> tuple[dict, bool]:
    """Run a tool. Returns (result, ok). Errors are returned to the model as {"error": message}, never raised."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Không có công cụ tên '{name}'."}, False
    try:
        args = tool.args_model.model_validate(arguments or {})
    except ValidationError as exc:
        return {"error": "Tham số không hợp lệ", "details": json.loads(exc.json(include_url=False))}, False
    try:
        return tool.handler(db, args, ctx), True
    except ServiceError as exc:
        return {"error": exc.message}, False


def dumps(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
