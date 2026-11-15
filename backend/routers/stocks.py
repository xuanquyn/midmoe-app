from datetime import date as Date

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.schemas import ExplanationOut, NewsItem, StockOut
from backend.services import stocks as svc

router = APIRouter(prefix="/stocks", tags=["stocks"])

TickerPath = Path(..., min_length=1, max_length=16, description="Mã cổ phiếu, ví dụ FPT")


@router.get("/{ticker}", response_model=StockOut, summary="OHLCV + 7 chỉ báo + lịch sử hạng")
def stock(
    ticker: str = TickerPath,
    from_: Date | None = Query(None, alias="from", description="Mặc định: 180 ngày trước `to`"),
    to: Date | None = Query(None, description="Mặc định: phiên mới nhất"),
    db: Session = Depends(get_db),
) -> StockOut:
    return svc.get_stock(db, ticker, from_, to)


@router.get("/{ticker}/explanation", response_model=ExplanationOut, summary="Giải thích khuyến nghị")
def explanation(
    ticker: str = TickerPath,
    date: Date | None = Query(None),
    db: Session = Depends(get_db),
) -> ExplanationOut:
    return svc.get_explanation(db, ticker, date)


@router.get("/{ticker}/news", response_model=list[NewsItem], summary="Tin tức gần đây")
def news(
    ticker: str = TickerPath,
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
) -> list[NewsItem]:
    return svc.get_news(db, ticker, days)
