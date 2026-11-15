from datetime import date as Date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.schemas import RankingsOut, RecommendationsOut
from backend.services import rankings as svc

router = APIRouter(tags=["recommendations"])


@router.get("/recommendations", response_model=RecommendationsOut, summary="Top-k mã của một ngày")
def recommendations(
    date: Date | None = Query(None, description="Mặc định: ngày mới nhất; ngày không có phiên -> phiên gần nhất trước đó"),
    k: int = Query(5, ge=1, le=78),
    db: Session = Depends(get_db),
) -> RecommendationsOut:
    return svc.get_recommendations(db, date, k)


@router.get("/rankings", response_model=RankingsOut, summary="Xếp hạng đủ 78 mã")
def rankings(
    date: Date | None = Query(None),
    sector: str | None = Query(None, description="Lọc theo ngành ICB"),
    db: Session = Depends(get_db),
) -> RankingsOut:
    return svc.get_rankings(db, date, sector)


@router.get("/sectors", response_model=list[str], summary="Danh sách ngành (cho bộ lọc)")
def sectors(db: Session = Depends(get_db)) -> list[str]:
    return svc.list_sectors(db)
