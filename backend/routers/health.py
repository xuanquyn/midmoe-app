from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.schemas import DatesOut, HealthOut
from backend.services import system

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthOut, summary="Trạng thái DB + lần chạy pipeline gần nhất")
def health(db: Session = Depends(get_db)) -> HealthOut:
    return system.get_health(db)


@router.get("/dates", response_model=DatesOut, summary="Danh sách ngày có dự đoán")
def dates(db: Session = Depends(get_db)) -> DatesOut:
    return system.get_dates(db)
