from datetime import date as Date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.schemas import BacktestOut
from backend.services import backtest as svc

router = APIRouter(tags=["backtest"])


@router.get("/backtest", response_model=BacktestOut, summary="Top-k vs random (vs VN-Index)")
def backtest(
    from_: Date | None = Query(None, alias="from"),
    to: Date | None = Query(None),
    k: int = Query(5, ge=1, le=78),
    db: Session = Depends(get_db),
) -> BacktestOut:
    return svc.run_backtest(db, from_, to, k)
