"""Health + available dates. Service functions take a Session and return plain schemas, so the assistant
tools (assistant/tools.py) can call them directly without HTTP."""
from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.db.models import PipelineRun, Prediction
from backend.schemas import DatesOut, HealthOut, PipelineRunOut


def last_pipeline_run(db: Session) -> PipelineRunOut | None:
    run = db.scalars(select(PipelineRun).order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc()).limit(1)).first()
    return PipelineRunOut.model_validate(run) if run else None


def get_health(db: Session) -> HealthOut:
    try:
        db.execute(text("SELECT 1"))
        last_run = last_pipeline_run(db)
    except SQLAlchemyError as exc:
        return HealthOut(status="degraded", db="error", db_error=str(exc.__class__.__name__))
    return HealthOut(status="ok", db="ok", last_run=last_run)


def get_dates(db: Session) -> DatesOut:
    dates = list(db.scalars(select(Prediction.date).distinct().order_by(Prediction.date)))
    return DatesOut(dates=dates, latest=dates[-1] if dates else None)
