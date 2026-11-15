"""Scheduler container entrypoint (M2.8 / M7.1).

    python -m pipeline.scheduler                 # blocking: 16:30 Asia/Ho_Chi_Minh, Mon-Fri
    python -m pipeline.scheduler --once          # run today's job now and exit
    python -m pipeline.scheduler --once --date 2026-11-13

Holidays are not listed here: on a day without a trading session the job finds no new bar and records
`skipped` itself (PLAN M2.8). If the job raises, a `failed` pipeline_runs row is written so /api/health and the
dashboard show it. A heartbeat file is touched every minute for the container healthcheck.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

from backend.config import get_settings
from backend.db.models import PipelineRun, RunStatus, utcnow
from backend.db.session import get_sessionmaker

log = logging.getLogger("midmoe.scheduler")

RUN_HOUR, RUN_MINUTE = 16, 30
HEARTBEAT = Path(os.environ.get("SCHEDULER_HEARTBEAT", "/tmp/midmoe-scheduler-heartbeat"))
MAX_ERROR_CHARS = 4000


def _tz() -> ZoneInfo:
    return ZoneInfo(get_settings().tz)


def record_failure(run_date: dt.date, started: dt.datetime, error: str) -> None:
    with get_sessionmaker()() as db:
        db.add(
            PipelineRun(
                run_date=run_date,
                status=RunStatus.failed,
                started_at=started,
                finished_at=utcnow(),
                error=error[-MAX_ERROR_CHARS:],
            )
        )
        db.commit()


def run_job(run_date: dt.date | None = None) -> bool:
    """Run the daily job for `run_date` (default: today in Vietnam). Returns True on success."""
    from pipeline import daily_job

    run_date = run_date or dt.datetime.now(_tz()).date()
    started = utcnow()
    log.info("daily job start run_date=%s", run_date)
    try:
        daily_job.run(run_date)
    except Exception as exc:  # the job owns success/degraded/skipped rows; failures are recorded here
        log.exception("daily job failed run_date=%s", run_date)
        record_failure(run_date, started, f"{exc}\n\n{traceback.format_exc()}")
        return False
    log.info("daily job done run_date=%s", run_date)
    return True


def heartbeat() -> None:
    HEARTBEAT.write_text(dt.datetime.now(dt.timezone.utc).isoformat())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="run once now and exit")
    ap.add_argument("--date", type=dt.date.fromisoformat, help="run date for --once (default: today, VN time)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.once:
        return 0 if run_job(args.date) else 1

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched = BlockingScheduler(timezone=_tz())
    sched.add_job(
        run_job,
        CronTrigger(day_of_week="mon-fri", hour=RUN_HOUR, minute=RUN_MINUTE, timezone=_tz()),
        id="daily_job",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3 * 3600,  # container restarted shortly after 16:30 -> still run that day
    )
    sched.add_job(heartbeat, "interval", minutes=1, id="heartbeat", next_run_time=dt.datetime.now(_tz()))
    log.info("scheduler started: daily job at %02d:%02d %s, Mon-Fri", RUN_HOUR, RUN_MINUTE, get_settings().tz)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
