"""Daily pipeline (M2.6) - NOT IMPLEMENTED YET.

Contract used by pipeline/scheduler.py:

    run(run_date: date) -> None
        fetch -> features -> news -> cube -> predict -> explain -> upsert DB, idempotent for `run_date`.
        Writes its own `pipeline_runs` row (success / degraded / skipped when there is no session that day).
        Raise on failure: the scheduler then records a `failed` run with the error message.

Blocked on: notebook `midmoe-vn-cube-build` (CafeF crawl + date rule, PhoBERT pooling, vnstock source/version,
VN normalisation params) and the exported artifacts (notebooks/export_vn_weights.py).
"""
from __future__ import annotations

import datetime as dt


def run(run_date: dt.date) -> None:
    raise NotImplementedError("Pipeline hằng ngày (M2.6) chưa được triển khai.")
