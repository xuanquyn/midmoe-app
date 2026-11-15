"""Seed / refresh the `tickers` table from artifacts.

    python -m backend.db.seed [--artifact-dir artifacts/vn]

sort_order = position in vn_tickers_sorted.json (the model's N axis). Metadata (name, exchange, icb_sector)
comes from icb_industry.csv when present; column names are matched loosely because the CSV is produced by
the VN cube-build notebook.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.db.models import Ticker
from backend.db.session import get_sessionmaker

log = logging.getLogger(__name__)

COLUMN_ALIASES = {
    "ticker": ["ticker", "symbol", "code", "ma"],
    "name": ["name", "organ_name", "company_name", "organ_short_name", "ten"],
    "exchange": ["exchange", "comgroupcode", "floor", "san"],
    "icb_sector": ["icb_sector", "icb_name2", "icb_name", "sector", "industry", "nganh"],
}


def _pick(df: pd.DataFrame, field: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    return next((lower[a] for a in COLUMN_ALIASES[field] if a in lower), None)


def load_ticker_rows(artifact_dir: Path) -> list[dict]:
    tickers: list[str] = json.loads((artifact_dir / "vn_tickers_sorted.json").read_text())
    meta: dict[str, dict] = {}
    csv = artifact_dir / "icb_industry.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        cols = {f: _pick(df, f) for f in COLUMN_ALIASES}
        if cols["ticker"] is None:
            raise ValueError(f"{csv}: no ticker column (columns={list(df.columns)})")
        for _, r in df.iterrows():
            meta[str(r[cols["ticker"]]).strip().upper()] = {
                f: (None if c is None or pd.isna(r[c]) else str(r[c]).strip())
                for f, c in cols.items()
                if f != "ticker"
            }
    else:
        log.warning("%s not found: seeding tickers without name/sector", csv)
    return [{"ticker": t, "sort_order": i, **meta.get(t.upper(), {})} for i, t in enumerate(tickers)]


def seed_tickers(db: Session, rows: list[dict]) -> int:
    for row in rows:
        db.merge(Ticker(**row))  # upsert by primary key -> idempotent
    db.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=get_settings().artifact_dir)
    parser.add_argument(
        "--skip-missing", action="store_true", help="exit quietly when artifacts are not there (container start)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.skip_missing and not (Path(args.artifact_dir) / "vn_tickers_sorted.json").exists():
        log.warning("No vn_tickers_sorted.json in %s: tickers not seeded", args.artifact_dir)
        return
    rows = load_ticker_rows(Path(args.artifact_dir))
    with get_sessionmaker()() as db:
        n = seed_tickers(db, rows)
    log.info("Seeded %d tickers", n)


if __name__ == "__main__":
    main()
