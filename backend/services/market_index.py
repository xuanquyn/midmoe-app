"""Market index bars (VN-Index): idempotent upsert for the pipeline + return lookup for the backtest."""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import MarketIndex

VNINDEX = "VNINDEX"
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def upsert_index_bars(db: Session, bars: pd.DataFrame, index_code: str = VNINDEX, commit: bool = True) -> int:
    """Insert/overwrite daily bars (columns: date, open, high, low, close[, volume]); re-running a day overwrites it.

    ret_1d = close.pct_change(1) over the stored series. It is recomputed from the session before the earliest
    written date onwards, so a backfill that inserts older data also fixes the returns that follow it.
    """
    if bars.empty:
        return 0
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    if df["close"].isna().any():
        raise ValueError(f"{index_code}: close must not be null")

    for rec in df.to_dict("records"):
        values = {c: (None if pd.isna(rec.get(c)) else rec.get(c)) for c in BAR_COLUMNS}
        if values["volume"] is not None:
            values["volume"] = int(values["volume"])
        db.merge(MarketIndex(index_code=index_code, date=rec["date"], **values))
    db.flush()

    first = df["date"].iloc[0]
    prev_date = db.scalar(
        select(MarketIndex.date)
        .where(MarketIndex.index_code == index_code, MarketIndex.date < first)
        .order_by(MarketIndex.date.desc())
        .limit(1)
    )
    start = prev_date or first
    rows = db.scalars(
        select(MarketIndex)
        .where(MarketIndex.index_code == index_code, MarketIndex.date >= start)
        .order_by(MarketIndex.date)
    ).all()
    prev_close = None
    for row in rows:
        if row.date == start and prev_date is not None:
            prev_close = row.close  # anchor only; its own ret_1d is unchanged
            continue
        row.ret_1d = None if prev_close is None or prev_close == 0 else row.close / prev_close - 1
        prev_close = row.close
    if commit:
        db.commit()
    return len(df)


def index_returns(db: Session, dates: Iterable[dt.date], index_code: str = VNINDEX) -> dict[dt.date, float]:
    """{date: ret_1d} for the requested dates (missing dates / null returns are omitted)."""
    dates = list(dates)
    if not dates:
        return {}
    rows = db.execute(
        select(MarketIndex.date, MarketIndex.ret_1d).where(
            MarketIndex.index_code == index_code, MarketIndex.date.between(min(dates), max(dates))
        )
    ).all()
    wanted = set(dates)
    return {d: r for d, r in rows if d in wanted and r is not None}
