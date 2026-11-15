"""Backtest of stored predictions: top-k vs random (vs VN-Index when available).

Label of a prediction made at the close of day d = ret_1d of every ticker on the next trading day in
`prices` (same as `build_all_windows` in the cube notebooks: label_date = window_end + 1 session).
Days where any ticker has no label are skipped, like the notebook. Metrics use midmoe.metrics, i.e. the
exact paper functions (daily return = SUM of top-k next-day returns; SR is daily, not annualised).

VN-Index (table market_index, code VNINDEX): return on the same label session, MULTIPLIED BY k. The paper's
top-k daily return is a sum over k tickers (k equal positions), so k x index return is the like-for-like
benchmark (k positions all in the index). SR is unaffected by the scaling; IRR and MaxDD are. VN-Index series
and metrics are only returned when every label session has index data (`vnindex_coverage` == 1); a partial
benchmark curve would be misleading.

Results are cached in-process (per worker), keyed on the request + a cheap "data version" (latest pipeline
run id, latest prediction date, prediction count, latest price date): data changes once a day.
"""
from __future__ import annotations

import datetime as dt
from collections import OrderedDict

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.db.models import MarketIndex, PipelineRun, Prediction, Price, Ticker
from backend.schemas import BacktestMetrics, BacktestOut, BacktestPoint
from backend.services.common import BadRequestError, NotFoundError
from backend.services.market_index import VNINDEX, index_returns
from midmoe.metrics import RANDOM_SEEDS, compute_metrics, daily_ret, max_drawdown, random_preds, sharpe

LABEL_SEARCH_DAYS = 20  # calendar days to look ahead for the next session's prices


def _load(db: Session, date_from: dt.date | None, date_to: dt.date | None):
    tickers = list(db.scalars(select(Ticker.ticker).order_by(Ticker.sort_order)))
    col = {t: i for i, t in enumerate(tickers)}

    q = select(Prediction.date, Prediction.ticker, Prediction.score)
    if date_from:
        q = q.where(Prediction.date >= date_from)
    if date_to:
        q = q.where(Prediction.date <= date_to)
    pred_rows = db.execute(q).all()
    if not pred_rows:
        return tickers, [], None, None

    pred_dates = sorted({d for d, _, _ in pred_rows})
    d_idx = {d: i for i, d in enumerate(pred_dates)}
    scores = np.full((len(pred_dates), len(tickers)), np.nan)
    for d, t, s in pred_rows:
        if t in col:
            scores[d_idx[d], col[t]] = s

    price_rows = db.execute(
        select(Price.date, Price.ticker, Price.ret_1d).where(
            Price.date > pred_dates[0], Price.date <= pred_dates[-1] + dt.timedelta(days=LABEL_SEARCH_DAYS)
        )
    ).all()
    price_dates = sorted({d for d, _, _ in price_rows})
    p_idx = {d: i for i, d in enumerate(price_dates)}
    rets = np.full((len(price_dates), len(tickers)), np.nan)
    for d, t, r in price_rows:
        if t in col and r is not None:
            rets[p_idx[d], col[t]] = r
    return tickers, pred_dates, scores, (price_dates, rets)


_CACHE: OrderedDict[tuple, BacktestOut] = OrderedDict()
_CACHE_SIZE = 32


def _data_version(db: Session) -> tuple:
    """Changes whenever the pipeline writes (each run inserts a pipeline_runs row) or predictions change."""
    last_run = db.scalar(select(func.max(PipelineRun.id)))
    max_date, n_pred = db.execute(select(func.max(Prediction.date), func.count()).select_from(Prediction)).one()
    max_price = db.scalar(select(func.max(Price.date)))
    max_idx, n_idx = db.execute(select(func.max(MarketIndex.date), func.count()).select_from(MarketIndex)).one()
    return last_run, max_date, n_pred, max_price, max_idx, n_idx


def clear_cache() -> None:
    _CACHE.clear()


def run_backtest(db: Session, date_from: dt.date | None = None, date_to: dt.date | None = None, k: int = 5) -> BacktestOut:
    if date_from and date_to and date_from > date_to:
        raise BadRequestError("Ngày bắt đầu phải trước ngày kết thúc.")
    key = (date_from, date_to, k, str(db.get_bind().url), _data_version(db))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    result = _compute_backtest(db, date_from, date_to, k)
    _CACHE[key] = result
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return result


def _compute_backtest(db: Session, date_from: dt.date | None, date_to: dt.date | None, k: int) -> BacktestOut:
    tickers, pred_dates, scores, prices = _load(db, date_from, date_to)
    if not pred_dates:
        raise NotFoundError("Không có dự đoán nào trong khoảng thời gian này.")
    if k > len(tickers):
        raise BadRequestError(f"k phải ≤ {len(tickers)}.")

    price_dates, rets = prices
    price_dates_arr = np.array(price_dates, dtype="datetime64[D]")
    keep_dates, label_dates, preds, labels = [], [], [], []
    for i, d in enumerate(pred_dates):
        j = int(np.searchsorted(price_dates_arr, np.datetime64(d), side="right"))  # first session after d
        if j >= len(price_dates):
            continue  # label not known yet (latest day)
        y, s = rets[j], scores[i]
        if np.isnan(y).any() or np.isnan(s).any():
            continue
        keep_dates.append(d)
        label_dates.append(price_dates[j])
        preds.append(s)
        labels.append(y)
    if not keep_dates:
        raise NotFoundError("Chưa đủ dữ liệu lợi suất phiên kế tiếp để backtest khoảng thời gian này.")

    P, Y = np.stack(preds), np.stack(labels)
    topk = daily_ret(P, Y, k)
    m = compute_metrics(list(P), list(Y), k=k)

    rand_daily, rand_metrics = [], []
    for seed in range(RANDOM_SEEDS):
        R = random_preds(len(Y), len(tickers), seed)
        rand_daily.append(daily_ret(R, Y, k))
        rm = compute_metrics(list(R), list(Y), k=k)
        rm["MaxDD"] = max_drawdown(rand_daily[-1])
        rand_metrics.append(rm)
    rand = np.mean(rand_daily, axis=0)

    # VN-Index on the same label sessions, scaled by k (see module docstring). Only used with full coverage.
    idx_ret = index_returns(db, label_dates, VNINDEX)
    coverage = sum(d in idx_ret for d in label_dates) / len(label_dates)
    vni = np.array([k * idx_ret[d] for d in label_dates]) if coverage == 1.0 else None
    vni_cum = np.cumsum(vni) if vni is not None else None

    topk_cum, rand_cum = np.cumsum(topk), np.cumsum(rand)
    series = [
        BacktestPoint(
            date=d,
            topk=float(topk[i]),
            random=float(rand[i]),
            vnindex=None if vni is None else float(vni[i]),
            topk_cum=float(topk_cum[i]),
            random_cum=float(rand_cum[i]),
            vnindex_cum=None if vni_cum is None else float(vni_cum[i]),
        )
        for i, d in enumerate(keep_dates)
    ]
    metrics = {
        "topk": BacktestMetrics(SR=float(m["SR"]), IRR=float(m["IRR"]), MaxDD=max_drawdown(topk), nDCG5=float(m["nDCG@5"])),
        "random": BacktestMetrics(
            SR=float(np.mean([r["SR"] for r in rand_metrics])),
            IRR=float(np.mean([r["IRR"] for r in rand_metrics])),
            MaxDD=float(np.mean([r["MaxDD"] for r in rand_metrics])),
            nDCG5=float(np.mean([r["nDCG@5"] for r in rand_metrics])),
        ),
    }
    if vni is not None:
        metrics["vnindex"] = BacktestMetrics(SR=sharpe(vni), IRR=float(vni.sum()), MaxDD=max_drawdown(vni), nDCG5=None)
    return BacktestOut(
        from_date=keep_dates[0],
        to_date=keep_dates[-1],
        k=k,
        n_days=len(keep_dates),
        vnindex_coverage=coverage,
        series=series,
        metrics=metrics,
    )
