"""Evaluation metrics, copied from notebook `midmoe-vn-transfer` (Phase 7 / Phase 8) so the app reports
numbers computed exactly like the paper.

Conventions (from the paper code):
- daily return of a strategy = SUM of the next-day returns of its top-k tickers (`daily_ret`)
- IRR = sum of daily returns; SR = mean / std of daily returns (daily, not annualised)
- nDCG@5 with 5 quantile grades of the label per day
`max_drawdown` is added for the app: largest peak-to-trough fall of the cumulative (summed) return curve,
consistent with IRR being a sum.
"""
from __future__ import annotations

import numpy as np


# ---- verbatim from the notebook -------------------------------------------------------------
def daily_quantile_grades(label, n_grades=5):
    ranks = np.argsort(np.argsort(label))
    return np.clip((ranks / len(label) * n_grades).astype(np.int32), 0, n_grades - 1)


def compute_metrics(preds_list, labels_list, k=5):
    daily_irr, ndcgs = [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for pred, label in zip(preds_list, labels_list):
        topk_pred = np.argsort(-pred)[:k]
        daily_irr.append(label[topk_pred].sum())
        grade = daily_quantile_grades(label)
        dcg = (grade[topk_pred] * disc).sum()
        idcg = (np.sort(grade)[::-1][:k] * disc).sum()
        ndcgs.append(dcg / idcg if idcg > 1e-8 else 0.0)
    daily_irr = np.array(daily_irr)
    return {'SR': daily_irr.mean() / (daily_irr.std() + 1e-8), 'IRR': daily_irr.sum(), 'nDCG@5': np.mean(ndcgs)}


def daily_ret(preds, labels, k=5):
    top = np.argsort(-preds, axis=1)[:, :k]
    return np.take_along_axis(labels, top, axis=1).sum(axis=1)


def random_preds(n_days: int, n_tickers: int, seed: int) -> np.ndarray:
    """Random baseline exactly as `evaluate_random_vn`: one RandomState(seed) permutation per day."""
    rng = np.random.RandomState(seed)
    return np.stack([rng.permutation(n_tickers).astype(np.float32) for _ in range(n_days)])


RANDOM_SEEDS = 5
# ---------------------------------------------------------------------------------------------


def sharpe(daily: np.ndarray) -> float:
    return float(daily.mean() / (daily.std() + 1e-8)) if len(daily) else 0.0


def max_drawdown(daily: np.ndarray) -> float:
    """Max peak-to-trough drop of cumsum(daily) (>= 0, same unit as IRR). Starts from 0 before day 1."""
    if len(daily) == 0:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(daily)])
    return float((np.maximum.accumulate(curve) - curve).max())
