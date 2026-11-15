"""Explanations (M3): source contributions, feature snapshot, sector context, news, confidence.

Source ablations are applied around the unmodified MarketMoE (PLAN §9: no change to model logic):
    news       -> news_mask = 0
    market     -> forward hook replaces the market_conv output with zeros
    timeseries -> forward hook replaces the ts_proj output with zeros
    graph      -> A_norm temporarily set to the identity (no propagation along the sector graph)
Every pass uses the same seeded diffusion noise as the full pass (predict.seeded_forward), so
`delta = score_full - score_ablated` reflects only the removed source. Deltas and ablated ranks are averaged
over the 10 seeds.

The output payload follows PLAN §5 / backend.schemas.ExplanationPayload.
"""
from __future__ import annotations

import contextlib
import datetime as dt
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from midmoe.features import percentile_rank
from midmoe.model import MarketMoE
from midmoe.predict import PredictionResult, Predictor, ranks_from_scores, seeded_forward

SOURCES = ("market", "news", "graph", "timeseries")
SOURCE_LABELS_VI = {"market": "thị trường", "news": "tin tức", "graph": "quan hệ ngành", "timeseries": "chuỗi giá"}
NEWS_LOOKBACK_SESSIONS = 5  # news of the day + 5 previous sessions (M3.4)
MAX_NEWS_PER_TICKER = 10


# ------------------------------------------------------------------------------------------- ablation
def _zero_output_hook(_module, _inputs, output):
    return torch.zeros_like(output)


@contextlib.contextmanager
def ablate(model: MarketMoE, source: str | None) -> Iterator[None]:
    """Temporarily remove one information source from `model` (news is handled on the inputs)."""
    if source in (None, "news"):
        yield
        return
    if source == "market":
        handle = model.market_conv.register_forward_hook(_zero_output_hook)
        try:
            yield
        finally:
            handle.remove()
    elif source == "timeseries":
        handle = model.ts_proj.register_forward_hook(_zero_output_hook)
        try:
            yield
        finally:
            handle.remove()
    elif source == "graph":
        original = model.A_norm.detach().clone()
        with torch.no_grad():
            model.A_norm.copy_(torch.eye(original.shape[0], dtype=original.dtype, device=original.device))
        try:
            yield
        finally:
            with torch.no_grad():
                model.A_norm.copy_(original)
    else:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


def _per_seed(predictor: Predictor, x, news_emb, news_mask, source: str | None) -> np.ndarray:
    x, news_emb, news_mask = predictor._to_tensors(x, news_emb, news_mask)
    if source == "news":
        news_mask = torch.zeros_like(news_mask)
    outs = []
    for m in predictor.models:
        with ablate(m, source):
            outs.append(seeded_forward(m, x, news_emb, news_mask, predictor.noise_seed)[0])
    return torch.stack(outs).cpu().numpy().astype(np.float64)  # (n_seeds, N)


def source_contributions(
    predictor: Predictor, x, news_emb, news_mask, full: PredictionResult | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """{source: {"delta": (N,), "rank_without": (N,), "score_without": (N,)}} averaged over seeds (M3.1)."""
    per_seed_full = full.per_seed if full is not None else _per_seed(predictor, x, news_emb, news_mask, None)
    out = {}
    for src in SOURCES:
        abl = _per_seed(predictor, x, news_emb, news_mask, src)
        score_without = abl.mean(axis=0)
        out[src] = {
            "delta": (per_seed_full - abl).mean(axis=0),
            "score_without": score_without,
            "rank_without": ranks_from_scores(score_without),
        }
    return out


# ------------------------------------------------------------------------------------------- features
def _feature_label(name: str, value: float | None, pct: float | None) -> str | None:
    if value is None or pct is None or np.isnan(value):
        return None
    if name == "rsi":
        if value > 70:
            return "quá mua"
        if value < 30:
            return "quá bán"
        return None
    horizon = {"ret_1d": "1 ngày", "ret_5d": "5 ngày", "ret_10d": "10 ngày"}
    if name in horizon:
        if pct >= 0.9:
            return f"top 10% tăng mạnh nhất ({horizon[name]})"
        if pct <= 0.1:
            return f"nhóm 10% giảm mạnh nhất ({horizon[name]})"
        return None
    if name == "macd":
        if pct >= 0.9:
            return "MACD thuộc nhóm cao nhất"
        if pct <= 0.1:
            return "MACD thuộc nhóm thấp nhất"
        return None
    if name == "close_open_ratio":
        if pct >= 0.9:
            return "tăng mạnh trong phiên"
        if pct <= 0.1:
            return "giảm mạnh trong phiên"
        return None
    if name == "high_low_ratio" and pct >= 0.9:
        return "biên độ trong phiên lớn"
    return None


def feature_snapshot(raw_today: pd.DataFrame, feature_order: Sequence[str]) -> dict[str, dict]:
    """raw_today: index = ticker, columns = raw (un-normalised) indicators of the day (M3.2).

    Returns {ticker: {feature: {"value", "pct", "label"}}}; pct = cross-sectional percentile in [0, 1].
    """
    pcts = {f: percentile_rank(raw_today[f]) for f in feature_order}
    out: dict[str, dict] = {}
    for t in raw_today.index:
        feats = {}
        for f in feature_order:
            v = raw_today.at[t, f]
            value = None if pd.isna(v) else float(v)
            p = pcts[f].get(t)
            pct = None if p is None or pd.isna(p) else float(p)
            feats[f] = {"value": value, "pct": pct, "label": _feature_label(f, value, pct)}
        out[t] = feats
    return out


# ------------------------------------------------------------------------------------------- sector / news
def sector_context(tickers: Sequence[str], sector_of: Mapping[str, str | None], rank: np.ndarray, top_n: int = 3) -> dict[str, dict]:
    """{ticker: {"name", "rank_in_sector", "n_in_sector", "top"}} (M3.3)."""
    by_sector: dict[str, list[tuple[int, str]]] = {}
    for t, r in zip(tickers, rank):
        by_sector.setdefault(sector_of.get(t) or "Khác", []).append((int(r), t))
    out = {}
    for name, members in by_sector.items():
        members.sort()
        order = [t for _, t in members]
        for pos, t in enumerate(order, start=1):
            out[t] = {"name": name, "rank_in_sector": pos, "n_in_sector": len(order), "top": order[:top_n]}
    return out


def attach_news(
    news: Sequence[Mapping], session_dates: Sequence[dt.date], as_of: dt.date, lookback: int = NEWS_LOOKBACK_SESSIONS
) -> dict[str, list[dict]]:
    """News of `as_of` and the `lookback` previous sessions, newest first (M3.4).

    news: items with ticker, date, title, url. session_dates: trading days (any order)."""
    past = sorted(d for d in set(session_dates) if d <= as_of)
    start = past[-(lookback + 1)] if len(past) > lookback else (past[0] if past else as_of)
    out: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for n in sorted(news, key=lambda n: n["date"], reverse=True):
        d = n["date"]
        if not (start <= d <= as_of) or n["url"] in seen:
            continue
        seen.add(n["url"])
        items = out.setdefault(n["ticker"], [])
        if len(items) < MAX_NEWS_PER_TICKER:
            items.append({"date": d.isoformat(), "title": n["title"], "url": n["url"]})
    return out


# ------------------------------------------------------------------------------------------- payload
def summary_text(rank: int, n: int, votes: int, n_models: int, top_source: str) -> str:
    return (
        f"Xếp hạng {rank}/{n}, {votes}/{n_models} mô hình đồng thuận; "
        f"yếu tố {SOURCE_LABELS_VI[top_source]} đóng góp nhiều nhất."
    )


def build_explanations(
    pred: PredictionResult,
    contributions: Mapping[str, Mapping[str, np.ndarray]],
    features: Mapping[str, dict],
    sectors: Mapping[str, dict],
    news: Mapping[str, list[dict]] | None = None,
) -> dict[str, dict]:
    """Assemble the explanations.payload JSON for every ticker (PLAN §5, M3.5 confidence included)."""
    n = len(pred.tickers)
    n_models = len(pred.seeds)
    news = news or {}
    out = {}
    for i, t in enumerate(pred.tickers):
        sources = {
            s: {"delta": float(contributions[s]["delta"][i]), "rank_without": int(contributions[s]["rank_without"][i])}
            for s in SOURCES
        }
        top_source = max(sources, key=lambda s: sources[s]["delta"])
        rank, votes = int(pred.rank[i]), int(pred.top5_votes[i])
        out[t] = {
            "rank": rank,
            "score": float(pred.score_mean[i]),
            "top5_votes": votes,
            "score_std": float(pred.score_std[i]),
            "n_models": n_models,
            "sources": sources,
            "features": features.get(t, {}),
            "sector": sectors[t],
            "news": news.get(t, []),
            "summary": summary_text(rank, n, votes, n_models, top_source),
        }
    return out


def explain_day(
    predictor: Predictor,
    x,
    news_emb,
    news_mask,
    raw_today: pd.DataFrame,
    sector_of: Mapping[str, str | None],
    news_items: Sequence[Mapping] = (),
    session_dates: Sequence[dt.date] = (),
    as_of: dt.date | None = None,
) -> tuple[PredictionResult, dict[str, dict]]:
    """Predict one day and build every ticker's explanation (5 forward passes per seed)."""
    pred = predictor.predict(x, news_emb, news_mask)
    contrib = source_contributions(predictor, x, news_emb, news_mask, full=pred)
    feats = feature_snapshot(raw_today.reindex(pred.tickers), predictor.art.feature_order)
    sectors = sector_context(pred.tickers, sector_of, pred.rank)
    news = attach_news(news_items, session_dates, as_of) if as_of else {}
    return pred, build_explanations(pred, contrib, feats, sectors, news)
