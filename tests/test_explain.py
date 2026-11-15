import datetime as dt
import time

import numpy as np
import pandas as pd
import pytest
import torch

from midmoe.explain import (
    SOURCES,
    ablate,
    attach_news,
    explain_day,
    feature_snapshot,
    sector_context,
    source_contributions,
)
from midmoe.predict import Predictor, seeded_forward
from tests.conftest import FEATURES


@pytest.fixture(scope="module")
def setup(fake_artifacts):
    pred = Predictor(str(fake_artifacts))
    rng = np.random.default_rng(1)
    x = rng.random((1, 1, pred.T, pred.N, pred.D), dtype=np.float32)
    ne = rng.standard_normal((1, pred.N, 768)).astype(np.float32)
    nm = np.ones((1, pred.N), dtype=np.float32)
    return pred, x, ne, nm


def test_ablation_restores_model(setup):
    pred, x, ne, nm = setup
    before = pred.predict_per_seed(x, ne, nm)
    a_norm = pred.models[0].A_norm.clone()
    for src in SOURCES:
        source_contributions(pred, x, ne, nm)
    assert torch.equal(a_norm, pred.models[0].A_norm)
    assert all(not m.market_conv._forward_hooks and not m.ts_proj._forward_hooks for m in pred.models)
    np.testing.assert_array_equal(before, pred.predict_per_seed(x, ne, nm))


def test_each_ablation_changes_output(setup):
    pred, x, ne, nm = setup
    contrib = source_contributions(pred, x, ne, nm)
    for src in SOURCES:
        d = contrib[src]["delta"]
        assert d.shape == (pred.N,) and np.abs(d).max() > 0, src
        assert sorted(contrib[src]["rank_without"].tolist()) == list(range(1, pred.N + 1))


def test_news_ablation_equals_zero_mask(setup):
    pred, x, ne, nm = setup
    full = pred.predict_per_seed(x, ne, nm)
    no_news = pred.predict_per_seed(x, ne, np.zeros_like(nm))
    contrib = source_contributions(pred, x, ne, nm)
    np.testing.assert_allclose(contrib["news"]["delta"], (full - no_news).mean(0), atol=1e-12)


def test_market_hook_zeroes_branch(setup):
    pred, x, ne, nm = setup
    m = pred.models[0]
    xt = torch.as_tensor(x)
    with ablate(m, "market"):
        assert torch.count_nonzero(m.market_conv(xt)) == 0
    assert torch.count_nonzero(m.market_conv(xt)) > 0
    with pytest.raises(ValueError):
        with ablate(m, "bogus"):
            pass


def test_graph_identity_inside_context(setup):
    pred, *_ = setup
    m = pred.models[0]
    with ablate(m, "graph"):
        assert torch.equal(m.A_norm, torch.eye(pred.N))
    assert not torch.equal(m.A_norm, torch.eye(pred.N))


def test_feature_snapshot_labels():
    raw = pd.DataFrame(
        {f: np.linspace(0, 1, 10) for f in FEATURES} | {"rsi": [75, 25, 50, 50, 50, 50, 50, 50, 50, 50]},
        index=[f"T{i}" for i in range(10)],
    )
    raw.loc["T5", "ret_5d"] = np.nan
    snap = feature_snapshot(raw, FEATURES)
    assert snap["T0"]["rsi"]["label"] == "quá mua" and snap["T1"]["rsi"]["label"] == "quá bán"
    assert snap["T9"]["ret_5d"]["label"].startswith("top 10%") and snap["T9"]["ret_5d"]["pct"] == 1.0
    assert snap["T0"]["ret_1d"]["label"].startswith("nhóm 10% giảm")
    assert snap["T5"]["ret_5d"] == {"value": None, "pct": None, "label": None}


def test_sector_context():
    ctx = sector_context(["A", "B", "C", "D"], {"A": "NH", "B": "NH", "C": "BĐS", "D": "NH"}, np.array([3, 1, 2, 4]))
    assert ctx["B"] == {"name": "NH", "rank_in_sector": 1, "n_in_sector": 3, "top": ["B", "A", "D"]}
    assert ctx["C"]["rank_in_sector"] == 1 and ctx["C"]["n_in_sector"] == 1


def test_attach_news_window_and_dedupe():
    sessions = [dt.date(2026, 11, d) for d in (3, 4, 5, 6, 7, 10, 11, 12, 13)]
    as_of = dt.date(2026, 11, 13)
    news = [
        {"ticker": "A", "date": dt.date(2026, 11, 13), "title": "new", "url": "u1"},
        {"ticker": "A", "date": dt.date(2026, 11, 6), "title": "5 sessions back", "url": "u2"},
        {"ticker": "A", "date": dt.date(2026, 11, 5), "title": "too old", "url": "u3"},
        {"ticker": "A", "date": dt.date(2026, 11, 14), "title": "future", "url": "u4"},
        {"ticker": "B", "date": dt.date(2026, 11, 12), "title": "dup", "url": "u1"},
    ]
    out = attach_news(news, sessions, as_of)
    assert [n["title"] for n in out["A"]] == ["new", "5 sessions back"]
    assert "B" not in out


def test_explain_day_payload_validates(setup):
    from backend.schemas import ExplanationPayload

    pred, x, ne, nm = setup
    raw = pd.DataFrame(np.random.default_rng(0).random((pred.N, len(FEATURES))), index=pred.tickers, columns=FEATURES)
    sectors = {t: ("Ngân hàng" if i % 2 else "Công nghệ") for i, t in enumerate(pred.tickers)}
    news = [{"ticker": pred.tickers[0], "date": dt.date(2026, 11, 13), "title": "Tin", "url": "https://cafef.vn/x"}]
    start = time.perf_counter()
    result, payloads = explain_day(pred, x, ne, nm, raw, sectors, news, [dt.date(2026, 11, 13)], dt.date(2026, 11, 13))
    elapsed = time.perf_counter() - start
    assert set(payloads) == set(pred.tickers)
    for t, p in payloads.items():
        ExplanationPayload.model_validate(p)
        assert p["rank"] == int(result.rank[pred.tickers.index(t)])
    top = min(payloads.values(), key=lambda p: p["rank"])
    assert top["summary"].startswith(f"Xếp hạng 1/{pred.N}")
    assert payloads[pred.tickers[0]]["news"][0]["url"] == "https://cafef.vn/x"
    assert elapsed < 30


def test_ablation_uses_same_noise(setup):
    """With no source removed, the ablation path must reproduce the full scores exactly."""
    pred, x, ne, nm = setup
    m = pred.models[0]
    xt, net, nmt = (torch.as_tensor(a) for a in (x, ne, nm))
    with ablate(m, None):
        a = seeded_forward(m, xt, net, nmt, pred.noise_seed)
    b = seeded_forward(m, xt, net, nmt, pred.noise_seed)
    assert torch.equal(a, b)
