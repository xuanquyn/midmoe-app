import numpy as np
import torch

from midmoe.predict import Predictor, aggregate, ranks_from_scores


def _inputs(pred: Predictor, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.random((1, 1, pred.T, pred.N, pred.D), dtype=np.float32)
    news_emb = rng.standard_normal((1, pred.N, 768)).astype(np.float32)
    news_mask = (rng.random((1, pred.N)) > 0.5).astype(np.float32)
    return x, news_emb, news_mask


def test_predict_shapes_and_determinism(fake_artifacts):
    pred = Predictor(str(fake_artifacts))
    assert pred.seeds == [0, 1, 2]
    inp = _inputs(pred)
    state_before = torch.get_rng_state()
    r1 = pred.predict(*inp)
    r2 = pred.predict(*inp)
    assert torch.equal(state_before, torch.get_rng_state()), "global RNG must not be consumed"
    assert r1.per_seed.shape == (3, pred.N)
    np.testing.assert_array_equal(r1.per_seed, r2.per_seed)
    assert sorted(r1.rank.tolist()) == list(range(1, pred.N + 1))
    assert r1.top5_votes.sum() == 3 * 5
    assert len(r1.to_records()) == pred.N


def test_aggregate_ranks():
    per_seed = np.array([[3.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    res = aggregate(per_seed, ["A", "B", "C"], [0, 1], top_k=1)
    np.testing.assert_allclose(res.score_mean, [2.5, 1.0, 2.5])
    assert res.rank.tolist() == [1, 3, 2]  # stable tie-break
    assert res.top5_votes.tolist() == [1, 0, 1]
    assert ranks_from_scores(np.array([0.1, 0.9, 0.5])).tolist() == [3, 1, 2]
