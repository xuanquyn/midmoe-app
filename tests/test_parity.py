"""M1.4 parity: packaged Predictor must reproduce the Kaggle outputs.

1. Exact parity vs `vn_parity_ref.npz` (written by notebooks/export_vn_weights.py with the same
   seeded-noise convention, CPU, batch=1): max abs diff < 1e-4.
2. Sanity vs `vn_preds.npz` (paper predictions: GPU, batch=8, *unseeded* diffusion noise). These can't
   match bit-for-bit because MarketMoE draws random noise in eval mode, so we check ranking agreement.
"""
import time

import numpy as np
import pytest
from scipy.stats import spearmanr

from tests.conftest import real_artifact_dir

ART = real_artifact_dir()
pytestmark = pytest.mark.skipif(
    not (ART / "weights").exists() or not any((ART / "weights").glob("model_seed*.pt")),
    reason=f"no deployed weights in {ART} (run notebooks/export_vn_weights.py on Kaggle)",
)

TOL = 1e-4


@pytest.fixture(scope="module")
def predictor():
    from midmoe.predict import Predictor

    return Predictor(str(ART))


@pytest.fixture(scope="module")
def test_cube():
    path = ART / "vn_test_cube.npz"
    if not path.exists():
        pytest.skip("vn_test_cube.npz missing")
    return np.load(path, allow_pickle=True)


def _all_days_per_seed(predictor, cube) -> np.ndarray:
    out = [
        predictor.predict_per_seed(cube["x"][i : i + 1], cube["news_emb"][i : i + 1], cube["news_mask"][i : i + 1])
        for i in range(len(cube["x"]))
    ]
    return np.stack(out, axis=1)  # (n_seeds, T, N)


@pytest.fixture(scope="module")
def ours(predictor, test_cube):
    return _all_days_per_seed(predictor, test_cube)


def test_exact_parity_with_export_ref(predictor, ours):
    path = ART / "vn_parity_ref.npz"
    if not path.exists():
        pytest.skip("vn_parity_ref.npz missing")
    ref = np.load(path, allow_pickle=True)
    assert int(ref["noise_seed"]) == predictor.noise_seed
    diff = np.abs(ours - ref["per_seed"]).max()
    print(f"max abs diff vs export ref: {diff:.3e}")
    assert diff < TOL


def test_ranking_agreement_with_paper_preds(ours):
    path = ART / "vn_preds.npz"
    if not path.exists():
        pytest.skip("vn_preds.npz missing")
    preds = np.load(path)
    key = next((k for k in preds.files if k.startswith("transfer_minimal__1")), None)
    assert key, f"no transfer_minimal__1.0 key in {preds.files}"
    paper = preds[key]
    assert paper.shape == ours.shape
    rho = np.mean([spearmanr(paper[s, t], ours[s, t]).correlation for s in range(paper.shape[0]) for t in range(paper.shape[1])])
    diff = np.abs(paper - ours).max()
    print(f"vs paper preds: mean Spearman={rho:.4f}, max abs diff={diff:.3e}")
    # Paper models were retrained for export only if the run is not reproducible; record the numbers either way.
    assert rho > 0.9


def test_predict_one_day_under_5s(predictor, test_cube):
    x, ne, nm = test_cube["x"][-1:], test_cube["news_emb"][-1:], test_cube["news_mask"][-1:]
    start = time.perf_counter()
    predictor.predict(x, ne, nm)
    assert time.perf_counter() - start < 5.0
