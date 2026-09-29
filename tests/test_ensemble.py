import numpy as np
from sklearn.metrics import roc_auc_score

from kaggle_tools.ensemble import hill_climb, rank_blend, rank_normalize


def test_rank_normalize_is_per_column():
    r = rank_normalize(np.array([[0.1, 10], [0.2, 30], [0.3, 20]]))
    assert np.allclose(r[:, 0], [1 / 3, 2 / 3, 1])
    assert np.allclose(r[:, 1], [1 / 3, 1, 2 / 3])


def test_rank_blend_ignores_scale():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    good = y + rng.normal(0, 0.5, 500)
    tiny_noise = rng.normal(0, 1, 500) * 1e-3  # small magnitude, zero signal
    big_noise = rng.normal(0, 1, 500) * 1e3  # huge magnitude, zero signal
    # Score-blend lets the big-magnitude model dominate; rank-blend does not.
    assert roc_auc_score(y, 0.5 * good + 0.5 * big_noise) < 0.6
    assert roc_auc_score(y, rank_blend([good, big_noise])) > 0.75
    assert roc_auc_score(y, rank_blend([good, tiny_noise], [0.9, 0.1])) > 0.85


def test_hill_climb_prefers_the_signal_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    good = y + rng.normal(0, 0.7, 1000)
    noise = rng.normal(0, 1, 1000)
    w = hill_climb([noise, good], y, roc_auc_score)
    assert w[1] > w[0]
    assert np.isclose(w.sum(), 1)
