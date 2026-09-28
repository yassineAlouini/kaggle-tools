"""Ensembling: rank-blend and greedy (Caruana) hill-climbing.

For rank-based metrics (AUC, Spearman, macro column-wise AUC) blend *ranks*, not raw
scores. Averaging raw probabilities lets whichever model has the larger score range
dominate regardless of its weight. On BirdCLEF 2026, switching from score-blend to a
per-column rank-blend of two arms gave +0.011 LB after a long plateau.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def rank_normalize(preds: np.ndarray) -> np.ndarray:
    """Per-column percentile rank in (0, 1]. AUC-invariant for a single model."""
    arr = np.asarray(preds, dtype=float)
    if np.isnan(arr).any():
        raise ValueError("predictions contain NaN; fill or drop them before ranking")
    squeeze = arr.ndim == 1
    ranked = pd.DataFrame(arr[:, None] if squeeze else arr).rank(axis=0, pct=True).to_numpy()
    return ranked[:, 0] if squeeze else ranked


def rank_blend(preds: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Weighted average of per-column percentile ranks."""
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(preds):
        raise ValueError("one weight per prediction array is required")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    w = w / w.sum()
    return sum(wi * rank_normalize(p) for wi, p in zip(w, preds, strict=True))


def hill_climb(
    oof_preds: Sequence[np.ndarray],
    y_true: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_iter: int = 50,
    maximize: bool = True,
    use_ranks: bool = False,
) -> np.ndarray:
    """Greedy forward selection with replacement (Caruana et al., 2004).

    Returns normalised weights (one per model). Evaluate the result on held-out
    folds: hill-climbing on the same OOF it is scored on will overfit.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    models = [rank_normalize(p) if use_ranks else np.asarray(p, dtype=float) for p in oof_preds]
    sign = 1.0 if maximize else -1.0
    counts = np.zeros(len(models))
    current = np.zeros_like(models[0])
    best_score = -np.inf
    for _ in range(n_iter):
        n = counts.sum()
        scores = [sign * metric(y_true, (current * n + m) / (n + 1)) for m in models]
        j = int(np.argmax(scores))
        if scores[j] <= best_score and n > 0:
            break
        best_score = scores[j]
        current = (current * n + models[j]) / (n + 1)
        counts[j] += 1
    return counts / counts.sum()
