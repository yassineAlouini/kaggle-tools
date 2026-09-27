"""Metrics that mirror common Kaggle scorers.

Before optimising anything, read the organisers' metric code. Details such as skipped
empty columns, unclipped penalty terms or micro- vs macro-averaging decide where the
score leverage is (on Biohub cell tracking the node-count penalty was unclipped and
the ground-truth size was shipped in the data, which made ``T_pred`` a tunable knob).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import fbeta_score, roc_auc_score


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Column-wise ROC-AUC averaged over columns that have both classes.

    This matches BirdCLEF-style scorers, which skip columns without positives
    instead of failing or counting them as 0.5.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    keep = (y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < len(y_true))
    if not keep.any():
        raise ValueError("no column has both positive and negative examples")
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Grid-search the decision threshold; returns ``(threshold, score)``.

    Defaults to F1. Pick the threshold on OOF predictions, never on the LB.
    """
    if metric is None:

        def metric(t: np.ndarray, p: np.ndarray) -> float:
            return float(fbeta_score(t, p, beta=1.0, average="micro", zero_division=0))

    grid = np.linspace(0.05, 0.95, 19) if thresholds is None else np.asarray(thresholds)
    y_score = np.asarray(y_score)
    scores = [metric(y_true, (y_score >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])
