"""Cross-validation splitters that match how the hidden test set is drawn.

A CV scheme is only useful if its ranking of experiments agrees with the leaderboard.
Two recurring leaks:

* **Spatial / group leakage** — random hold-out lets neighbouring wells, patients or
  recordings leak into validation. On ROGII, random CV looked far better than the LB;
  K-means-on-coordinates folds were an honest, slightly pessimistic proxy (CV/LB≈1.02).
* **Empty strata for rare classes** — a class with 1–4 examples cannot be split across
  5 folds. Pin such rows to "always train" (fold ``-1``) instead of letting them land
  alone in a validation fold and make the metric explode.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

TRAIN_ONLY = -1


def spatial_group_folds(
    df: pd.DataFrame,
    group_col: str,
    coord_cols: Sequence[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Assign each row a fold by K-means-clustering its group's mean coordinates.

    Every row of a group lands in the same fold, and nearby groups tend to share a
    fold, so validation measures extrapolation to *new regions* rather than
    interpolation between neighbours. Clusters (and so folds) can be very uneven in
    size; check ``np.bincount(folds)`` and adjust ``n_splits`` if one fold dominates.
    """
    if df[group_col].isna().any():
        raise ValueError(f"{group_col!r} contains missing values; every row needs a group")
    centroids = df.groupby(group_col)[list(coord_cols)].mean()
    if len(centroids) < n_splits:
        raise ValueError(f"{len(centroids)} groups is fewer than n_splits={n_splits}")
    labels = KMeans(n_clusters=n_splits, n_init=10, random_state=random_state).fit_predict(
        centroids.to_numpy()
    )
    fold_of_group = pd.Series(labels, index=centroids.index)
    return df[group_col].map(fold_of_group).to_numpy()


def rare_safe_folds(
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    min_count: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Stratified (group) folds with rare-class rows pinned to ``TRAIN_ONLY``.

    ``labels`` may be 1-D class labels or a 2-D multi-hot matrix. For multi-label
    data, stratification uses each row's rarest positive class, which keeps the tail
    spread across folds (a cheap alternative to iterative stratification).

    With ``groups``, a group containing any rare row is pinned to ``TRAIN_ONLY`` as a
    whole, so no group is ever split between train and validation.
    """
    labels = np.asarray(labels)
    min_count = n_splits if min_count is None else min_count
    if labels.ndim == 2:
        counts = labels.sum(axis=0)
        rare_cols = counts < min_count
        rare_rows = labels[:, rare_cols].any(axis=1)
        # Rarest positive class per row; rows with no positive get their own stratum.
        masked = np.where(labels > 0, counts[None, :], np.inf)
        strata = np.where(labels.any(axis=1), masked.argmin(axis=1), -1)
    else:
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        rare_rows = counts[inverse] < min_count
        strata = inverse

    folds = np.full(len(labels), TRAIN_ONLY, dtype=int)
    if groups is None:
        idx = np.flatnonzero(~rare_rows)
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=random_state).split(
            idx, strata[idx]
        )
    else:
        grp = np.asarray(groups)
        if len(grp) != len(labels):
            raise ValueError("groups must have one entry per row")
        rare_rows = np.isin(grp, grp[rare_rows])  # pin whole groups, not single rows
        idx = np.flatnonzero(~rare_rows)
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state).split(
            idx, strata[idx], grp[idx]
        )
    for fold, (_, val) in enumerate(splits):
        folds[idx[val]] = fold
    return folds


def iter_folds(folds: np.ndarray):
    """Yield ``(fold, train_idx, valid_idx)``; ``TRAIN_ONLY`` rows are always in train."""
    for fold in sorted(set(folds.tolist()) - {TRAIN_ONLY}):
        yield fold, np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
