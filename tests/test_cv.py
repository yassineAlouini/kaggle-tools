import numpy as np
import pandas as pd

from kaggle_tools.cv import TRAIN_ONLY, iter_folds, rare_safe_folds, spatial_group_folds


def test_spatial_folds_keep_groups_together():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 10)
    centers = rng.uniform(0, 100, (40, 2))
    df = pd.DataFrame({"g": groups, "x": centers[groups, 0], "y": centers[groups, 1]})
    folds = spatial_group_folds(df, "g", ["x", "y"], n_splits=4)
    assert set(folds) == {0, 1, 2, 3}
    assert (df.assign(f=folds).groupby("g")["f"].nunique() == 1).all()


def test_rare_classes_pinned_to_train():
    labels = np.array(["a"] * 50 + ["b"] * 50 + ["rare"] * 2)
    folds = rare_safe_folds(labels, n_splits=5)
    assert (folds[-2:] == TRAIN_ONLY).all()
    assert set(folds[:-2]) == set(range(5))
    for _, tr, va in iter_folds(folds):
        assert set(np.flatnonzero(folds == TRAIN_ONLY)) <= set(tr)
        assert not set(tr) & set(va)


def test_rare_safe_folds_multilabel_with_groups():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 4)) < 0.3).astype(int)
    y[:, 3] = 0
    y[0, 3] = 1  # a class with a single positive
    groups = np.arange(200) // 4
    folds = rare_safe_folds(y, groups=groups, n_splits=5)
    assert folds[0] == TRAIN_ONLY
    df = pd.DataFrame({"g": groups[1:], "f": folds[1:]})
    assert (df.groupby("g")["f"].nunique() == 1).all()
