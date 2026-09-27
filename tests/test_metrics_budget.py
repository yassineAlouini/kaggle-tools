import time

import numpy as np
import pytest

from kaggle_tools.budget import TimeBudget, timed
from kaggle_tools.metrics import best_threshold, macro_auc, rmse


def test_macro_auc_skips_columns_without_positives():
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    p = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.5], [0.2, 0.3]])
    assert macro_auc(y, p) == 1.0
    with pytest.raises(ValueError, match="both positive"):
        macro_auc(np.zeros((3, 2)), np.zeros((3, 2)))


def test_rmse_and_threshold():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    y = np.array([0, 0, 1, 1])
    thr, score = best_threshold(y, np.array([0.1, 0.35, 0.4, 0.9]))
    assert score == 1.0 and 0.35 < thr <= 0.4


def test_time_budget(capsys):
    b = TimeBudget(total_seconds=100, margin_seconds=10)
    assert b.has(80) and not b.has(95)
    assert b.per_item(3) == pytest.approx(30, abs=0.1)

    @timed
    def f():
        time.sleep(0.01)
        return 1

    assert f() == 1
    assert "[timed]" in capsys.readouterr().out
