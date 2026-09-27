import numpy as np
import pandas as pd
import pytest

from kaggle_tools.submission import ProgressiveSubmission, build_submission, validate_submission


@pytest.fixture
def sample():
    return pd.DataFrame({"row_id": ["a", "b", "c"], "x": 0.5, "y": 0.5})


def test_build_submission_shape_check():
    with pytest.raises(ValueError, match="shape"):
        build_submission(["a"], np.zeros((2, 2)), ["x", "y"])


def test_validate_catches_problems(sample):
    bad = sample.copy()
    bad.loc[0, "x"] = np.nan
    assert validate_submission(sample, sample) == []
    assert any("NaN" in p for p in validate_submission(bad, sample))
    assert any("columns" in p for p in validate_submission(sample[["row_id", "y", "x"]], sample))


def test_progressive_keeps_last_good_output(sample, tmp_path):
    path = tmp_path / "submission.csv"
    sub = ProgressiveSubmission(sample, path)
    pd.testing.assert_frame_equal(pd.read_csv(path), sample)

    with sub.stage("stage1"):
        sub.write(np.full((3, 2), 0.9))
    with sub.stage("stage2"):
        sub.write(np.full((3, 2), 0.1))
        raise RuntimeError("boom")  # after write: file already updated
    with sub.stage("stage3"):
        sub.write(np.full((2, 2), 0.3))  # wrong shape -> rejected, stage fails

    assert sub.completed == ["stage1"]
    assert sub.failed == ["stage2", "stage3"]
    assert sub.ok("stage1") and not sub.ok("stage3")
    assert np.allclose(pd.read_csv(path)[["x", "y"]], 0.1)
