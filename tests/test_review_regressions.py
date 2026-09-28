"""Regression tests for issues found in code review of PR #4."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from kaggle_tools import kernel
from kaggle_tools.cv import TRAIN_ONLY, rare_safe_folds
from kaggle_tools.features import date_features
from kaggle_tools.metrics import best_threshold
from kaggle_tools.submission import ProgressiveSubmission, validate_submission
from kaggle_tools.tracking import Ledger


def test_best_threshold_is_f1_on_imbalanced_binary():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.5 * y + rng.normal(0.3, 0.2, 2000), 0, 1)
    thr, score = best_threshold(y, p)
    assert score == pytest.approx(f1_score(y, (p >= thr).astype(int)))
    grid = np.linspace(0.05, 0.95, 19)
    assert score == pytest.approx(max(f1_score(y, (p >= t).astype(int)) for t in grid))


def test_rare_row_pins_its_whole_group():
    labels = np.array(["rare"] + ["a"] * 60 + ["b"] * 60)
    groups = np.concatenate([[0, 0, 0], np.arange(1, 1 + 118) // 2 + 1])
    folds = rare_safe_folds(labels, groups=groups, n_splits=5)
    per_group = pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"].nunique()
    assert (per_group == 1).all()
    assert (folds[groups == 0] == TRAIN_ONLY).all()


def test_ledger_delta_survives_unscored_rows(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.log(kernel="k", version=1, change="a", lb=0.80, status="SCORED")
    led.log(kernel="k", version=2, change="b", status="ERROR")
    led.log(kernel="k", version=3, change="c", lb=0.85, status="SCORED")
    assert led.table()["delta_vs_best"].iloc[2] == pytest.approx(0.05)


def test_ledger_update_reports_missing_entry(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.log(kernel="k", version=3, change="a")
    with pytest.raises(KeyError):
        led.update("k", 4, lb=0.9)
    assert led.update("k", 3, lb=0.9) == 1


def test_competition_prefixed_dataset_is_not_flagged():
    src = "p = '/kaggle/input/birdclef-2026-perch-onnx/model.onnx'\nsub.to_csv('submission.csv')"
    assert kernel.check_source(src, ["birdclef-2026"]) == []


def test_inference_server_notebook_needs_no_submission_write():
    src = (
        "import kaggle_evaluation.aimo_3_inference_server as srv\n"
        "srv.AIMO3InferenceServer(p).serve()"
    )
    assert kernel.check_source(src, ["aimo"]) == []


def test_reading_sample_submission_is_not_writing():
    src = "s = pd.read_csv('/kaggle/input/competitions/c/sample_submission.csv')"
    assert any("submission" in f.message for f in kernel.check_source(src, ["c"]))


def test_status_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(kernel, "_kaggle", lambda *a: 'me/k has status "complete"')
    assert kernel.status("me/k") == "COMPLETE"


def test_ids_compared_as_strings():
    sample = pd.DataFrame({"id": [1, 2], "t": 0.5})
    sub = pd.DataFrame({"id": ["1", "2"], "t": 0.7})
    assert validate_submission(sub, sample) == []


def test_failed_stage_output_is_not_kept(tmp_path):
    sample = pd.DataFrame({"id": [1, 2], "t": 0.5})
    path = tmp_path / "s.csv"
    sub = ProgressiveSubmission(sample, path)
    with sub.stage("good"):
        sub.write(np.array([0.9, 0.9]))
    with sub.stage("bad"):
        sub.write(np.array([0.1, 0.1]))
        raise RuntimeError("boom")
    assert np.allclose(pd.read_csv(path)["t"], 0.9)
    assert sub.last_stage == "good"


def test_date_features_handle_nat():
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-01-03", None])})
    out = date_features(df, "ts")
    assert out["ts_week"].isna().iloc[1]


def test_train_only_rows_are_encoded_leave_one_out():
    from kaggle_tools.features import oof_target_encode

    df = pd.DataFrame({"c": ["a", "a", "a"], "y": [1.0, 0.0, 0.0]})
    enc = oof_target_encode(df, "c", "y", folds=np.array([-1, -1, -1]), smoothing=0)
    assert enc.tolist() == [0.0, 0.5, 0.5]  # each row excludes its own target


def test_ensemble_guards():
    from kaggle_tools.ensemble import hill_climb, rank_blend

    with pytest.raises(ValueError, match="NaN"):
        rank_blend([np.array([0.1, np.nan])])
    with pytest.raises(ValueError, match="weights"):
        rank_blend([np.ones(2), np.ones(2)], [0, 0])
    with pytest.raises(ValueError, match="n_iter"):
        hill_climb([np.ones(2)], np.array([0, 1]), lambda y, p: 0.0, n_iter=0)


def test_rerun_flag_parsing(monkeypatch):
    from kaggle_tools.env import is_competition_rerun

    for value, expected in [("1", True), ("True", True), ("0", False), ("false", False)]:
        monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", value)
        assert is_competition_rerun() is expected
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN")
    assert is_competition_rerun() is False


def test_offline_wheels_dedupe_and_conflicts(tmp_path, monkeypatch):
    import subprocess

    from kaggle_tools.env import install_offline_wheels

    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "foo-1.0-py3-none-any.whl").touch()
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))
    assert len(install_offline_wheels("foo-*.whl", root=tmp_path)) == 1
    (tmp_path / "b" / "foo-2.0-py3-none-any.whl").touch()
    with pytest.raises(ValueError, match="narrow"):
        install_offline_wheels("foo-*.whl", root=tmp_path)


def test_boxed_variants():
    from kaggle_tools.llm import extract_integer_answer

    assert extract_integer_answer(r"\boxed{12,345}") == 12345
    assert extract_integer_answer(r"\boxed{\text{42}}") == 42
    assert extract_integer_answer(r"\boxed{-3}", modulus=100000) == 99997


def test_kernel_edge_cases(tmp_path):
    (tmp_path / "kernel-metadata.json").write_text("{not json")
    assert "not valid JSON" in kernel.check_kernel_dir(tmp_path)[0].message
    src = (
        "!pip3 install foo\n!uv pip install bar\n"
        "subprocess.run(['pip', 'install', 'x',\n '--no-index'])"
    )
    flagged = [f for f in kernel.check_source(src) if "pip install" in f.message]
    assert len(flagged) == 2  # online !pip3 and !uv pip; the offline subprocess call is fine
    src = "!pip install online\n!pip install --no-index /kaggle/input/w/x.whl"
    assert len(kernel.check_source(src)) == 1  # look-ahead doesn't borrow the next line


def test_adapter_zip_counts_as_submission_write():
    src = 'with zipfile.ZipFile("/kaggle/working/submission.zip", "w") as zf:\n    zf.write(f)'
    assert kernel.check_source(src, ["nemotron"]) == []
