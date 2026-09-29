import numpy as np
import pandas as pd

from kaggle_tools.audio import frame_windows, parse_row_id, window_index, window_row_ids
from kaggle_tools.features import date_features, oof_target_encode
from kaggle_tools.llm import (
    ConsensusVoter,
    extract_integer_answer,
    majority_vote,
    weighted_vote,
)
from kaggle_tools.tracking import Ledger


def test_extract_answer():
    assert extract_integer_answer("so \\boxed{12} wait \\boxed{123456}", modulus=100000) == 23456
    assert extract_integer_answer("The final answer is: 42.") == 42
    assert extract_integer_answer("no idea") is None


def test_voting():
    assert majority_vote([None, 3, 3, 5]) == 3
    assert majority_vote([None, None], fallback=0) == 0
    v = ConsensusVoter(min_agree=2)
    assert not v.add(1) and not v.add(None) and v.add(1)
    assert v.result() == 1


def test_audio_windows():
    stem, end = parse_row_id("BC2026_Test_0001_S05_20250227_010002_5")
    assert stem == "BC2026_Test_0001_S05_20250227_010002" and end == 5
    assert window_index(end) == 0
    assert window_row_ids("f", 2) == ["f_5", "f_10"]
    assert frame_windows(np.ones(11), sample_rate=1, window_seconds=5).shape == (3, 5)


def test_ledger(tmp_path):
    led = Ledger(tmp_path / "subs.jsonl")
    led.log(kernel="k", version=1, change="baseline", lb=0.80, status="SCORED")
    led.log(kernel="k", version=2, change="rank blend", cv=0.9)
    led.update("k", 2, lb=0.85, status="SCORED")
    led.log(kernel="k", version=3, change="regression", lb=0.82, status="SCORED")
    t = led.table()
    assert led.best()["version"] == 2
    assert np.isclose(t["delta_vs_best"].iloc[2], -0.03)


def test_date_features_and_target_encoding():
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-01-03 23:30", "2026-06-15 12:00"])})
    out = date_features(df, "ts", tz="Europe/Paris")
    assert out["ts_day"].tolist() == [4, 15]  # UTC 23:30 -> next day in Paris
    assert {"ts_hour_sin", "ts_is_weekend", "ts_week"} <= set(out.columns)

    df = pd.DataFrame({"c": ["a", "a", "b", "b"], "y": [1, 1, 0, 0]})
    enc = oof_target_encode(df, "c", "y", folds=np.array([0, 1, 0, 1]), smoothing=0)
    assert enc.tolist() == [1, 1, 0, 0]


def test_weighted_vote_prefers_confident_chains():
    answers = [1, 1, 2, None]
    assert weighted_vote(answers, [1.0, 1.0, 0.1, 0.0]) == 2  # 10 > 1 + 1
    assert weighted_vote(answers, [0.5, 0.5, 2.0, 0.0]) == 1
    assert weighted_vote([None], [0.1], fallback=-1) == -1
