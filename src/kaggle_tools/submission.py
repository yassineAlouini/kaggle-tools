"""Submission building, validation and progressive "safety writes".

Lessons encoded here:

* Build submissions with one vectorised assignment. A per-row ``df.iloc[i, 1:] = ...``
  loop is O(n^2) in pandas and silently timed out several BirdCLEF re-runs.
* Write a valid submission *after every stage* (prior -> stage 1 -> ... -> final), so a
  crash or timeout later in the pipeline still leaves something that scores.
* Write atomically (tmp file + ``os.replace``) so a kill mid-write never leaves a
  truncated CSV.
"""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


def build_submission(
    ids: Sequence[object], preds: np.ndarray, columns: Sequence[str], id_col: str = "row_id"
) -> pd.DataFrame:
    """Build a submission frame with a single vectorised assignment."""
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds[:, None]
    if preds.shape != (len(ids), len(columns)):
        raise ValueError(f"preds shape {preds.shape} != ({len(ids)}, {len(columns)})")
    df = pd.DataFrame(preds, columns=list(columns))
    df.insert(0, id_col, list(ids))
    return df


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list means the submission looks valid)."""
    problems: list[str] = []
    if list(sub.columns) != list(sample.columns):
        problems.append("columns differ from sample_submission (names or order)")
    if len(sub) != len(sample):
        problems.append(f"row count {len(sub)} != sample {len(sample)}")
    id_col = sample.columns[0]
    if id_col in sub.columns and set(sub[id_col]) != set(sample[id_col]):
        problems.append(f"{id_col} values differ from sample_submission")
    numeric = sub.select_dtypes("number")
    if numeric.isna().any().any():
        problems.append("submission contains NaN")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        problems.append("submission contains inf")
    return problems


def atomic_write_csv(df: pd.DataFrame, path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


class ProgressiveSubmission:
    """Keep a valid submission on disk at every stage of an inference pipeline.

    >>> sub = ProgressiveSubmission(sample, "submission.csv")  # writes the prior
    >>> with sub.stage("backbone"):
    ...     sub.write(backbone_preds)
    >>> if sub.ok("backbone"):
    ...     with sub.stage("ensemble"):
    ...         sub.write(ensemble_preds)

    A failing stage is logged (with traceback) and swallowed, so the last good
    submission stays on disk instead of becoming a "Submission Scoring Error".
    """

    def __init__(self, sample: pd.DataFrame, path: str | os.PathLike[str] = "submission.csv"):
        self.sample = sample.reset_index(drop=True)
        self.path = Path(path)
        self.id_col = sample.columns[0]
        self.columns = list(sample.columns[1:])
        self.completed: list[str] = []
        self.failed: list[str] = []
        self.last_stage = "prior"
        atomic_write_csv(self.sample, self.path)

    def write(self, preds: np.ndarray | pd.DataFrame) -> None:
        df = (
            preds
            if isinstance(preds, pd.DataFrame)
            else build_submission(self.sample[self.id_col], preds, self.columns, self.id_col)
        )
        problems = validate_submission(df, self.sample)
        if problems:
            raise ValueError("; ".join(problems))
        atomic_write_csv(df, self.path)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        except Exception:
            self.failed.append(name)
            print(f"[submission] stage {name!r} FAILED — keeping {self.last_stage!r} output")
            traceback.print_exc()
        else:
            self.completed.append(name)
            self.last_stage = name
            print(f"[submission] stage {name!r} ok ({time.perf_counter() - start:.1f}s)")

    def ok(self, name: str) -> bool:
        return name in self.completed
