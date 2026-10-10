import subprocess
import sys
import textwrap
from multiprocessing import get_context

import pandas as pd
import pytest

from kaggle_tools.runtime import (
    guarded_initializer,
    raise_if_init_failed,
    rehearsal_root,
    shard_ids,
    watchdog,
)


def _run(code: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_watchdog_kills_a_hung_block_and_dumps_tracebacks():
    r = _run("""
        import time
        from kaggle_tools.runtime import watchdog
        with watchdog(0.5):
            time.sleep(10)
        print("unreachable")
    """)
    assert r.returncode != 0
    assert "unreachable" not in r.stdout
    assert "Thread" in r.stderr or "Traceback" in r.stderr


def test_watchdog_is_cancelled_after_a_successful_block():
    r = _run("""
        import time
        from kaggle_tools.runtime import watchdog
        with watchdog(0.5):
            pass
        time.sleep(1.5)
        print("still alive")
    """)
    assert r.returncode == 0 and "still alive" in r.stdout


def test_watchdog_rejects_non_positive_limits():
    with pytest.raises(ValueError, match="seconds"), watchdog(0):
        pass


def _bad_init():
    raise NameError("undefined_thing")


def _good_init(value):
    import os

    os.environ["KT_TEST_INIT"] = value


def _task(x):
    raise_if_init_failed()
    return x * 2


def test_guarded_initializer_raises_in_the_parent_instead_of_hanging():
    pool = get_context("spawn").Pool(1, initializer=guarded_initializer(_bad_init))
    with pool, pytest.raises(RuntimeError, match="initialisation failed"):
        pool.apply_async(_task, (3,)).get(timeout=60)


def test_guarded_initializer_passes_through_on_success():
    init = guarded_initializer(_good_init)
    with get_context("spawn").Pool(1, initializer=init, initargs=("ok",)) as pool:
        assert pool.apply_async(_task, (3,)).get(timeout=60) == 6


def test_shard_ids_disjoint_complete_and_seeded():
    ids = [f"s{i:03d}" for i in range(103)]
    parts = [shard_ids(ids, p, 4, seed=7) for p in range(4)]
    assert sorted(x for p in parts for x in p) == sorted(ids)
    assert len({x for p in parts for x in p}) == len(ids)
    assert shard_ids(ids, 1, 4, seed=7) == parts[1]
    assert shard_ids(ids, 0, 4, seed=None) == sorted(ids)[0::4]
    with pytest.raises(ValueError, match="part"):
        shard_ids(ids, 4, 4)
    with pytest.raises(ValueError, match="duplicates"):
        shard_ids(["a", "a"], 0, 2)


def test_rehearsal_root_filters_tables_and_links_dirs(tmp_path):
    comp = tmp_path / "comp"
    (comp / "train_series" / "s1").mkdir(parents=True)
    pd.DataFrame({"StudyId": ["s1", "s2", "s3"], "y": [1, 0, 1]}).to_csv(
        comp / "train.csv", index=False
    )
    (comp / "sample_submission.csv").write_text("StudyId,y\n")
    root = rehearsal_root(
        tmp_path / "fake",
        ids=["s1", "s3"],
        id_column="StudyId",
        tables={"test.csv": comp / "train.csv"},
        links={"test_series": comp / "train_series"},
        copies=[comp / "sample_submission.csv"],
    )
    assert pd.read_csv(root / "test.csv").StudyId.tolist() == ["s1", "s3"]
    assert (root / "test_series" / "s1").is_dir() and (root / "test_series").is_symlink()
    assert (root / "sample_submission.csv").read_text() == "StudyId,y\n"
    with pytest.raises(ValueError, match="no rows"):
        rehearsal_root(tmp_path / "x", ["zz"], "StudyId", {"test.csv": comp / "train.csv"})
