"""In-kernel guards for long inference runs, from the RSNA Knee 2026 code competition.

* ``watchdog``: a rehearsal run once hung for hours on two T4s and burned the weekly GPU
  quota. ``faulthandler.dump_traceback_later(exit=True)`` uses a native thread, so it
  fires even when Python code holds the GIL; it is always cancelled on exit so a
  successful run cannot be killed later.
* ``guarded_initializer`` / ``raise_if_init_failed``: a ``multiprocessing.Pool``
  initializer raised ``NameError``; the pool silently respawned workers forever and the
  parent waited forever. The initializer's return value is discarded, so the failure is
  stored in worker state and re-raised by the task function, which reaches the parent.
* ``shard_ids``: large preprocessing was split across several kernels (saved-output
  limits). Shards come from one id manifest; sorted ids interleaved are not random, so
  shuffle once with a fixed seed when shards must be representative.
* ``rehearsal_root``: rehearse the hidden-test run on training rows posing as the test
  set. Only build it behind an explicit option; never select it implicitly in the
  submitted notebook.
"""

from __future__ import annotations

import faulthandler
import os
import random
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

_INIT_ERROR: dict[str, BaseException] = {}


@contextmanager
def watchdog(seconds: float, file: Any = None) -> Iterator[None]:
    """Dump every thread's traceback and exit the process if the block runs too long.

    Forced exit skips cleanup and does not kill child processes; use an external
    process-group timeout when children must die too.
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    faulthandler.dump_traceback_later(seconds, exit=True, file=file or sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


class guarded_initializer:  # lower-case: used like a function, initializer=guarded_initializer(f)
    """Wrap a ``Pool`` initializer so an exception is kept instead of respawning workers.

    Use with ``raise_if_init_failed()`` at the top of the task function::

        pool = Pool(n, initializer=guarded_initializer(setup), initargs=(cfg,))
        def task(x):
            raise_if_init_failed()
            ...

    A class (not a closure) so it pickles under the ``spawn`` start method too; ``init``
    must itself be a module-level function.
    """

    def __init__(self, init: Callable[..., Any]) -> None:
        self.init = init

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        _INIT_ERROR.pop("error", None)
        try:
            self.init(*args, **kwargs)
        except Exception as exc:  # re-raised from the task function
            _INIT_ERROR["error"] = exc


def raise_if_init_failed() -> None:
    """Re-raise, inside a task, the exception the guarded initializer stored."""
    exc = _INIT_ERROR.get("error")
    if exc is not None:
        raise RuntimeError(f"pool worker initialisation failed: {exc!r}") from exc


def shard_ids(ids: Sequence[str], part: int, n_parts: int, seed: int | None = None) -> list[str]:
    """Shard ``part`` of ``n_parts`` from one id manifest (sorted, then optionally shuffled).

    Every kernel must call this with the same ids, ``n_parts`` and ``seed``; the shards
    are then disjoint and cover every id.
    """
    if not 0 <= part < n_parts:
        raise ValueError(f"part must be in [0, {n_parts})")
    manifest = sorted(set(ids))
    if len(manifest) != len(ids):
        raise ValueError("ids contain duplicates")
    if seed is not None:
        random.Random(seed).shuffle(manifest)
    return manifest[part::n_parts]


def rehearsal_root(
    out_dir: str | os.PathLike[str],
    ids: Sequence[str],
    id_column: str,
    tables: Mapping[str, str | os.PathLike[str]],
    links: Mapping[str, str | os.PathLike[str]] | None = None,
    copies: Sequence[str | os.PathLike[str]] = (),
) -> Path:
    """Build a fake competition root whose "test" rows are the given training ids.

    ``tables`` maps an output file name (e.g. ``"test.csv"``) to a source csv, which is
    filtered to ``ids`` on ``id_column``; ``links`` maps an output name to a directory
    to symlink (e.g. ``{"test_series": comp / "train_series"}``); ``copies`` are copied
    as-is (e.g. ``sample_submission.csv``). Point the notebook at the returned path
    only through an explicit rehearsal option.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(ids)
    for name, src in tables.items():
        df = pd.read_csv(src, dtype={id_column: str})
        if id_column not in df.columns:
            raise KeyError(f"{src} has no column {id_column!r}")
        df = df[df[id_column].isin(wanted)]
        if df.empty:
            raise ValueError(f"no rows of {src} match the rehearsal ids")
        df.to_csv(out / name, index=False)
    for name, target in (links or {}).items():
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(target).resolve(), target_is_directory=True)
    for src in copies:
        (out / Path(src).name).write_bytes(Path(src).read_bytes())
    return out
