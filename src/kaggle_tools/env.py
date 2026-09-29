"""Kaggle runtime detection and robust input-path resolution.

Kaggle has changed its ``/kaggle/input`` mount layout over time and uses different
layouts for competition data, older datasets and newer datasets:

* competition data:  ``/kaggle/input/competitions/<slug>/``
* older datasets:    ``/kaggle/input/datasets/<owner>/<slug>/``
* newer datasets:    ``/kaggle/input/<slug>/``
* models:            ``/kaggle/input/<model>/<framework>/<variation>/<version>/``

Hard-coding one layout is the #1 cause of notebooks that run locally and then fail
(silently) in the competition re-run. Always resolve through these helpers, which try
every known layout and fail loudly with the list of paths that were tried.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def is_kaggle() -> bool:
    """True when running inside a Kaggle kernel (interactive, batch or re-run)."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or KAGGLE_INPUT.is_dir()


def is_competition_rerun() -> bool:
    """True during the hidden-test scoring re-run of a code competition.

    In that run there is no internet, the hidden test set is mounted, and any
    uncaught exception becomes an opaque "Submission Scoring Error".
    """
    value = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "")
    return value.strip().lower() not in {"", "0", "false", "no"}


def working_dir() -> Path:
    """``/kaggle/working`` on Kaggle (the only writable dir), the CWD elsewhere."""
    return KAGGLE_WORKING if is_kaggle() else Path.cwd()


def first_existing(candidates: Iterable[str | os.PathLike[str]], what: str = "path") -> Path:
    """Return the first candidate that exists, or raise listing every candidate tried."""
    tried = [Path(c) for c in candidates]
    for path in tried:
        if path.exists():
            return path
    listing = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(f"Could not find {what}. Tried:\n  {listing}")


def competition_dir(
    slug: str,
    local_dir: str | os.PathLike[str] = "data",
    input_root: str | os.PathLike[str] = KAGGLE_INPUT,
) -> Path:
    """Locate competition data on Kaggle (any mount layout) or locally."""
    root = Path(input_root)
    return first_existing(
        [root / "competitions" / slug, root / slug, Path(local_dir)],
        what=f"competition data for {slug!r}",
    )


def dataset_dir(
    slug: str,
    owner: str | None = None,
    local_dir: str | os.PathLike[str] | None = None,
    input_root: str | os.PathLike[str] = KAGGLE_INPUT,
) -> Path:
    """Locate an attached dataset (or kernel output) under any known mount layout."""
    root = Path(input_root)
    candidates: list[Path] = []
    if owner:
        candidates.append(root / "datasets" / owner / slug)
    candidates.append(root / slug)
    if owner:
        # Kernel outputs attached via ``kernel_sources`` mount like datasets.
        candidates.append(root / "notebooks" / owner / slug)
    if local_dir is not None:
        candidates.append(Path(local_dir))
    return first_existing(candidates, what=f"dataset {owner + '/' if owner else ''}{slug}")


def find_files(pattern: str, root: str | os.PathLike[str] = KAGGLE_INPUT) -> list[Path]:
    """Recursively glob under ``root`` (sorted). Handy for wheels and checkpoints."""
    return sorted(Path(root).rglob(pattern))


def install_offline_wheels(
    *patterns: str,
    root: str | os.PathLike[str] = KAGGLE_INPUT,
    no_deps: bool = True,
) -> list[Path]:
    """``pip install --no-index`` every wheel matching ``patterns`` under ``root``.

    Code-competition re-runs have no internet, so every non-default package must be
    attached as a dataset and installed from the mounted wheel. Raises if a pattern
    matches nothing — a silent ImportError followed by a try/except fallback is how
    a submission quietly degrades to a uniform-prior score.

    Point ``root`` at the wheel dataset (e.g. ``dataset_dir("my-wheels", "me")``):
    searching all of ``/kaggle/input`` is slow when large datasets are attached.
    Identical wheel files mounted twice are installed once; two *different* versions
    of the same package raise, since pip would refuse to install both.
    """
    wheels: dict[str, Path] = {}
    for pattern in patterns:
        found = find_files(pattern, root)
        if not found:
            raise FileNotFoundError(f"No wheel matching {pattern!r} under {root}")
        for wheel in found:
            wheels.setdefault(wheel.name, wheel)
    by_package: dict[str, set[str]] = {}
    for name in wheels:
        package = name.split("-")[0].lower().replace("_", "-")
        by_package.setdefault(package, set()).add(name)
    conflicts = {pkg: sorted(names) for pkg, names in by_package.items() if len(names) > 1}
    if conflicts:
        raise ValueError(f"Several wheels for the same package, narrow the pattern: {conflicts}")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-index"]
    if no_deps:
        cmd.append("--no-deps")
    paths = list(wheels.values())
    subprocess.run([*cmd, *map(str, paths)], check=True)
    return paths
