"""Wall-clock budgeting for time-limited notebooks.

Kaggle kills a kernel at its hard limit with no output, and scoring workers vary in
speed: the same code can pass once and TIMEOUT the next time. Budget against the hard
limit minus a generous margin, and check the budget *inside* loops rather than only at
stage boundaries. Measure where the time goes before adding gates: a gate that always
fires (because the worker is always behind at that point) silently drops a stage.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class TimeBudget:
    """Track elapsed / remaining time against a fixed limit.

    >>> budget = TimeBudget(total_seconds=5 * 3600, margin_seconds=1000)
    >>> for problem in problems:
    ...     if not budget.has(60):
    ...         break  # fall back to a cheap default
    ...     solve(problem, timeout=budget.per_item(n_left))
    """

    def __init__(self, total_seconds: float, margin_seconds: float = 0.0):
        self.total = float(total_seconds)
        self.margin = float(margin_seconds)
        self.start = time.monotonic()
        self.marks: list[tuple[str, float]] = []

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.margin - self.elapsed)

    def has(self, seconds: float) -> bool:
        return self.remaining >= seconds

    def per_item(self, n_left: int, floor: float = 0.0) -> float:
        """Fair share of the remaining time for each of ``n_left`` items."""
        return max(floor, self.remaining / max(1, n_left))

    def mark(self, name: str) -> float:
        """Record and print a checkpoint; returns elapsed seconds."""
        elapsed = self.elapsed
        self.marks.append((name, elapsed))
        print(f"[budget] {name}: {elapsed:.0f}s elapsed, {self.remaining:.0f}s left")
        return elapsed


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator printing how long ``fn`` took."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            name = getattr(fn, "__qualname__", repr(fn))
            print(f"[timed] {name}: {time.perf_counter() - start:.2f}s")

    return wrapper
