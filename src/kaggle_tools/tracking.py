"""A plain-file submission ledger, with optional Trackio logging.

The most useful experiment tracker in a competition is an honest ledger of *every*
submission, including regressions, timeouts and "no score" runs, keyed by the Kaggle
kernel version (the integer ``kaggle kernels push`` assigns, not a private "v" number
that drifts). It shows which offline gains carried over to the LB and which did not.

The ledger is JSONL so it diffs well in git and stays readable by humans and agents.
For live training curves, install the ``tracking`` extra to use Trackio (a local-first,
wandb-compatible tracker) through :func:`trackio_run`.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Submission:
    kernel: str
    version: int | None
    change: str
    cv: float | None = None
    lb: float | None = None
    status: str = "PENDING"  # PENDING | SCORED | ERROR | TIMEOUT | NO_SCORE
    date: str = field(default_factory=lambda: dt.date.today().isoformat())
    extra: dict[str, Any] = field(default_factory=dict)


class Ledger:
    def __init__(self, path: str | Path = "submissions.jsonl", higher_is_better: bool = True):
        self.path = Path(path)
        self.higher_is_better = higher_is_better

    def log(self, **fields: Any) -> Submission:
        entry = Submission(**fields)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def update(self, kernel: str, version: int, **fields: Any) -> None:
        """Fill in LB / status for an earlier entry (e.g. once scoring finishes)."""
        rows = self.rows()
        for row in rows:
            if row["kernel"] == kernel and row["version"] == version:
                row.update(fields)
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def table(self) -> pd.DataFrame:
        """All entries plus the LB delta against the best score *before* each entry."""
        df = pd.DataFrame(self.rows())
        if df.empty:
            return df
        lb = pd.to_numeric(df["lb"], errors="coerce")
        running = (lb.cummax() if self.higher_is_better else lb.cummin()).shift()
        df["delta_vs_best"] = lb - running
        return df

    def best(self) -> dict[str, Any] | None:
        scored = [r for r in self.rows() if r.get("lb") is not None]
        if not scored:
            return None
        pick = max if self.higher_is_better else min
        return pick(scored, key=lambda r: r["lb"])


def trackio_run(project: str, config: dict[str, Any] | None = None, **kwargs: Any):
    """Start a Trackio run (``pip install 'kaggle-tools[tracking]'``).

    Trackio mirrors the wandb API: ``run = trackio_run("birdclef"); trackio.log({...});
    trackio.finish()``. Offline on Kaggle it logs locally; copy the files out of
    ``/kaggle/working`` if you want to keep them.
    """
    import trackio  # ty: ignore[unresolved-import]

    return trackio.init(project=project, config=config or {}, **kwargs)
