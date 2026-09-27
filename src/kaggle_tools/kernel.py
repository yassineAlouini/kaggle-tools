"""Pre-push checks for Kaggle kernels, plus thin wrappers over the ``kaggle`` CLI.

Every check below corresponds to a real failed or wasted submission:

* internet enabled on a code-competition kernel -> cannot be submitted;
* GPU kernel without ``machine_shape`` -> default P100 (sm_60), which stock torch
  >= 2.10 no longer supports (``cudaErrorNoKernelImageForDevice``). Pin a T4;
* ``/kaggle/input/<competition-slug>/`` -> competition data lives under
  ``/kaggle/input/competitions/<slug>/``;
* no ``submission.csv`` written -> the "Submit to Competition" button stays disabled;
* bare ``assert`` / unguarded ``pip install`` -> opaque "Submission Scoring Error" in
  the offline re-run;
* ``kernel_sources`` used to consume another notebook's *predictions* -> frozen at the
  parent's last run, never recomputed on the hidden test set.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Level = Literal["error", "warning"]
METADATA_FILE = "kernel-metadata.json"
# `!pip install`, `%pip install`, `python -m pip install`, or subprocess ["pip", "install"]
_PIP_INSTALL = re.compile(
    r"""^\s*[!%]pip\s+install|-m\s+pip\s+install|["']pip["']\s*,\s*["']install["']"""
)
TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}


@dataclass(frozen=True)
class Finding:
    level: Level
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7} {self.message}"


def metadata_template(
    kernel_id: str,
    competition: str,
    code_file: str = "notebook.ipynb",
    gpu: bool = True,
    machine_shape: str = "NvidiaTeslaT4",
) -> dict:
    """A code-competition ``kernel-metadata.json`` with safe defaults."""
    meta = {
        "id": kernel_id,
        "title": kernel_id.split("/", 1)[-1].replace("-", " ").title(),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook" if code_file.endswith(".ipynb") else "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [competition],
        "kernel_sources": [],
        "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = machine_shape
    return meta


def check_metadata(meta: dict, kernel_dir: Path | None = None) -> list[Finding]:
    findings: list[Finding] = []
    kid = meta.get("id", "")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", kid):
        findings.append(Finding("error", f"id {kid!r} must look like 'username/kernel-slug'"))
    competitions = meta.get("competition_sources") or []
    if not competitions:
        findings.append(Finding("warning", "no competition_sources: competition data won't mount"))
    if competitions and meta.get("enable_internet"):
        findings.append(
            Finding("error", "enable_internet=true: code competitions require internet OFF")
        )
    if meta.get("enable_gpu") and not meta.get("machine_shape"):
        findings.append(
            Finding(
                "warning",
                "GPU without machine_shape defaults to a P100 (sm_60), unsupported by stock "
                'torch>=2.10 — set "machine_shape": "NvidiaTeslaT4"',
            )
        )
    for key in ("dataset_sources", "kernel_sources"):
        for src in meta.get(key) or []:
            if "/" not in src:
                findings.append(Finding("error", f"{key} entry {src!r} must be 'owner/slug'"))
    if meta.get("kernel_sources"):
        findings.append(
            Finding(
                "warning",
                "kernel_sources outputs are frozen at the parent's last run; never use them "
                "for predictions that must be recomputed on the hidden test set",
            )
        )
    if kernel_dir is not None:
        code_file = meta.get("code_file")
        if not code_file or not (kernel_dir / code_file).is_file():
            findings.append(Finding("error", f"code_file {code_file!r} not found in {kernel_dir}"))
    return findings


def notebook_source(path: Path) -> str:
    """Concatenated code of a ``.ipynb`` (code cells only) or a plain script."""
    if path.suffix != ".ipynb":
        return path.read_text()
    nb = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def check_source(source: str, competitions: list[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    if competitions and "submission" not in source:
        findings.append(Finding("error", "source never mentions submission.csv/parquet"))
    for slug in competitions or []:
        if re.search(rf"/kaggle/input/{re.escape(slug)}\b", source):
            findings.append(
                Finding(
                    "error", f"use /kaggle/input/competitions/{slug}/, not /kaggle/input/{slug}/"
                )
            )
    if re.search(r"^\s*assert\s", source, flags=re.MULTILINE):
        findings.append(
            Finding("warning", "bare assert: a failure in the re-run is an opaque scoring error")
        )
    lines = source.splitlines()
    for i, line in enumerate(lines):
        call = " ".join(lines[i : i + 4])  # multi-line subprocess.run([...]) calls
        offline = any(tok in call for tok in ("--no-index", ".whl", "--find-links"))
        if _PIP_INSTALL.search(line) and not offline:
            findings.append(
                Finding(
                    "warning",
                    f"pip install without --no-index (no internet in re-run): {line.strip()[:80]}",
                )
            )
    return findings


def check_kernel_dir(kernel_dir: str | Path) -> list[Finding]:
    """Run every check on a directory containing ``kernel-metadata.json``."""
    kernel_dir = Path(kernel_dir)
    meta_path = kernel_dir / METADATA_FILE
    if not meta_path.is_file():
        return [Finding("error", f"{meta_path} not found")]
    meta = json.loads(meta_path.read_text())
    findings = check_metadata(meta, kernel_dir)
    code = kernel_dir / meta.get("code_file", "")
    if code.is_file():
        findings += check_source(notebook_source(code), meta.get("competition_sources"))
    return findings


# --- kaggle CLI wrappers -----------------------------------------------------------


def _kaggle(*args: str) -> str:
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True, check=True)
    return result.stdout


def push(kernel_dir: str | Path) -> str:
    return _kaggle("kernels", "push", "-p", str(kernel_dir))


def status(kernel_id: str) -> str:
    """Return the kernel status, e.g. ``RUNNING``, ``COMPLETE`` or ``ERROR``."""
    out = _kaggle("kernels", "status", kernel_id)
    match = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', out)
    return match.group(1) if match else out.strip()


def wait(kernel_id: str, poll_seconds: float = 60, timeout_seconds: float = 13 * 3600) -> str:
    """Poll until the kernel reaches a terminal state; returns that state."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = status(kernel_id)
        if state in TERMINAL_STATES or time.monotonic() > deadline:
            return state
        time.sleep(poll_seconds)


def output(kernel_id: str, dest: str | Path) -> str:
    """Download a kernel's output files and log (always read the log after an ERROR)."""
    return _kaggle("kernels", "output", kernel_id, "-p", str(dest))
