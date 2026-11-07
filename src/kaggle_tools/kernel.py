"""Pre-push checks for Kaggle kernels, plus thin wrappers over the ``kaggle`` CLI.

Every check below corresponds to a real failed or wasted submission:

* internet enabled on a code-competition kernel -> cannot be submitted;
* GPU kernel without ``machine_shape`` -> default P100 (sm_60), which stock torch
  >= 2.10 no longer supports (``cudaErrorNoKernelImageForDevice``). Pin a T4;
* ``/kaggle/input/<competition-slug>/`` -> competition data lives under
  ``/kaggle/input/competitions/<slug>/`` (resolve with ``env.competition_dir``);
* no ``submission.csv`` written -> the "Submit to Competition" button stays disabled;
* bare ``assert`` / unguarded ``pip install`` -> opaque "Submission Scoring Error" in
  the offline re-run;
* ``kernel_sources`` used to consume another notebook's *predictions* -> frozen at the
  parent's last run, never recomputed on the hidden test set;
* a recursive glob / ``os.walk`` over ``/kaggle/input`` to find weights -> walked ~10^5
  DICOMs of the competition mount for 9 minutes (RSNA Knee 2026).

The CLI wrappers at the bottom add what the same competition needed: submitting a
saved version of a code-competition kernel (``submit``; ``-v`` is mandatory with
``-k``) and downloading multi-GB outputs that the CLI truncated (``download_resumable``:
``curl --fail --location -C -`` with bounded retries, a size probe that requires HTTP 206
and a valid ``Content-Range``, and an optional sha256).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# "error" fails `kt check`; "warning" fails only with --strict; "info" never fails.
Level = Literal["error", "warning", "info"]
METADATA_FILE = "kernel-metadata.json"
# `!pip install`, `%pip3 install`, `!uv pip install`, `python -m pip install`,
# or subprocess ["pip", "install", ...]
_PIP_MAGIC = re.compile(r"^\s*[!%]\s*(?:uv\s+)?pip3?\s+install")
_PIP_INSTALL = re.compile(
    r"""^\s*[!%]\s*(?:uv\s+)?pip3?\s+install|-m\s+pip3?\s+install|uv\s+pip\s+install"""
    r"""|["']pip3?["']\s*,\s*["']install["']"""
)
_OFFLINE_TOKENS = ("--no-index", ".whl", "--find-links", "--offline")
_SUBMISSION_FILE = re.compile(r"(?<!sample_)submission\.(?:csv|parquet|zip)")
_WRITE_CALL = re.compile(
    r"\.to_csv\(|\.to_parquet\(|\.write_csv\(|\.write_parquet\(|ZipFile\(|make_archive\("
)
_INFERENCE_SERVER = re.compile(r"kaggle_evaluation|InferenceServer")
_INPUT_SCAN = re.compile(
    # glob("/kaggle/input/**", recursive=True)
    r"""glob\([^\n]*/kaggle/input[^\n]*recursive\s*=\s*True"""
    r"""|os\.walk\(\s*["']/kaggle/input/?["']"""  # os.walk("/kaggle/input")
    r"""|["']/kaggle/input/?["']\s*\)\s*\.rglob\("""  # Path("/kaggle/input").rglob(...)
)
_PUSHED_VERSION = re.compile(r"[Kk]ernel version (\d+) successfully pushed")
_CONTENT_RANGE = re.compile(
    r"^content-range:\s*bytes\s+0-0/(\d+)\s*$", re.IGNORECASE | re.MULTILINE
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
                "info",
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
        "".join(src) if isinstance(src := cell.get("source", ""), list) else src
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def check_source(source: str, competitions: list[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    # Inference-server competitions (AIMO, ...) have the gateway write the submission.
    uses_server = bool(_INFERENCE_SERVER.search(source))
    writes_submission = "ProgressiveSubmission" in source or (
        bool(_SUBMISSION_FILE.search(source)) and bool(_WRITE_CALL.search(source))
    )
    if competitions and not uses_server and not writes_submission:
        findings.append(
            Finding(
                "warning",
                "no submission.csv/parquet/zip write found (to_csv/to_parquet/ZipFile of a "
                "'submission.*' path); without it the Submit button stays disabled",
            )
        )
    for slug in competitions or []:
        if re.search(rf"""/kaggle/input/{re.escape(slug)}(?:/|['"]|$)""", source, re.MULTILINE):
            findings.append(
                Finding(
                    "warning",
                    f"/kaggle/input/{slug}/ is only one of the competition mount layouts "
                    f"(also /kaggle/input/competitions/{slug}/); resolve it with "
                    f"kaggle_tools.env.competition_dir instead of hard-coding",
                )
            )
    for line in source.splitlines():
        if _INPUT_SCAN.search(line):
            findings.append(
                Finding(
                    "warning",
                    "recursive scan of /kaggle/input: the competition mount can hold ~10^5 files "
                    f"(9 min on RSNA Knee); search attached dataset dirs only: {line.strip()[:80]}",
                )
            )
            break
    if re.search(r"^\s*assert\s", source, flags=re.MULTILINE):
        findings.append(
            Finding("warning", "bare assert: a failure in the re-run is an opaque scoring error")
        )
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if not _PIP_INSTALL.search(line):
            continue
        call = [line]
        if not _PIP_MAGIC.search(line):  # subprocess.run([...]) may span a few lines
            for nxt in lines[i + 1 : i + 4]:
                if _PIP_INSTALL.search(nxt):
                    break
                call.append(nxt)
        if not any(tok in " ".join(call) for tok in _OFFLINE_TOKENS):
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
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        return [Finding("error", f"{meta_path} is not valid JSON: {exc}")]
    findings = check_metadata(meta, kernel_dir)
    code = kernel_dir / meta.get("code_file", "")
    if code.is_file():
        try:
            source = notebook_source(code)
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            return [*findings, Finding("error", f"could not parse {code}: {exc}")]
        findings += check_source(source, meta.get("competition_sources"))
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
    match = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', out)
    return match.group(1).upper() if match else out.strip()


def wait(
    kernel_id: str,
    poll_seconds: float = 60,
    timeout_seconds: float = 13 * 3600,
    max_consecutive_errors: int = 5,
) -> str:
    """Poll until the kernel reaches a terminal state; returns that state.

    Transient CLI failures (network, API hiccups) are retried; only
    ``max_consecutive_errors`` failures in a row are re-raised.
    """
    deadline = time.monotonic() + timeout_seconds
    errors = 0
    while True:
        try:
            state = status(kernel_id)
            errors = 0
        except subprocess.CalledProcessError:
            errors += 1
            if errors >= max_consecutive_errors:
                raise
            state = "UNKNOWN"
        if state in TERMINAL_STATES or time.monotonic() > deadline:
            return state
        time.sleep(poll_seconds)


def output(kernel_id: str, dest: str | Path) -> str:
    """Download a kernel's output files and log (always read the log after an ERROR)."""
    return _kaggle("kernels", "output", kernel_id, "-p", str(dest))


def parse_pushed_version(push_output: str) -> int | None:
    """Version from ``kaggle kernels push`` output: "Kernel version N successfully pushed"."""
    match = _PUSHED_VERSION.search(push_output)
    return int(match.group(1)) if match else None


def submit(
    competition: str,
    kernel_id: str,
    version: int,
    message: str,
    output_file: str = "submission.csv",
) -> str:
    """Submit a saved, COMPLETE kernel version to a code competition.

    ``output_file`` names that version's output, not a local file to upload. Check the
    version's status and log first (``status``/``output``); submitting the latest
    version while another push is running can pick the wrong one.
    """
    if int(version) < 1:
        raise ValueError("version must be the pushed kernel version number (>= 1)")
    return _kaggle(
        "competitions", "submit", competition,
        "-k", kernel_id, "-v", str(int(version)), "-f", output_file, "-m", message,
    )  # fmt: skip


def _parse_range_probe(headers: str) -> int | None:
    """Total size from the headers of a ``Range: bytes=0-0`` GET, or None.

    Requires the final response to be HTTP 206 with ``Content-Range: bytes 0-0/<total>``;
    a server that ignores Range (200) or omits the header gives None.
    """
    blocks = [b for b in re.split(r"\r?\n\r?\n", headers.strip()) if b.strip()]
    if not blocks:
        return None
    last = blocks[-1]
    status_line = last.splitlines()[0]
    if not re.match(r"HTTP/\S+\s+206\b", status_line):
        return None
    match = _CONTENT_RANGE.search(last)
    return int(match.group(1)) if match else None


def remote_size(url: str, timeout: float = 60) -> int | None:
    """Size of a remote file via a 1-byte ranged GET (signed URLs often reject HEAD)."""
    result = subprocess.run(
        ["curl", "--fail", "--location", "-sS", "-r", "0-0", "-D", "-", "-o", "/dev/null",
         "--max-time", str(timeout), url],
        capture_output=True, text=True,
    )  # fmt: skip
    return _parse_range_probe(result.stdout) if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(
    url: str,
    dest: str | Path,
    expected_size: int | None = None,
    sha256: str | None = None,
    retries: int = 20,
    wait_seconds: float = 5,
) -> Path:
    """Download (and resume) one immutable file until it has the expected size.

    ``expected_size`` defaults to ``remote_size(url)``; without either a size or a
    checksum the download is refused, because a truncated file would look complete.
    Raises on a size overshoot (the source changed), a checksum mismatch, or after
    ``retries`` attempts. Refresh an expired signed URL for the same version, never a
    different file.
    """
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    want = expected_size if expected_size is not None else remote_size(url)
    if want is None and sha256 is None:
        raise RuntimeError(
            "cannot verify completeness: no size (HTTP 206 probe failed) and no sha256"
        )
    for _ in range(max(1, retries)):
        have = out.stat().st_size if out.exists() else 0
        if want is not None and have > want:
            raise RuntimeError(f"{out} is larger than expected ({have} > {want}); source changed?")
        if want is not None and have == want:
            break
        subprocess.run(["curl", "--fail", "--location", "-sS", "-C", "-", "-o", str(out), url])
        if want is None:
            break
        if out.exists() and out.stat().st_size < want:
            time.sleep(wait_seconds)
    size = out.stat().st_size if out.exists() else 0
    if want is not None and size != want:
        raise RuntimeError(f"download incomplete after {retries} attempts: {size}/{want} bytes")
    if sha256 is not None and _sha256(out) != sha256.lower():
        raise RuntimeError(f"sha256 mismatch for {out}")
    return out
