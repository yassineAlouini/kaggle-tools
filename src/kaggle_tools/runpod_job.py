"""Run one batch job on a rented RunPod GPU pod, and never leave the pod running.

Lessons from renting pods for training runs that a local GPU could not take:

- **A pod that outlives its job bills by the hour.** The local watcher can crash, the laptop can
  sleep, a scheduler can kill the step. So the pod gets a dead man's switch of its own (it removes
  itself after ``max_hours``, or shortly after the job writes ``DONE``/``FAILED``), the watcher
  terminates it in a ``finally`` (``SIGTERM`` included) and then checks with the API that it is
  gone.
- **On a shared account, only touch your own pods.** Pods are created and terminated only under a
  name prefix; nothing else on the account is ever listed for removal.
- **Check the money before you start.** The run refuses to launch when the balance minus the worst
  case (price cap x GPUs x ``max_hours``) would fall below a floor, and every pod gets a ledger row
  with its price, hours and cost, failures included.
- **Secure-cloud stock is thin.** Minimum vCPU / memory filters or a CUDA-version filter can make
  every GPU type report "no instances available". Ask for 1 vCPU / 1 GB, use the image's own CUDA
  build, and cycle a list of GPU types with retries.
- **``nproc`` lies inside a pod.** It shows the host's cores (64-96); the real share is the cgroup
  quota (``/sys/fs/cgroup/cpu.max``), often 8-18 vCPU. The job script exports it as ``NPROC``.
- **Decide on flags, not on log tails.** The job ends when the wrapper writes an explicit ``DONE``
  or ``FAILED`` file; a log line that happens to say "done" never ends the run early.

Needs the ``runpod`` extra (``uv sync --extra runpod``), ``ssh``/``scp`` on the path, and an SSH key
registered in the RunPod account. The API key is read from ``RUNPOD_API_KEY`` or
``~/.runpod/config.toml`` (``api_key = "..."``) and is never printed, logged or sent to the pod.

A job is described by a TOML file (paths are relative to the file)::

    name = "fp-fold0"                      # pod name = prefix + name
    gpu_types = ["NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA A40"]
    max_hours = 3.0
    price_cap = 1.5                        # $/GPU-hour, used only for the budget check
    setup = "pip install -q rdkit polars"
    command = "python train.py --out out"
    outputs = ["out"]                      # remote paths (relative to the work dir) pulled back
    local_dir = "runs/pod"

    [[upload]]
    src = "src/train.py"
    dst = "train.py"

Then ``kt runpod run job.toml``; ``--dry-run`` prints the plan and the budget and creates nothing.
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import signal
import subprocess
import tarfile
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPUS = (
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
)
DONE_FLAG, FAILED_FLAG = "KT_JOB_DONE_FLAG", "KT_JOB_FAILED_FLAG"
NO_STOCK = re.compile(
    r"no longer any instances|no instances|not enough|unavailable|out of stock", re.I
)
SSH_OPTS = (
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "LogLevel=ERROR",
)


@dataclass
class Upload:
    src: Path
    dst: str


@dataclass
class JobSpec:
    """Everything a pod run needs; build it with :func:`load_spec` or directly."""

    name: str
    command: str
    prefix: str = "kt-"
    gpu_types: list[str] = field(default_factory=lambda: list(DEFAULT_GPUS))
    gpu_count: int = 1
    image: str = DEFAULT_IMAGE
    cloud_type: str = "SECURE"
    container_disk_gb: int = 80
    max_hours: float = 3.0
    price_cap: float = 2.0
    floor: float = 5.0
    setup: str = ""
    uploads: list[Upload] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    workdir: str = "/workspace/job"
    local_dir: Path = Path("runs/pod")
    poll_seconds: int = 180
    pull_while_running: bool = True
    grace_minutes: int = 45
    create_rounds: int = 6
    round_wait_seconds: int = 120

    @property
    def pod_name(self) -> str:
        return self.prefix + self.name

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.name):
            raise ValueError(f"job name must be [A-Za-z0-9._-]: {self.name!r}")
        if not self.prefix:
            raise ValueError("prefix must not be empty: it is what keeps other pods safe")
        if self.max_hours <= 0 or self.price_cap <= 0 or self.gpu_count < 1:
            raise ValueError("max_hours, price_cap and gpu_count must be positive")
        if not self.gpu_types:
            raise ValueError("gpu_types is empty")
        for u in self.uploads:
            if not u.src.exists():
                raise ValueError(f"upload source missing: {u.src}")
            if Path(u.dst).is_absolute() or ".." in Path(u.dst).parts:
                raise ValueError(f"upload destination must stay inside the work dir: {u.dst!r}")
        for o in self.outputs:
            if Path(o).is_absolute() or ".." in Path(o).parts:
                raise ValueError(f"output path must stay inside the work dir: {o!r}")


def load_spec(path: str | Path) -> JobSpec:
    """Read a job TOML file; relative ``src`` and ``local_dir`` paths resolve against its folder."""
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    base = path.resolve().parent
    uploads = [Upload(src=(base / u["src"]).resolve(), dst=u["dst"]) for u in raw.pop("upload", [])]
    if "local_dir" in raw:
        raw["local_dir"] = (base / raw["local_dir"]).resolve()
    known = set(JobSpec.__dataclass_fields__) - {"uploads"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown job keys: {sorted(unknown)}")
    spec = JobSpec(uploads=uploads, **raw)
    spec.validate()
    return spec


def worst_case_cost(spec: JobSpec) -> float:
    return spec.price_cap * spec.gpu_count * spec.max_hours


def budget_ok(balance: float, spec: JobSpec) -> tuple[bool, float]:
    """(launch allowed, worst-case cost): refuse when balance - worst case < floor."""
    worst = worst_case_cost(spec)
    return balance - worst >= spec.floor, worst


def parse_cpu_max(text: str) -> int | None:
    """vCPU share from a cgroup v2 ``cpu.max`` line (``"800000 100000"`` -> 8), None if unlimited"""
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    quota, period = int(parts[0]), int(parts[1])
    return max(1, -(-quota // period))


def deadman_script(spec: JobSpec) -> str:
    """Pod-side switch: remove the pod after max_hours, or grace_minutes after DONE/FAILED."""
    kt = f"{spec.workdir}/.kt"
    return f"""#!/bin/bash
# kaggle_tools.runpod_job dead man's switch: runs detached on the pod, independent of the watcher.
MAX={int(spec.max_hours * 3600)}; GRACE={spec.grace_minutes * 60}
DEADLINE=$(( $(date +%s) + MAX ))
while IFS='=' read -r key value; do
  case "$key" in RUNPOD_POD_ID|RUNPOD_API_KEY) export "$key=$value" ;; esac
done < <(tr '\\0' '\\n' < /proc/1/environ)
[ -n "$RUNPOD_POD_ID" ] || {{ echo "no RUNPOD_POD_ID: deadman cannot work"; exit 1; }}
command -v runpodctl >/dev/null || {{ echo "no runpodctl: deadman cannot work"; exit 1; }}
echo "deadman for $RUNPOD_POD_ID: max $MAX s, started $(date -u +%FT%TZ)"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if [ -e {kt}/DONE ] || [ -e {kt}/FAILED ]; then
    echo "job finished at $(date -u +%FT%TZ); removing within $GRACE s"
    [ "$(( $(date +%s) + GRACE ))" -ge "$DEADLINE" ] || DEADLINE=$(( $(date +%s) + GRACE ))
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 10; done
    break
  fi
  sleep 30
done
echo "removing pod at $(date -u +%FT%TZ)"
runpodctl remove pod "$RUNPOD_POD_ID"
"""


def job_script(spec: JobSpec) -> str:
    """Wrapper that exports NPROC (cgroup share) and writes .kt/DONE or .kt/FAILED."""
    return f"""#!/bin/bash
cd {shlex.quote(spec.workdir)}
mkdir -p .kt
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)
set -- $Q
if [ -n "$1" ] && [ "$1" != "max" ]; then export NPROC=$(( ($1 + $2 - 1) / $2 )); fi
export NPROC=${{NPROC:-${{KT_VCPU:-8}}}}
echo "start $(date -u +%FT%TZ) NPROC=$NPROC" > .kt/job.log
( {spec.command} ) >> .kt/job.log 2>&1
rc=$?
echo "exit $rc at $(date -u +%FT%TZ)" >> .kt/job.log
if [ $rc -eq 0 ]; then touch .kt/DONE; else touch .kt/FAILED; fi
"""


def poll_command(spec: JobSpec) -> str:
    kt = f"{shlex.quote(spec.workdir)}/.kt"
    return (
        f"test -f {kt}/DONE && echo {DONE_FLAG}; test -f {kt}/FAILED && echo {FAILED_FLAG}; "
        f"tail -n 3 {kt}/job.log 2>/dev/null; true"
    )


def job_status(poll_output: str) -> str | None:
    """'done' / 'failed' from explicit flag lines only, else None (still running)."""
    lines = {ln.strip() for ln in poll_output.splitlines()}
    if FAILED_FLAG in lines:
        return "failed"
    if DONE_FLAG in lines:
        return "done"
    return None


def read_api_key(config: Path | None = None) -> str:
    """API key from RUNPOD_API_KEY or the runpod config file; never logged."""
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    config = config or Path.home() / ".runpod" / "config.toml"
    m = (
        re.search(r"api_key\s*=\s*[\"']?([^\"'\s]+)", config.read_text())
        if config.exists()
        else None
    )
    if not m:
        raise ValueError("no RunPod API key: set RUNPOD_API_KEY or ~/.runpod/config.toml")
    return m.group(1)


class RunpodClient:
    """Thin wrapper over the ``runpod`` SDK (the only place it is imported)."""

    def __init__(self, api_key: str):
        import runpod  # ty: ignore[import-not-found]

        runpod.api_key = api_key
        self._rp = runpod

    def balance(self) -> float:
        from runpod.api.graphql import run_graphql_query  # ty: ignore[import-not-found]

        data = run_graphql_query("query { myself { clientBalance } }")
        return float(data["data"]["myself"]["clientBalance"])

    def create_pod(self, **kwargs: Any) -> dict:
        return self._rp.create_pod(**kwargs)

    def get_pod(self, pod_id: str) -> dict:
        return self._rp.get_pod(pod_id) or {}

    def get_pods(self) -> list[dict]:
        return self._rp.get_pods() or []

    def terminate_pod(self, pod_id: str) -> None:
        self._rp.terminate_pod(pod_id)


class Remote:
    """ssh / scp / tar-over-ssh to ``root@ip:port``."""

    def __init__(self, ip: str, port: int):
        self.ip, self.port = ip, port

    def sh(self, cmd: str, check: bool = True, timeout: int = 1800) -> str:
        r = subprocess.run(
            ["ssh", "-p", str(self.port), *SSH_OPTS, f"root@{self.ip}", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and r.returncode != 0:
            raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr[-1000:]}")
        return r.stdout

    def put(self, local: Path, remote: str) -> None:
        subprocess.run(
            [
                "scp",
                "-q",
                "-r",
                "-P",
                str(self.port),
                *SSH_OPTS,
                str(local),
                f"root@{self.ip}:{remote}",
            ],
            check=True,
            timeout=6 * 3600,
        )

    def pull(self, workdir: str, paths: list[str], dest: Path) -> None:
        if not paths:
            return
        dest.mkdir(parents=True, exist_ok=True)
        quoted = " ".join(shlex.quote(p) for p in paths)
        r = subprocess.run(
            [
                "ssh",
                "-p",
                str(self.port),
                *SSH_OPTS,
                f"root@{self.ip}",
                f"cd {shlex.quote(workdir)} && tar cf - --ignore-failed-read {quoted} 2>/dev/null",
            ],
            capture_output=True,
            timeout=6 * 3600,
            check=False,
        )
        if not r.stdout:
            raise RuntimeError(f"pull returned nothing (rc {r.returncode})")
        with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
            t.extractall(dest, filter="data")


def create_pod(
    spec: JobSpec, client: Any, log: Callable[..., None], sleep: Callable[[float], None]
) -> tuple[str, str]:
    """(pod id, GPU type): cycle gpu_types for create_rounds; only stock errors retry."""
    for rnd in range(spec.create_rounds):
        for gpu in spec.gpu_types:
            try:
                pod = client.create_pod(
                    name=spec.pod_name,
                    image_name=spec.image,
                    gpu_type_id=gpu,
                    gpu_count=spec.gpu_count,
                    cloud_type=spec.cloud_type,
                    container_disk_in_gb=spec.container_disk_gb,
                    min_vcpu_count=1,
                    min_memory_in_gb=1,
                    ports="22/tcp",
                    support_public_ip=True,
                    start_ssh=True,
                )
            except Exception as exc:  # the SDK raises plain errors for "no stock"
                if NO_STOCK.search(str(exc)):
                    log(f"no stock: {gpu}")
                    continue
                raise
            log(f"created {pod['id']} on {gpu}")
            return pod["id"], gpu
        if rnd < spec.create_rounds - 1:
            log(f"no GPU type available (round {rnd + 1}); retrying in {spec.round_wait_seconds} s")
            sleep(spec.round_wait_seconds)
    raise RuntimeError("no GPU available after every round")


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    """(public ip, port) of the pod's port 22, once the runtime reports it."""
    for p in (pod.get("runtime") or {}).get("ports") or []:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p["ip"], int(p["publicPort"])
    return None


def wait_ssh(
    pod_id: str, client: Any, sleep: Callable[[float], None], timeout_s: int = 1200
) -> tuple[str, int, dict]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        pod = client.get_pod(pod_id)
        ep = ssh_endpoint(pod)
        if ep:
            return ep[0], ep[1], pod
        sleep(10)
    raise RuntimeError(f"pod {pod_id} exposed no ssh port within {timeout_s} s")


def terminate(
    pod_id: str, client: Any, log: Callable[..., None], sleep: Callable[[float], None]
) -> bool:
    """Terminate with retries, then confirm the pod is no longer listed. True when it is gone."""
    for _ in range(5):
        try:
            client.terminate_pod(pod_id)
            break
        except Exception as exc:
            log(f"terminate failed, retrying: {str(exc)[:200]}")
            sleep(20)
    sleep(10)
    gone = not any(p.get("id") == pod_id for p in client.get_pods())
    log("pod gone (API check)" if gone else f"POD {pod_id} STILL LISTED: remove it by hand")
    return gone


def _exit_on_sigterm(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def run(
    spec: JobSpec,
    client: Any = None,
    remote_factory: Callable[[str, int], Any] = Remote,
    log: Callable[..., None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    dry_run: bool = False,
) -> str:
    """Budget, create, deadman, upload, setup, job, poll/pull, terminate; returns the status."""
    spec.validate()
    log = log or (lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True))
    client = client or RunpodClient(read_api_key())
    bal = client.balance()
    ok, worst = budget_ok(bal, spec)
    log(
        f"balance ${bal:.2f}; worst case ${worst:.2f} ({spec.max_hours} h); floor ${spec.floor:.2f}"
    )
    if not ok:
        raise RuntimeError(
            "refusing to launch: the balance would fall below the floor (top-up needed)"
        )
    if dry_run:
        log(f"dry run: pod {spec.pod_name}, GPUs {spec.gpu_types}, {len(spec.uploads)} upload(s)")
        return "dry-run"
    old = signal.signal(signal.SIGTERM, _exit_on_sigterm)  # a killed watcher still runs the finally
    pod_id, gpu, price, status, t0 = None, None, None, "error", time.monotonic()
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    try:
        pod_id, gpu = create_pod(spec, client, log, sleep)
        t0 = time.monotonic()
        ip, port, pod = wait_ssh(pod_id, client, sleep)
        price = pod.get("costPerHr")
        log(f"ssh ready; ${price}/h; {pod.get('vcpuCount')} vCPU")
        rem = remote_factory(ip, port)
        wd = shlex.quote(spec.workdir)
        rem.sh(f"mkdir -p {wd}/.kt")
        chk = rem.sh(
            "command -v runpodctl >/dev/null && echo ctl; tr '\\0' '\\n' < /proc/1/environ | "
            "grep -c -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)='",
            check=False,
        )
        if "ctl" not in chk or not chk.strip().endswith("2"):
            raise RuntimeError("deadman prerequisites missing on the pod (runpodctl / pod env)")
        rem.sh(
            f"cat > {wd}/.kt/deadman.sh <<'KT_EOF'\n{deadman_script(spec)}KT_EOF\n"
            f"chmod +x {wd}/.kt/deadman.sh && "
            f"(nohup {wd}/.kt/deadman.sh > {wd}/.kt/deadman.log 2>&1 &)"
        )
        for u in spec.uploads:
            parent = shlex.quote(str(Path(spec.workdir) / Path(u.dst).parent))
            rem.sh(f"mkdir -p {parent}")
            rem.put(u.src, f"{spec.workdir}/{u.dst}")
            log(f"uploaded {u.dst}")
        if spec.setup:
            out = rem.sh(
                f"cd {wd} && ( {spec.setup} ) > .kt/setup.log 2>&1; echo rc=$?", timeout=3600
            )
            if "rc=0" not in out:
                raise RuntimeError(
                    "setup failed: " + rem.sh(f"tail -n 30 {wd}/.kt/setup.log", check=False)
                )
        rem.sh(
            f"cat > {wd}/.kt/job.sh <<'KT_EOF'\n{job_script(spec)}KT_EOF\n"
            f"chmod +x {wd}/.kt/job.sh && (KT_VCPU={int(pod.get('vcpuCount') or 8)} "
            f"nohup {wd}/.kt/job.sh > /dev/null 2>&1 &)"
        )
        log("job started")
        while True:
            sleep(spec.poll_seconds)
            out = rem.sh(poll_command(spec), check=False)
            hrs = (time.monotonic() - t0) / 3600
            log(f"[{hrs:.2f} h ~${(price or 0) * hrs:.2f}] " + " | ".join(out.split("\n")[-3:]))
            st = job_status(out)
            if spec.pull_while_running and st is None:
                try:
                    rem.pull(spec.workdir, spec.outputs, spec.local_dir)
                except Exception as exc:
                    log(f"pull failed (will retry): {str(exc)[:200]}")
            if st:
                status = st
                break
            if hrs > spec.max_hours - 0.1:
                status = "max-hours"
                break
        rem.pull(spec.workdir, [*spec.outputs, ".kt"], spec.local_dir)
        log(f"pulled to {spec.local_dir}; status {status}")
    finally:
        if pod_id:
            terminate(pod_id, client, log, sleep)
        hrs = (time.monotonic() - t0) / 3600
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "name": spec.pod_name,
            "pod": pod_id,
            "gpu": gpu,
            "price": price,
            "hours": round(hrs, 3),
            "cost": round((price or 0) * hrs, 2),
            "status": status,
        }
        with open(spec.local_dir / "pods.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        log("ledger", row)
        signal.signal(signal.SIGTERM, old)
    return status


def list_own_pods(prefix: str, client: Any = None) -> list[dict]:
    """Pods on the account whose name starts with prefix (never anything else)."""
    if not prefix:
        raise ValueError("prefix must not be empty")
    client = client or RunpodClient(read_api_key())
    return [p for p in client.get_pods() if str(p.get("name", "")).startswith(prefix)]
