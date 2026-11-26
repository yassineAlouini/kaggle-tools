import json

import pytest

from kaggle_tools import runpod_job as rj
from kaggle_tools.cli import main


class FakeClient:
    def __init__(self, balance=100.0, stock_fail=0, pods=None):
        self._balance, self.stock_fail = balance, stock_fail
        self.created, self.terminated = [], []
        self.pods = pods or []

    def balance(self):
        return self._balance

    def create_pod(self, **kw):
        if self.stock_fail:
            self.stock_fail -= 1
            raise RuntimeError(
                "There are no longer any instances available with the requested specifications"
            )
        self.created.append(kw)
        self.pods.append({"id": "p1", "name": kw["name"]})
        return {"id": "p1"}

    def get_pod(self, pod_id):
        ports = [{"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 22, "publicPort": 2222}]
        return {"id": pod_id, "costPerHr": 0.8, "vcpuCount": 12, "runtime": {"ports": ports}}

    def get_pods(self):
        return [p for p in self.pods if p["id"] not in self.terminated]

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


class FakeRemote:
    def __init__(self, polls, ip=None, port=None, fail_on=None):
        self.polls, self.cmds, self.puts, self.pulls, self.fail_on = (
            list(polls),
            [],
            [],
            [],
            fail_on,
        )

    def sh(self, cmd, check=True, timeout=0):
        self.cmds.append(cmd)
        if self.fail_on and self.fail_on in cmd:
            raise RuntimeError("boom")
        if "runpodctl" in cmd and "grep -c" in cmd:
            return "ctl\n2\n"
        if "echo rc=$?" in cmd:
            return "rc=0\n"
        if rj.DONE_FLAG in cmd:
            return self.polls.pop(0) if self.polls else ""
        return ""

    def put(self, local, remote):
        self.puts.append((local, remote))

    def pull(self, workdir, paths, dest):
        self.pulls.append(list(paths))


def _spec(tmp_path, **kw):
    f = tmp_path / "train.py"
    f.write_text("print('hi')")
    base = {
        "name": "job1",
        "command": "python train.py",
        "uploads": [rj.Upload(f, "train.py")],
        "outputs": ["out"],
        "local_dir": tmp_path / "pod",
        "setup": "pip install -q x",
        "poll_seconds": 1,
    }
    return rj.JobSpec(**(base | kw))


def test_load_spec_resolves_paths_and_rejects_unknown_keys(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("")
    job = tmp_path / "job.toml"
    job.write_text(
        'name = "a"\ncommand = "true"\nlocal_dir = "runs"\n'
        '[[upload]]\nsrc = "src/train.py"\ndst = "t.py"\n'
    )
    spec = rj.load_spec(job)
    assert spec.uploads[0].src == (tmp_path / "src" / "train.py").resolve()
    assert spec.local_dir == (tmp_path / "runs").resolve()
    job.write_text('name = "a"\ncommand = "true"\nbogus = 1\n')
    with pytest.raises(ValueError, match="unknown"):
        rj.load_spec(job)


def test_validate_keeps_paths_inside_workdir(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        _spec(tmp_path, outputs=["../etc"]).validate()
    with pytest.raises(ValueError, match="prefix"):
        _spec(tmp_path, prefix="").validate()


def test_budget_and_cpu_quota(tmp_path):
    spec = _spec(tmp_path, price_cap=2.0, max_hours=3.0, floor=5.0)
    assert rj.worst_case_cost(spec) == 6.0
    assert rj.budget_ok(11.0, spec) == (True, 6.0)
    assert rj.budget_ok(10.9, spec)[0] is False
    assert rj.parse_cpu_max("800000 100000") == 8
    assert rj.parse_cpu_max("850000 100000") == 9
    assert rj.parse_cpu_max("max 100000") is None


def test_scripts_and_status_flags(tmp_path):
    spec = _spec(tmp_path, max_hours=2.0, grace_minutes=10)
    dm = rj.deadman_script(spec)
    assert "MAX=7200" in dm and "GRACE=600" in dm and "runpodctl remove pod" in dm
    js = rj.job_script(spec)
    assert "touch .kt/DONE" in js and "touch .kt/FAILED" in js and "NPROC" in js
    assert rj.job_status(f"x\n{rj.DONE_FLAG}\n") == "done"
    assert rj.job_status(f"{rj.FAILED_FLAG}\n") == "failed"
    assert rj.job_status("epoch 3 done\n") is None


def test_create_cycles_gpus_on_stock_errors(tmp_path):
    client, logs = FakeClient(stock_fail=3), []
    spec = _spec(tmp_path, gpu_types=["A", "B"], create_rounds=3, round_wait_seconds=0)
    pod_id, gpu = rj.create_pod(spec, client, logs.append, lambda s: None)
    assert (pod_id, gpu) == ("p1", "B")
    assert client.created[0]["min_vcpu_count"] == 1 and client.created[0]["name"] == "kt-job1"
    with pytest.raises(RuntimeError, match="no GPU"):
        rj.create_pod(spec, FakeClient(stock_fail=99), logs.append, lambda s: None)


def test_run_success_terminates_and_writes_ledger(tmp_path):
    client, rem = FakeClient(), FakeRemote(["running", f"{rj.DONE_FLAG}\n"])
    spec = _spec(tmp_path)
    status = rj.run(
        spec,
        client=client,
        remote_factory=lambda ip, port: rem,
        log=lambda *a: None,
        sleep=lambda s: None,
    )
    assert status == "done"
    assert client.terminated == ["p1"]
    assert rem.puts and rem.puts[0][1].endswith("/train.py")
    assert any("deadman.sh" in c for c in rem.cmds)
    assert rem.pulls[-1] == ["out", ".kt"]
    row = json.loads((spec.local_dir / "pods.jsonl").read_text().splitlines()[-1])
    assert row["status"] == "done" and row["name"] == "kt-job1"


def test_run_terminates_on_failure(tmp_path):
    client = FakeClient()
    rem = FakeRemote([], fail_on="echo rc=$?")
    with pytest.raises(RuntimeError):
        rj.run(
            _spec(tmp_path),
            client=client,
            remote_factory=lambda ip, port: rem,
            log=lambda *a: None,
            sleep=lambda s: None,
        )
    assert client.terminated == ["p1"]


def test_run_refuses_low_balance_and_dry_run(tmp_path):
    with pytest.raises(RuntimeError, match="refusing"):
        rj.run(_spec(tmp_path, floor=99.0), client=FakeClient(balance=100.0), log=lambda *a: None)
    client = FakeClient()
    assert rj.run(_spec(tmp_path), client=client, log=lambda *a: None, dry_run=True) == "dry-run"
    assert client.created == []


def test_list_own_pods_filters_by_prefix():
    client = FakeClient(pods=[{"id": "a", "name": "kt-x"}, {"id": "b", "name": "someone-else"}])
    assert [p["id"] for p in rj.list_own_pods("kt-", client)] == ["a"]
    with pytest.raises(ValueError, match="prefix"):
        rj.list_own_pods("", client)


def test_read_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[default]\napi_key = "abc123"\n')
    assert rj.read_api_key(cfg) == "abc123"
    monkeypatch.setenv("RUNPOD_API_KEY", "envkey")
    assert rj.read_api_key(cfg) == "envkey"
    monkeypatch.delenv("RUNPOD_API_KEY")
    with pytest.raises(ValueError, match="API key"):
        rj.read_api_key(tmp_path / "missing.toml")


def test_cli_runpod_dry_run_reports_missing_extra_or_key(tmp_path, monkeypatch):
    job = tmp_path / "job.toml"
    job.write_text('name = "a"\ncommand = "true"\n')
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert main(["runpod", "run", str(job), "--dry-run"]) == 1
