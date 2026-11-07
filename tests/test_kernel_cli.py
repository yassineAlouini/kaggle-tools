import json

import pandas as pd

from kaggle_tools import kernel
from kaggle_tools.cli import main


def _write(d, meta, source):
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
    nb = {"cells": [{"cell_type": "code", "source": source.splitlines(keepends=True)}]}
    (d / meta["code_file"]).write_text(json.dumps(nb))


def test_template_passes_checks(tmp_path):
    meta = kernel.metadata_template("me/my-kernel", "comp")
    src = "df = read('/kaggle/input/competitions/comp/test.csv')\nsub.to_csv('submission.csv')"
    _write(tmp_path, meta, src)
    assert kernel.check_kernel_dir(tmp_path) == []


def test_catches_classic_mistakes(tmp_path):
    meta = kernel.metadata_template("me/my-kernel", "comp")
    meta["enable_internet"] = True
    del meta["machine_shape"]
    src = "!pip install lightgbm\nassert x\ndf = read('/kaggle/input/comp/test.csv')"
    _write(tmp_path, meta, src)
    messages = " | ".join(str(f) for f in kernel.check_kernel_dir(tmp_path))
    for needle in ("internet", "P100", "submission", "competitions/comp", "assert", "pip install"):
        assert needle in messages


def test_cli_init_check_validate(tmp_path, capsys):
    kdir = tmp_path / "k"
    assert main(["init-kernel", str(kdir), "--id", "me/k", "--competition", "comp"]) == 0
    assert main(["check", str(kdir)]) == 1  # code_file missing
    sample = pd.DataFrame({"row_id": [1, 2], "target": 0.5})
    sample.to_csv(tmp_path / "sample.csv", index=False)
    assert main(["validate", str(tmp_path / "sample.csv"), str(tmp_path / "sample.csv")]) == 0
    assert "OK" in capsys.readouterr().out


def test_status_parsing(monkeypatch):
    monkeypatch.setattr(
        kernel, "_kaggle", lambda *a: 'me/k has status "KernelWorkerStatus.COMPLETE"'
    )
    assert kernel.status("me/k") == "COMPLETE"
    monkeypatch.setattr(kernel, "_kaggle", lambda *a: 'me/k has status "RUNNING"')
    assert kernel.status("me/k") == "RUNNING"


def test_source_check_avoids_false_positives():
    src = 'print("pip install will fail offline")\n# pip install foo\nx = 1'
    assert kernel.check_source(src, competitions=None) == []
    flagged = kernel.check_source('subprocess.run(["pip", "install", "x"])', None)
    assert [f.level for f in flagged] == ["warning"]


def test_recursive_input_scan_is_flagged():
    for src in [
        "w = glob.glob('/kaggle/input/**/*.pt', recursive=True)",
        "for d, _, f in os.walk('/kaggle/input'):",
        "w = list(Path('/kaggle/input').rglob('*.pt'))",
    ]:
        assert any("recursive scan" in f.message for f in kernel.check_source(src)), src
    assert not any(
        "recursive scan" in f.message
        for f in kernel.check_source("w = glob.glob('/kaggle/input/my-weights/*.pt')")
    )


def test_parse_pushed_version():
    out = "Kernel version 7 successfully pushed.  Please check progress at https://..."
    assert kernel.parse_pushed_version(out) == 7
    assert kernel.parse_pushed_version("Kernel push error: quota") is None


def test_submit_requires_a_version_and_passes_k_v_f(monkeypatch):
    calls = []
    monkeypatch.setattr(kernel, "_kaggle", lambda *a: calls.append(a) or "ok")
    kernel.submit("comp", "me/k", 3, "v3 test")
    assert calls[0] == ("competitions", "submit", "comp", "-k", "me/k", "-v", "3",
                        "-f", "submission.csv", "-m", "v3 test")  # fmt: skip
    import pytest

    with pytest.raises(ValueError, match="version"):
        kernel.submit("comp", "me/k", 0, "x")


def test_parse_range_probe_requires_206_and_content_range():
    redirect_then_206 = (
        "HTTP/2 302\r\nlocation: https://x\r\n\r\n"
        "HTTP/2 206\r\ncontent-range: bytes 0-0/12345\r\ncontent-length: 1\r\n\r\n"
    )
    assert kernel._parse_range_probe(redirect_then_206) == 12345
    assert kernel._parse_range_probe("HTTP/1.1 200 OK\r\ncontent-length: 12345\r\n\r\n") is None
    assert kernel._parse_range_probe("HTTP/1.1 206 Partial\r\ncontent-length: 1\r\n\r\n") is None
    assert kernel._parse_range_probe("") is None


def test_download_resumable_resumes_until_complete(monkeypatch, tmp_path):
    import hashlib

    import pytest

    payload = b"0123456789" * 10
    dest = tmp_path / "big.bin"
    chunks = iter([payload[:40], payload[40:70], payload[70:]])  # two truncated transfers

    def fake_run(cmd, *a, **k):
        assert cmd[:2] == ["curl", "--fail"] and "-C" in cmd
        with dest.open("ab") as fh:
            fh.write(next(chunks))

    monkeypatch.setattr(kernel.subprocess, "run", fake_run)
    monkeypatch.setattr(kernel.time, "sleep", lambda s: None)
    sha = hashlib.sha256(payload).hexdigest()
    assert kernel.download_resumable("https://x", dest, expected_size=100, sha256=sha) == dest
    assert dest.read_bytes() == payload
    with pytest.raises(RuntimeError, match="sha256"):
        kernel.download_resumable("https://x", dest, expected_size=100, sha256="00" * 32)
    monkeypatch.setattr(kernel, "remote_size", lambda url: None)
    with pytest.raises(RuntimeError, match="cannot verify"):
        kernel.download_resumable("https://x", tmp_path / "y.bin")
