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
