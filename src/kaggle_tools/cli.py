"""``kt`` — command-line helpers.

kt check <kernel_dir>...        lint kernel-metadata.json + notebook before pushing
kt init-kernel <dir> --id user/slug --competition slug [--cpu]
kt validate <submission.csv> <sample_submission.csv>
kt ledger [submissions.jsonl]   print the submission ledger
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from kaggle_tools import kernel
from kaggle_tools.submission import validate_submission
from kaggle_tools.tracking import Ledger


def _check(args: argparse.Namespace) -> int:
    exit_code = 0
    for d in args.kernel_dirs:
        findings = kernel.check_kernel_dir(d)
        print(f"== {d}: {len(findings)} finding(s)")
        for f in findings:
            print(f"  {f}")
        failing = {"error", "warning"} if args.strict else {"error"}
        if any(f.level in failing for f in findings):
            exit_code = 1
    return exit_code


def _init_kernel(args: argparse.Namespace) -> int:
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / kernel.METADATA_FILE
    if path.exists() and not args.force:
        print(f"{path} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    meta = kernel.metadata_template(args.id, args.competition, args.code_file, gpu=not args.cpu)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    problems = validate_submission(pd.read_csv(args.submission), pd.read_csv(args.sample))
    for p in problems:
        print(f"ERROR {p}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _ledger(args: argparse.Namespace) -> int:
    table = Ledger(args.path, higher_is_better=not args.lower_is_better).table()
    print("(empty ledger)" if table.empty else table.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kt", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("check", help="lint kernel dirs before `kaggle kernels push`")
    p.add_argument("kernel_dirs", nargs="+")
    p.add_argument("--strict", action="store_true", help="fail on warnings too")
    p.set_defaults(func=_check)

    p = sub.add_parser("init-kernel", help="write a safe kernel-metadata.json")
    p.add_argument("dir")
    p.add_argument("--id", required=True, help="username/kernel-slug")
    p.add_argument("--competition", required=True)
    p.add_argument("--code-file", default="notebook.ipynb")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_init_kernel)

    p = sub.add_parser("validate", help="check a submission against sample_submission")
    p.add_argument("submission")
    p.add_argument("sample")
    p.set_defaults(func=_validate)

    p = sub.add_parser("ledger", help="print the submission ledger")
    p.add_argument("path", nargs="?", default="submissions.jsonl")
    p.add_argument("--lower-is-better", action="store_true")
    p.set_defaults(func=_ledger)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
