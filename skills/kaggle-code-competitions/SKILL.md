---
name: kaggle-code-competitions
description: Use when working on any Kaggle code competition — notebook submission, data paths, offline packages, GPU/accelerator choice, kernel push workflow, time limits, or debugging "Submission Scoring Error", TIMEOUT and NO_SCORE
---

# Kaggle Code Competition Infrastructure

Code competitions re-run your saved notebook in a fresh, **offline** container on the
hidden test set. The #1 failure mode: a notebook that works interactively but fails
silently in the re-run. Run `kt check <kernel_dir>` (from `kaggle-tools`) before every push.

## Paths (resolve, don't hard-code)

| What | Path |
|---|---|
| Competition data | `/kaggle/input/competitions/<slug>/` or `/kaggle/input/<slug>/` |
| Datasets (older layout) | `/kaggle/input/datasets/<owner>/<slug>/` |
| Datasets (newer layout) | `/kaggle/input/<slug>/` |
| Models | `/kaggle/input/<model>/<framework>/<variation>/<version>/` |
| Writable | `/kaggle/working/` only (plus `/kaggle/tmp`) |

The layout differs between script and notebook kernels and has changed over time.
Use `kaggle_tools.env.competition_dir` / `dataset_dir`, which try all layouts and fail
listing every path tried. Don't "fix" a vendored notebook's working paths to a newer
layout — that broke ROGII.

## The submission file

- Must be written to `/kaggle/working/submission.csv` (or the parquet the comp asks
  for). Without it the "Submit to Competition" button stays disabled.
- **Write progressively**: prior → after stage 1 → … → final
  (`kaggle_tools.submission.ProgressiveSubmission`). Any later crash or timeout still
  leaves a scoreable file. This rescued BirdCLEF after five consecutive NO_SCORE runs
  caused by a silently changed scoring environment.
- Build it vectorised. A per-row `df.iloc[i, 1:] = …` loop was O(n²) and consumed
  10–60 min in the re-run.
- Validate against `sample_submission.csv` (`kt validate`).

## Re-run behaviour

- `KAGGLE_IS_COMPETITION_RERUN=1` is set; no internet; fresh environment.
- Any uncaught exception → opaque "Submission Scoring Error" (no traceback).
  Avoid bare `assert`; wrap optional stages in try/except with a fallback.
- **But beware silent fallbacks.** A missing wheel → ImportError caught → uniform
  prior written → LB 0.500. Log loudly, and grep the save-mode log for `ImportError`,
  `FileNotFoundError` and "fallback" before submitting.
- Hidden test is usually larger than the public sample: budget time for the real size.

## Offline packages

Attach wheels as a dataset (or `kernel_sources` of a wheel-building notebook) and
install with `pip install --no-index --no-deps /kaggle/input/.../*.whl`
(`kaggle_tools.env.install_offline_wheels`, which raises when a wheel is missing).
Match versions to the Kaggle image (e.g. an OpenVINO IR built with 2026.0 failed on
Kaggle's 2025.4.1).

## Accelerators

- Default GPU `machine_shape` is a **P100 (sm_60)**. Kaggle's stock torch ≥ 2.10 ships
  no sm_60 kernels → `cudaErrorNoKernelImageForDevice` on *any* CUDA op.
- Pin a T4 in metadata: `"machine_shape": "NvidiaTeslaT4"` (the CLI `--accelerator`
  flag did not stick; the metadata field does).
- Don't unpickle models fitted with `device='cuda'`; ship arrays/weights and re-fit or
  load on the live device. CPU fallback for big in-context models can OOM-kill (SIGKILL is
  not catchable) → skip the stage gracefully instead.
- GPU notebooks have ~32 GB host RAM.

## Time limits

- Budget against the hard wall-clock limit with a margin (AIMO3: 5 h limit, ~600 s
  setup, `notebook_limit≈17000` s; 28800 got the kernel killed).
- Scoring workers vary: identical code can score once and TIMEOUT next time. A
  near-limit pipeline will randomly fail. Keep ≥10–15% headroom.
- Time gates must be measured, not guessed: a gate at "1200 s left" that always fired
  silently dropped the best stage and cost −0.015 (`kaggle_tools.budget.TimeBudget`).
- Speed levers: ONNX/OpenVINO instead of TF SavedModel (freed ~10 min on BirdCLEF),
  batched GPU preprocessing, fewer TTA passes.

## Push workflow

```bash
kt check notebooks/my-kernel/                     # lint metadata + source
kaggle datasets version -p artifacts/ -m "..."    # FIRST: kernels pin dataset versions at push
kaggle kernels push -p notebooks/my-kernel/
kaggle kernels status <user>/<slug>                # poll until COMPLETE / ERROR
kaggle kernels output <user>/<slug> -p /tmp/out/   # always read the log on ERROR
# Submit the COMPLETE version from the CLI (works for code competitions, CLI 2.0, 2026-10):
kaggle competitions submit <comp-slug> -k <user>/<slug> -v <version> -f submission.csv -m "..."
```

When using `-k`, `-v` is required too. `-f` names the saved notebook's output file,
not a local CSV to upload. Record the version printed by `kernels push` and check
that version's completion and logs before submitting it; avoid overlapping pushes
when polling the latest kernel status. The notebook UI button still works.
See the [Kaggle CLI implementation](https://github.com/Kaggle/kaggle-api/blob/main/src/kaggle/api/kaggle_api_extended.py)
(`competition_submit_cli`, `competition_submit_code`).

Or from Python: `kaggle_tools.kernel.push / wait / output`, then
`parse_pushed_version` + `submit(competition, kernel_id, version, message)`.

## GPU quota, hung kernels and session limits

- A normal `kernels push` starts a save-mode run; GPU runs consume GPU quota.
  RSNA Knee 2026 runs encountered a 30 h weekly quota and a two-session GPU limit.
  Check the account's current quota, reset countdown and concurrency limits in
  Kaggle before scheduling work; these observations are not platform guarantees.
  CPU-only preprocessing does not use GPU quota.
- On a concurrency rejection, wait for an existing session to finish before a
  bounded retry with backoff. On quota exhaustion, stop until the displayed reset.
  If a push's response is lost, check whether it created a version before retrying.
- **A hung kernel burns quota for hours.** RSNA Knee 2026: a `multiprocessing.Pool`
  initializer raised `NameError`; the pool silently respawned workers forever and the
  save-mode run sat on two T4s until cancelled by hand. Validate setup before starting
  the pool. If initialization must happen in each worker, catch `Exception`, store
  the failure in worker state, and have the task function raise it to the parent;
  a `Pool` initializer's return value is discarded. Terminate and join the pool on
  failure, and bound waits in the parent. See [Python's Pool documentation](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool).
  For rehearsal runs, arm `faulthandler.dump_traceback_later(limit, exit=True)` and
  cancel it in `finally` with `faulthandler.cancel_dump_traceback_later()` so a
  successful run cannot be killed later. This uses a native watchdog thread; an
  ordinary Python thread may be blocked by code holding the GIL. Forced exit skips
  cleanup, so use an external process-group timeout when child processes must also
  be killed. See [faulthandler](https://docs.python.org/3/library/faulthandler.html#faulthandler.dump_traceback_later).
  Helpers: `kaggle_tools.runtime.watchdog(seconds)` and
  `guarded_initializer(setup)` + `raise_if_init_failed()` in the task function.
- Never `glob("**/...", recursive=True)` over `/kaggle/input` to find your weights:
  the competition mount may sit under `/kaggle/input/<slug>/` and a recursive walk of
  ~10^5 DICOMs took 9 minutes of the run. Skip competition mounts explicitly.

## Rehearse the hidden-test run on train data

Before submitting a new model, rehearse inference on a representative subset of
training inputs using the competition's test schema. For RSNA Knee, use e.g. ~64
studies: build a fake root with `test.csv`, the per-series CSV for those ids, and a
symlink `test_series -> train_series`. Select this root only through an explicit
rehearsal option; leave it disabled in the submitted notebook. Include large studies
and time setup separately from per-study work when estimating hidden-test runtime
(`kaggle_tools.runtime.rehearsal_root` builds the fake root).
**Compare against the same ensemble run locally, not against OOF.** Fold models were
trained on 4/5 of those studies, so a 5-fold average differs from OOF by up to ~0.7
per prediction; an automatic gate that compared to OOF blocked a correct submission
(RSNA Knee v3, later 0.940 LB). Against the local 5-fold average the kernel matched to
max |diff| 0.0099 (fp16 vs bf16).
Align sample ids and output columns and use a precision tolerance measured for your
pipeline; 0.0099 is an observation, not a universal threshold. This is a plumbing
and runtime check, not an unbiased accuracy estimate; keep OOF for model validation.

## Big artefacts: corpora and outputs

- Check the current saved-output limit separately from free scratch disk space
  ([Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)). The RSNA
  Knee corpus build budgeted ~20 GB per kernel for a ~48 GB, 96-slice MRI corpus.
  Shard using one shared id manifest (`ids[part::n_parts]`); shuffle it once with a
  fixed seed if representative shards matter. Interleaving sorted ids is not random,
  and variable study sizes require checking each shard's estimated bytes.
- Multi-GB output downloads were observed to truncate in the RSNA Knee workflow;
  verify size and preferably a producer-recorded checksum before consuming them.
  If a signed-URL download needs resuming, use `curl --fail --location -C -` with
  an explicit output path and bounded retries. Resume only the same immutable file;
  refresh expired URLs for that version. If HEAD is rejected, a `Range: bytes=0-0`
  GET can reveal the total, but require HTTP 206 and a valid `Content-Range`; a server
  may ignore Range. Stop on unsupported resume or a checksum mismatch.
  See [curl's resume options](https://curl.se/docs/manpage.html#-C).
  Helper: `kaggle_tools.kernel.download_resumable(url, dest, expected_size=None, sha256=None)`;
  shards: `kaggle_tools.runtime.shard_ids(ids, part, n_parts, seed)`.
- Share the preprocessing function and configuration between training and inference.
  If offline packaging requires a copy, verify it stays in sync and compare tensors
  for the same raw input; a one-time verbatim copy does not prevent later drift.

## kernel_sources pitfalls

- A parent kernel's outputs are **frozen at its last run**; they are not recomputed
  on the hidden test. Never orchestrate a blend by consuming another notebook's
  predictions; vendor both pipelines inline (append cells to a copy of the notebook).
- `kernel_sources` are fine for static assets (wheels, tokenizer files, weights).
- Kernel vs dataset matters: a kernel listed under `dataset_sources` won't mount.

## Checklist before "Submit to Competition"

1. `enable_internet: false`, `competition_sources` set, T4 pinned if torch GPU.
2. All non-default packages installed from attached wheels; versions printed in log.
3. Save-mode run COMPLETE; log has no ImportError / FileNotFoundError / fallback.
4. `submission.csv` written early and re-written after each stage.
5. Dataset versions pushed *before* the kernel.
6. Runtime in save mode leaves headroom for the full hidden test.
7. Record the kernel version in the ledger before submitting.
8. Rehearsed on train data posing as test (runtime extrapolated, predictions matched
   against the local ensemble, not OOF); watchdog in place for test runs.
