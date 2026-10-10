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
| Competition data | `/kaggle/input/competitions/<slug>/` (NOT `/kaggle/input/<slug>/`) |
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

`-v` is required: without it the CLI answers "Code competition submissions require both
the output file name and the version number". The version number is printed by
`kernels push` ("Kernel version N successfully pushed"). The notebook UI button still works.

Or from Python: `kaggle_tools.kernel.push / wait / output`.

## GPU quota, hung kernels and session limits

- Every `kernels push` of a GPU notebook runs it once in save mode, and that run counts
  against the weekly GPU quota (~30 h). Once it is used up, pushes fail with "Maximum
  weekly GPU quota of 30.00 hours reached" until the weekly reset (Saturday 00:00 UTC).
  CPU kernels (preprocessing, corpus builds, audits) do not touch the GPU quota.
- At most 2 GPU sessions run at once; a third push waits or fails. Retry in a loop.
- **A hung kernel burns quota for hours.** RSNA Knee 2026: a `multiprocessing.Pool`
  initializer raised `NameError`; the pool silently respawned workers forever and the
  save-mode run sat on two T4s until cancelled by hand. Catch everything inside a pool
  initializer and return the error per item, and start a watchdog thread in any test
  run (`time.sleep(limit); faulthandler.dump_traceback(); os._exit(3)`).
- Never `glob("**/...", recursive=True)` over `/kaggle/input` to find your weights:
  the competition mount may sit under `/kaggle/input/<slug>/` and a recursive walk of
  ~10^5 DICOMs took 9 minutes of the run. Skip competition mounts explicitly.

## Rehearse the hidden-test run on train data

Before submitting a new model, run the inference code on ~64 training studies posing
as the test set: build a fake root with `test.csv` and the per-series CSV for those
ids and a symlink `test_series -> train_series`, point the notebook at it via an env
var, and time it to extrapolate the hidden-test runtime.
**Compare against the same ensemble run locally, not against OOF.** Fold models were
trained on 4/5 of those studies, so a 5-fold average differs from OOF by up to ~0.7
per prediction; an automatic gate that compared to OOF blocked a correct submission
(RSNA Knee v3, later 0.940 LB). Against the local 5-fold average the kernel matched to
max |diff| 0.0099 (fp16 vs bf16).

## Big artefacts: corpora and outputs

- One kernel's `/kaggle/working` output is capped (~20 GB). Shard large preprocessing
  (e.g. a 96-slice MRI corpus, ~48 GB) across several CPU kernels by interleaved ids
  (`ids[part::n_parts]`) so each part is a random-ish sample.
- `kaggle kernels output` truncates multi-GB files without an error. Download from the
  signed URL with `curl -C -` in a retry loop and check the size against the
  `Content-Range` total of a 1-byte ranged GET (signed URLs reject HEAD).
- Build the training corpus with the *same function* the inference kernel uses
  (copy it verbatim), so train and test preprocessing cannot drift.

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
