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
# Submit from the notebook UI → "Submit to Competition"
# (`kaggle competitions submit` returns 400 for code competitions)
```

Or from Python: `kaggle_tools.kernel.push / wait / output`.

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
