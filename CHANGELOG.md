# 0.1.3 (2026-10-24)

* Add a portable medical-imaging skill and evidence-linked RSNA Knee 5th/22nd-place case study.
* Extract label calibration, anatomical input, teacher-lineage, site validation and marginal-value lessons.
* Compare probability/rank blending instead of prescribing rank blending for every AUC ensemble.
* Document shared preprocessing and feature reuse with prediction-parity and runtime checks.

# 0.1.0 (2026-09-27)

Complete rewrite for a modern Python stack, based on 2025–2026 competition experience.

## Breaking changes

* Removed the 2018-era modules (Keras 2 callbacks and model I/O, Keras/OpenCV image
  helpers, d3 tree visualisation, Dask scaler, `sklearn.cross_validation` split). They
  no longer ran on current library versions.
* Removed the Sphinx docs and the committed build output.
* Python >= 3.11. Core dependencies reduced to numpy, pandas and scikit-learn.

## Tooling

* `uv` + `uv_build` backend, `src/` layout, dependency groups, lockfile.
* `ruff` (lint + format), `ty` type checking, pytest, GitHub Actions, pre-commit.
* Optional extras: `tracking` (Trackio), `tune` (Optuna), `gbdt`.

## New modules

`env`, `submission`, `ensemble`, `cv`, `metrics`, `budget`, `kernel`, `llm`, `audio`,
`features`, `tracking`, and the `kt` CLI (`check`, `init-kernel`, `validate`, `ledger`).

## Claude Code plugin

Six skills under `skills/`, installable via `/plugin marketplace add yassineAlouini/kaggle-tools`.

# 0.0.3 (2018-08-26)

* Add the `FBetaMetricCallback` Keras callback.

Previous versions (prior to 0.0.3) didn't have CHANGELOG entries.
