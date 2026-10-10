# 0.1.2 (2026-10-10)

Helpers for the code-competition lessons added to the skills in 0.1.1 (RSNA Knee 2026).

## New

* `kaggle_tools.runtime`: `watchdog` (native `faulthandler` timeout, always cancelled on
  exit), `guarded_initializer` / `raise_if_init_failed` (a failing `Pool` initializer is
  re-raised in the parent instead of respawning workers forever; picklable under
  `spawn`), `shard_ids` (disjoint, complete shards from one seeded manifest),
  `rehearsal_root` (training rows posing as the test set, opt-in only).
* `kaggle_tools.kernel`: `parse_pushed_version`, `submit` (code competitions, `-k`/`-v`
  required), `remote_size` (1-byte ranged GET, requires HTTP 206 + `Content-Range`),
  `download_resumable` (bounded `curl --fail --location -C -` resumes, refuses to run
  without a size or checksum, sha256 verification).
* `kt check` warns on recursive scans of `/kaggle/input` (glob `recursive=True`,
  `os.walk`, `Path.rglob`).

## Changed

* The `/kaggle/input/<slug>/` path warning now says it is one of the valid competition
  layouts and points to `env.competition_dir`, matching the corrected skill text.

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
