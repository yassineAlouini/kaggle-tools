# 1.1.0 (unreleased)

- New skill `kaggle-llm-evolution`: AlphaEvolve-style loops (LLM-written search programs scored by an exact
  verifier), with evidence from Google Quantum AI VQCC: the loop rebuilt the known best GF(2^n) multiplier ranks
  (24 / 42 / 99) where a hand-written flip-graph search stalled, and failed where the target sat on a counting
  bound. Covers program contracts, islands and lineage, calibration gates, counting bounds, shared LLM quota and
  live W&B tracking.

# 1.0.0 (2026-11-15)

The first stable release of the modern kaggle-tools: **nine agent skills**, a
**Python competition toolkit**, and the **`kt` CLI**. This release brings together
the 2026 rewrite and the skills, runtime guards, reports and operational lessons
developed since the legacy 0.0.x tags.

The guidance draws on AIMO3, BirdCLEF 2026, RSNA Knee 2026, ROGII Wellbore Geology,
NVIDIA Nemotron, Biohub cell tracking and ARC-AGI-2. Skills link recommendations to
competition evidence and distinguish measured results from projections.

## Nine agent skills

| Skill | What it covers |
|---|---|
| `kaggle-competition-playbook` | Metric implementation, CV design, CV-to-leaderboard transfer, experiment selection, ensembling and submission quotas. Includes a Dagu guide for long and scheduled jobs, named queues, logs, systemd timer migration and authenticated remote access through Tailscale. |
| `kaggle-code-competitions` | Offline hidden-test reruns, input layouts, wheel packaging, accelerator selection, runtime budgets, push/submit workflow and pre-submit checks. |
| `kaggle-leaderboard-analysis` | Public/private rank shifts, team matching, account history and yearly competition progress reports. Includes CSV schemas and API limitations. |
| `kaggle-audio-bioacoustics` | BirdCLEF-style soundscapes, Perch embeddings, probes, temporal models and rank blending. |
| `kaggle-llm-reasoning` | AIMO-style mathematical reasoning, model choice, vLLM, tool-integrated reasoning, voting and inference budgets. |
| `kaggle-llm-serving` | Throughput on Kaggle GPUs: tensor parallelism versus replicas, fitting a 27B model on a 24 GB L4, KV/state capacity, speculative decoding and quality gates. Documents the patches required by experimental configurations. |
| `kaggle-llm-finetuning` | LoRA/SFT with TRL and PEFT, sequence packing, verified chain-of-thought data and adapter packaging. |
| `kaggle-medical-imaging` | MRI/CT classification, weak report labels, anatomical inputs, teacher lineage, site-aware validation and shared inference. Includes evidence-linked RSNA Knee 1st-, 5th- and 22nd-place case studies. |
| `kaggle-tabular` | GBDT and TabICL stacks, spatial validation, feature ablations and local training followed by Kaggle inference. |

Install all nine through the Claude Code marketplace, or copy individual skill
folders into clients that support `SKILL.md`, including Codex. Keep each folder's
`references/` directory. The plugin ships instructions and references and runs
nothing on installation.

## Python competition toolkit

* **Reliable submissions:** `ProgressiveSubmission` writes a valid baseline
  immediately, replaces it atomically after successful stages and preserves the
  previous output if a later stage fails. Submission builders and validators
  check the final file against the competition schema.
* **Offline environments:** resolve competition/dataset input mounts, detect
  competition reruns and install bundled wheels with explicit failures.
* **Runtime guards:** wall-clock budgets, native `faulthandler` watchdogs, pool
  initializer failures propagated to the parent, deterministic disjoint shards
  and opt-in training-data rehearsals.
* **Kernel workflow:** lint metadata and source, push and monitor kernels, submit
  an explicit saved kernel version, and download large outputs with bounded
  resumes and size/checksum verification.
* **Modeling utilities:** spatial and rare-class-aware folds, percentile-rank
  blending, greedy ensemble weights, macro AUC, RMSE and threshold selection;
  date features and out-of-fold target encoding.
* **Domain helpers:** integer-answer extraction and consensus/weighted voting
  for LLMs; soundscape row IDs and frame windows for audio.
* **Experiment records:** a JSONL submission ledger keyed by kernel version,
  leaderboard deltas and optional Trackio integration.

## Leaderboard and progress reports

* `kt leaderboard-plot` compares public/private standings, preserving board ranks,
  fetching complete pagination and matching teams by ID. Unverifiable public-board
  fallbacks are rejected.
* `kt competition-progress` reports **top percentage = 100 × rank / teams**
  (lower is better), with individual competition bars grouped by year and means
  calculated from ended results. Unranked entries can be included explicitly.
* Generate PNG/SVG, readable multipage PDF and standalone interactive HTML, plus
  CSV/JSON data and provenance. Use live Kaggle data or saved CSVs; reports stay local.
* `kt check`, `kt init-kernel`, `kt validate` and `kt ledger` cover the rest of the
  notebook workflow. Pre-push checks include internet access, source paths,
  accelerator configuration, output writes and recursive `/kaggle/input` scans.

## Installation

Python package pinned to this release:

```bash
uv add "kaggle-tools @ git+https://github.com/yassineAlouini/kaggle-tools@v1.0.0"
# Optional reporting and live Kaggle integrations:
uv add "kaggle-tools[plots,kaggle] @ git+https://github.com/yassineAlouini/kaggle-tools@v1.0.0"
```

Claude Code plugin:

```text
/plugin marketplace add yassineAlouini/kaggle-tools
/plugin install kaggle-tools@kaggle-tools
# Existing installations:
/plugin marketplace update kaggle-tools
```

For Codex, copy or link the desired `skills/<name>/` folder into
`~/.codex/skills/`, then invoke `$<skill-name>`.

## Compatibility and maintenance

* **Python 3.11+**; CI covers Python 3.11, 3.12 and 3.13. Core dependencies are
  `numpy>=1.26`, `pandas>=3.1.1` and `scikit-learn>=1.5`.
* Optional extras: `tracking` (Trackio), `tune` (Optuna), `gbdt` (LightGBM,
  XGBoost, CatBoost), `plots` (Matplotlib, Plotly) and `kaggle` (API/SDK).
* **Upgrading from legacy 0.0.x:** the rewrite removed Keras 2 callbacks/model I/O,
  old Keras/OpenCV image helpers, D3 tree visualization, the Dask scaler and
  `sklearn.cross_validation` utilities. The old Sphinx documentation and committed
  build output were also removed. Migrate imports to the current `kaggle_tools`
  modules; see the README for examples.
* Uses a `src/` layout, `uv`/`uv_build`, a lockfile, Ruff, ty, pytest, pre-commit
  and GitHub Actions. Actions are pinned to commit SHAs; a separate plugin scan
  checks plugin hygiene. `SECURITY.md` documents plugin behavior and reporting.
* Package metadata and both Claude Code plugin manifests now share version 1.0.0.

[Full comparison from the last legacy tag](https://github.com/yassineAlouini/kaggle-tools/compare/0.0.3...v1.0.0)

# 0.2.1 (2026-11-07)

* `kaggle-competition-playbook`: run every long or scheduled job through one local job
  runner with a web UI (Dagu). New `references/dagu-job-ui.md` covers the
  localhost-plus-login setup, the systemd service, the per-competition job-file
  template, agent commands, and migrating systemd timers. `systemd-run` stays as the
  fallback.

# 0.2.0 (2026-10-23)

* Add optional `plots` and `kaggle` extras with generic leaderboard and history commands.
* Preserve complete pagination and board ranks; detect unverifiable public/private fallbacks.
* Plot top percentage (`100 * rank / teams`, lower is better) per competition in yearly
  subplots, with ended-result means and readable multipage PDF/HTML output.
* Add the `kaggle-leaderboard-analysis` skill for Claude Code and Codex.

# 0.1.4 (2026-11-07)

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

# 0.1.3 (2026-10-25)

* Add a portable medical-imaging skill and evidence-linked RSNA Knee 5th/22nd-place case study.
* Extract label calibration, anatomical input, teacher-lineage, site validation and marginal-value lessons.
* Compare probability/rank blending instead of prescribing rank blending for every AUC ensemble.
* Document shared preprocessing and feature reuse with prediction-parity and runtime checks.
* Extend the review to both first-place teammates: spatial/privileged supervision,
  annotation grounding, validation selection, input contracts and cold-I/O budgets.

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
