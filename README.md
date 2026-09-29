# kaggle-tools

Small, battle-tested Python utilities **and Claude Code skills** for Kaggle competitions.

The code encodes lessons from real competitions (AIMO3, BirdCLEF 2026, ROGII Wellbore
Geology, NVIDIA Nemotron, Biohub cell tracking, …). Most of them were learned the hard way:
silent "Submission Scoring Error"s, timeouts, P100s that current torch can't run,
offline re-runs without internet, and OOF gains that never reached the leaderboard.

## Install

```bash
uv add git+https://github.com/yassineAlouini/kaggle-tools        # or: pip install git+...
uv add "kaggle-tools[tracking,tune,gbdt] @ git+https://github.com/yassineAlouini/kaggle-tools"
```

Core dependencies are only `numpy`, `pandas` and `scikit-learn`. Optional extras:

| extra | adds | for |
|---|---|---|
| `tracking` | [Trackio](https://github.com/gradio-app/trackio) | local-first, wandb-compatible experiment tracking |
| `tune` | Optuna | hyper-parameter search |
| `gbdt` | LightGBM, XGBoost, CatBoost | tabular stacks |

## What's inside

| module | highlights |
|---|---|
| `env` | `competition_dir` / `dataset_dir` resolve every `/kaggle/input` mount layout; `is_competition_rerun`; `install_offline_wheels` (fails loudly instead of silently degrading) |
| `submission` | `ProgressiveSubmission` keeps a valid file on disk after every stage (atomic writes; a stage's output only replaces the file if the stage finishes, failing stages are logged and skipped); vectorised `build_submission`; `validate_submission` |
| `ensemble` | `rank_blend` (per-column percentile ranks, for AUC-like metrics); `hill_climb` (Caruana greedy weights) |
| `cv` | `spatial_group_folds` (K-means on entity coordinates); `rare_safe_folds` (stratified-group, rare classes pinned to train, multi-label aware) |
| `metrics` | `macro_auc` (skips empty columns, like BirdCLEF), `rmse`, `best_threshold` |
| `budget` | `TimeBudget` for wall-clock-limited notebooks; `@timed` |
| `kernel` | pre-push lint for `kernel-metadata.json` + notebook source; `push` / `status` / `wait` / `output` wrappers |
| `llm` | `extract_integer_answer`, `majority_vote`, inverse-entropy `weighted_vote`, early-stopping `ConsensusVoter` |
| `audio` | BirdCLEF `row_id` parsing, window ids, `frame_windows` |
| `features` | `date_features` (tz-aware, cyclical), `oof_target_encode` |
| `tracking` | `Ledger`: JSONL submission ledger keyed by Kaggle kernel version, with LB deltas; `trackio_run` |

### A robust inference notebook

```python
import pandas as pd
from kaggle_tools.env import competition_dir, working_dir
from kaggle_tools.submission import ProgressiveSubmission
from kaggle_tools.ensemble import rank_blend
from kaggle_tools.budget import TimeBudget

budget = TimeBudget(total_seconds=90 * 60, margin_seconds=10 * 60)
data = competition_dir("birdclef-2026")
sub = ProgressiveSubmission(
    pd.read_csv(data / "sample_submission.csv"), working_dir() / "submission.csv"
)  # prior written now

with sub.stage("backbone"):
    p_backbone = run_backbone(data)
    sub.write(p_backbone)

if sub.ok("backbone") and budget.has(20 * 60):
    with sub.stage("ensemble"):
        sub.write(rank_blend([p_backbone, run_second_arm(data)], [0.5, 0.5]))
```

### CLI

```bash
kt init-kernel notebooks/my-kernel --id me/my-kernel --competition birdclef-2026  # T4, internet off
kt check notebooks/*/            # lint before `kaggle kernels push`
kt validate submission.csv sample_submission.csv
kt ledger submissions.jsonl      # every submission, with delta vs best-so-far
```

`kt check` flags internet enabled on a competition kernel and a missing `code_file` as
**errors** (exit 1). It **warns** (exit 1 only with `--strict`) about a GPU kernel without a
pinned T4, `/kaggle/input/<comp>/` instead of `/kaggle/input/competitions/<comp>/`, no
`submission.csv`/`.parquet`/`.zip` write (inference-server notebooks are exempt), bare `assert`s and online
`pip install`s. Using `kernel_sources` is reported as **info**: fine for wheels and weights,
wrong for predictions that must be recomputed on the hidden test.

## Claude Code skills

`skills/` ships as a Claude Code plugin:

| skill | use it for |
|---|---|
| `kaggle-competition-playbook` | reading the metric code, CV design, CV→LB transfer, experiment selection, ensembling, quota |
| `kaggle-code-competitions` | offline re-runs, paths, wheels, accelerators, time limits, push workflow, pre-submit checklist |
| `kaggle-audio-bioacoustics` | BirdCLEF-style soundscapes: Perch embeddings + probes + temporal model + rank-blend |
| `kaggle-llm-reasoning` | AIMO-style: model choice, vLLM, SC-TIR, voting, time budget |
| `kaggle-llm-finetuning` | LoRA/SFT on Kaggle: TRL/PEFT, packing, verified CoT data, adapter packaging |
| `kaggle-tabular` | GBDT + TabICL stacks, spatial CV, feature ablations, local training → Kaggle inference |

Install:

```text
/plugin marketplace add yassineAlouini/kaggle-tools
/plugin install kaggle-tools@kaggle-tools
```

## Development

```bash
uv sync                     # creates .venv with dev tools
uv run pytest
uv run ruff format . && uv run ruff check .
uv run ty check src
uvx pre-commit install
```

## License

MIT © 2018–2026 Yassine Alouini
