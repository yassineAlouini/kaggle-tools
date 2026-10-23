# kaggle-tools

Small, battle-tested Python utilities **and reusable agent skills** for Kaggle competitions.

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
| `plots` | Matplotlib, Plotly | local PNG/SVG/HTML and paginated PDF reports |
| `kaggle` | Kaggle API and SDK | read-only live leaderboard/account history fetching |
| `tracking` | [Trackio](https://github.com/gradio-app/trackio) | local-first, wandb-compatible experiment tracking |
| `tune` | Optuna | hyper-parameter search |
| `gbdt` | LightGBM, XGBoost, CatBoost | tabular stacks |

## What's inside

| module | highlights |
|---|---|
| `leaderboard` | complete public/private fetching, ID matching, account history, top percentages |
| `leaderboard_plots` | public/private scatter + zoom; per-year competition bars, readable multipage PDF and standalone HTML |
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

## Leaderboard reports

Install the optional dependencies with `uv sync --extra plots --extra kaggle`, then:

```bash
uv run kt leaderboard-plot --competition COMPETITION_SLUG --output reports/leaderboard
uv run kt competition-progress --include-unranked --output reports/history
# Saved CSVs need only the plots extra:
uv run kt leaderboard-plot --public-csv public.csv --private-csv private.csv --output reports/leaderboard
uv run kt competition-progress --input history.csv --output reports/history
```

History uses **top percentage = 100 * rank / teams (lower is better)**. Each
competition keeps its own bar in a yearly subplot, with the mean top percentage
of ended results in the heading. The PDF packs readable subplots across pages;
HTML embeds Plotly for offline hover/zoom. CSV/JSON preserve data and provenance.
The history PNG/SVG shows the yearly overview with competition counts and mean
top percentages; PDF/HTML also include each competition’s bar.
Use `--include-unranked` to display entries with no rank as empty, labeled rows.
Live history always uses the authenticated account, not a supplied display name.

The leaderboard command preserves board positions and matches by team ID; it
refuses an unverifiable public-board fallback. `--highlight-team` takes a team ID.
See the [skill input reference](skills/kaggle-leaderboard-analysis/references/inputs.md)
for CSV schemas, snapshot dates, and API limitations. Generated reports stay local.

## Claude Code and Codex skills

`skills/` ships as a Claude Code plugin:

| skill | use it for |
|---|---|
| `kaggle-competition-playbook` | reading the metric code, CV design, CV→LB transfer, experiment selection, ensembling, quota |
| `kaggle-code-competitions` | offline re-runs, paths, wheels, accelerators, time limits, push workflow, pre-submit checklist |
| `kaggle-audio-bioacoustics` | BirdCLEF-style soundscapes: Perch embeddings + probes + temporal model + rank-blend |
| `kaggle-llm-reasoning` | AIMO-style: model choice, vLLM, SC-TIR, voting, time budget |
| `kaggle-llm-finetuning` | LoRA/SFT on Kaggle: TRL/PEFT, packing, verified CoT data, adapter packaging |
| `kaggle-leaderboard-analysis` | public/private rank shifts and yearly competition progress reports |
| `kaggle-tabular` | GBDT + TabICL stacks, spatial CV, feature ablations, local training → Kaggle inference |

Install:

```text
/plugin marketplace add yassineAlouini/kaggle-tools
/plugin install kaggle-tools@kaggle-tools
```

For Codex, link the skill from a checkout (or copy the complete skill directory):

```bash
mkdir -p ~/.codex/skills
ln -s "$PWD/skills/kaggle-leaderboard-analysis" ~/.codex/skills/kaggle-leaderboard-analysis
```

Invoke `$kaggle-leaderboard-analysis`. The skill uses the same Python helpers and
CLI in either agent; optional Codex metadata lives in `agents/openai.yaml`.

## Development

```bash
uv sync --extra plots --extra kaggle  # includes report test dependencies
uv run pytest
uv run ruff format . && uv run ruff check .
uv run ty check src
uvx pre-commit install
```

## License

MIT © 2018–2026 Yassine Alouini
