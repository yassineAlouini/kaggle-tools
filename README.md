# kaggle-tools

[![CI](https://github.com/yassineAlouini/kaggle-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/yassineAlouini/kaggle-tools/actions/workflows/ci.yml)
[![Plugin scan](https://github.com/yassineAlouini/kaggle-tools/actions/workflows/plugin-scan.yml/badge.svg)](https://github.com/yassineAlouini/kaggle-tools/actions/workflows/plugin-scan.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Agent skills and a small Python library for Kaggle competitions, built from what went
wrong in real ones.

Each piece exists because something failed: a silent "Submission Scoring Error", a
notebook that timed out on the hidden test, a P100 that current PyTorch can no longer
run, a re-run without internet, an out-of-fold gain that never reached the leaderboard.
The lessons come from AIMO3, BirdCLEF 2026, RSNA Knee 2026, ROGII Wellbore Geology,
NVIDIA Nemotron, Biohub cell tracking and ARC-AGI-2.

The repository has two independent parts:

- **[Agent skills](#agent-skills)**: a Claude Code plugin of 8 skills that guide an
  agent through CV design, code-competition submissions, LLM inference and fine-tuning
  on Kaggle GPUs, and domain-specific pipelines. Recommendations are tied to the
  competition results that motivated them.
- **[Python package](#python-package)**: `kaggle_tools`, small utilities for
  notebooks that must survive Kaggle's offline, time-limited re-runs, plus the `kt`
  CLI that lints a kernel before you push it.

## Agent skills

| skill | use it for |
|---|---|
| `kaggle-competition-playbook` | reading the metric code, CV design, CV→LB transfer, experiment selection, ensembling, submission quota |
| `kaggle-code-competitions` | offline re-runs, data paths, wheels, accelerators, time limits, push workflow, pre-submit checklist |
| `kaggle-audio-bioacoustics` | BirdCLEF-style soundscapes: Perch embeddings + probes + temporal model + rank blend |
| `kaggle-llm-reasoning` | AIMO-style math: model choice, vLLM, tool-integrated reasoning, voting, time budget |
| `kaggle-llm-serving` | vLLM throughput on 4x L4: TP vs one replica per GPU, fitting a 27B on 24 GB, KV/state capacity, MTP depth, quality gate |
| `kaggle-llm-finetuning` | LoRA/SFT on Kaggle: TRL/PEFT, packing, verified chain-of-thought data, adapter packaging |
| `kaggle-medical-imaging` | MRI/CT classification: weak report labels, anatomical inputs, teacher lineage, site-aware validation |
| `kaggle-tabular` | GBDT + TabICL stacks, spatial CV, feature ablations, local training → Kaggle inference |

### Install

In Claude Code (inside a session, or the equivalent `claude plugin ...` shell commands):

```text
/plugin marketplace add yassineAlouini/kaggle-tools
/plugin install kaggle-tools@kaggle-tools
/plugin marketplace update kaggle-tools     # later, to pull new skills
```

Skills load on demand when a task matches their description. The plugin contains only
`SKILL.md` files and Markdown references: no hooks, MCP servers or commands, and nothing
runs on install (see [SECURITY.md](SECURITY.md)).

**Supported clients.** Claude Code through the plugin above (tested with
`claude plugin install`). Each `skills/<name>/` folder is also a standalone
[Agent Skill](https://agentskills.io) that any client reading `SKILL.md` folders can use:

- **Claude Code without the plugin**: copy a skill folder into `~/.claude/skills/`.
- **Codex**: copy or link a skill folder into `~/.codex/skills/`. The optional
  `agents/openai.yaml` adds Codex UI metadata.

Keep each folder's `references/` directory when copying.

### Case studies

The [RSNA Knee 2026 case study](skills/kaggle-medical-imaging/references/rsna-knee-2026.md)
and its [first-place extension](skills/kaggle-medical-imaging/references/rsna-knee-2026-winner.md)
extract evidence from the 1st-, 5th- and 22nd-place writeups, with ablation limits and
source links. The medical-imaging skill applies the transferable decisions; the playbook
and code-competition skills carry the validation, blending and shared-inference lessons.

## Python package

```bash
uv add git+https://github.com/yassineAlouini/kaggle-tools        # or: pip install git+...
uv add "kaggle-tools[tracking,tune,gbdt] @ git+https://github.com/yassineAlouini/kaggle-tools"
```

Python 3.11+. Core dependencies are only `numpy`, `pandas` and `scikit-learn`; heavier
integrations are optional extras, imported only by the functions that need them:

| extra | adds | for |
|---|---|---|
| `tracking` | [Trackio](https://github.com/gradio-app/trackio) | local-first, wandb-compatible experiment tracking |
| `tune` | Optuna | hyper-parameter search |
| `gbdt` | LightGBM, XGBoost, CatBoost | tabular stacks |

### Modules

| module | highlights |
|---|---|
| `env` | `competition_dir` / `dataset_dir` resolve every `/kaggle/input` mount layout; `is_competition_rerun`; `install_offline_wheels` (fails loudly instead of silently degrading) |
| `submission` | `ProgressiveSubmission` keeps a valid file on disk after every stage (atomic writes; a stage's output only replaces the file if the stage finishes, failing stages are logged and skipped); vectorised `build_submission`; `validate_submission` |
| `ensemble` | `rank_blend` (per-column percentile ranks, for AUC-like metrics); `hill_climb` (Caruana greedy weights) |
| `cv` | `spatial_group_folds` (K-means on entity coordinates); `rare_safe_folds` (stratified-group, rare classes pinned to train, multi-label aware) |
| `metrics` | `macro_auc` (skips empty columns, like BirdCLEF), `rmse`, `best_threshold` |
| `budget` | `TimeBudget` for wall-clock-limited notebooks; `@timed` |
| `kernel` | pre-push lint for `kernel-metadata.json` + notebook source (incl. recursive `/kaggle/input` scans); `push` / `status` / `wait` / `output` wrappers; `parse_pushed_version` + `submit` (code competitions: `-k`/`-v`); `download_resumable` for multi-GB outputs (HTTP 206 size probe, bounded resume, sha256) |
| `runtime` | in-kernel guards: `watchdog` (native faulthandler timeout, cancelled on exit); `guarded_initializer` + `raise_if_init_failed` (Pool init errors reach the parent instead of hanging); `shard_ids` (one seeded manifest); `rehearsal_root` (train rows posing as test) |
| `llm` | `extract_integer_answer`, `majority_vote`, inverse-entropy `weighted_vote`, early-stopping `ConsensusVoter` |
| `audio` | BirdCLEF `row_id` parsing, window ids, `frame_windows` |
| `features` | `date_features` (tz-aware, cyclical), `oof_target_encode` |
| `tracking` | `Ledger`: JSONL submission ledger keyed by Kaggle kernel version, with LB deltas; `trackio_run` |

### Example: an inference notebook that always submits

A valid submission is on disk from the first line, and each stage only replaces it if
the stage finishes. If the second model crashes, or the time budget says it would not
finish, the notebook still submits the first. (A notebook that hits Kaggle's hard time
limit is not scored at all, which is why the budget check skips stages up front.)

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

`kt check` reports three levels:

- **error** (exit 1): internet enabled on a competition kernel, a missing `code_file`,
  malformed `id` or source slugs.
- **warning** (exit 1 only with `--strict`): a GPU kernel without a pinned T4 (the default
  P100 is unsupported by current PyTorch), a hard-coded `/kaggle/input/<comp>/` path
  instead of `env.competition_dir`, no `submission.csv`/`.parquet`/`.zip` write
  (inference-server notebooks are exempt), recursive scans of `/kaggle/input`, bare
  `assert`s and online `pip install`s.
- **info**: `kernel_sources` in use. Fine for wheels and weights, wrong for predictions
  that must be recomputed on the hidden test.

## Development

```bash
uv sync                     # creates .venv with dev tools
uv run pytest
uv run ruff format . && uv run ruff check .
uv run ty check src
claude plugin validate .    # after editing .claude-plugin/ or skills/
uvx pre-commit install
```

New utilities and skills follow the same rule as the rest of the repo: tie each one to
the competition failure it prevents, and back every "works" or "doesn't work" claim with
evidence (competition, version, leaderboard delta). See [CLAUDE.md](CLAUDE.md) for the
conventions.

## License

MIT © 2018–2026 Yassine Alouini
