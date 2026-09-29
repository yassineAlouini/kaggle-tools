---
name: kaggle-competition-playbook
description: Use when starting, planning or steering any Kaggle competition — reading the metric, choosing CV, deciding which experiment to run next, interpreting CV/LB gaps, ensembling, and managing submission/GPU quota
---

# Kaggle Competition Playbook

Cross-competition lessons from AIMO3, BirdCLEF 2026, ROGII Wellbore, Nemotron
reasoning, Biohub cell tracking, March Mania, Stanford RNA 3D and ARC-AGI-3.
Domain-specific skills: `kaggle-code-competitions`, `kaggle-audio-bioacoustics`,
`kaggle-llm-reasoning`, `kaggle-llm-finetuning`, `kaggle-tabular`.

## 1. Before any modelling: read the metric *code*

Fetch the organisers' scorer and read it line by line. Look for:

- **Skipped / clipped terms.** Biohub's node-count penalty had `max(0, …)` but no
  `min(1, …)`: predicting fewer nodes than `T_true` *boosted* the score, and 11×
  over-detection scored exactly 0.
- **Parameters shipped in the data.** Biohub's `T_true` sat in the geff metadata
  (`estimated_number_of_nodes`), which made it a directly tunable knob.
- **Aggregation weights.** Micro-averaged by GT edges meant two of four volumes
  decided ~95% of the score, so effort on the other two was wasted.
- **Target definition.** ROGII scored against hidden `manualTVT`, not the train
  `TVT` column. Blending toward train `TVT` looked great locally and cost +0.077 LB.
- **Sparse annotation.** If unmatched predictions aren't penalised, recall is cheap.

Write the findings down (`FINDINGS.md`) before writing a model.

## 2. Build the harness first

1. Reproduce the official metric locally (vendor their code if possible).
2. Pick a CV that mimics how the test is drawn (spatial / group / time / by file).
   Random hold-out inflated ROGII scores for anything using neighbour info.
   Use `kaggle_tools.cv.spatial_group_folds` / `rare_safe_folds`.
3. Submit a trivial baseline (constant, prior or public notebook) to calibrate CV→LB.
4. Keep a submission ledger (`kaggle_tools.tracking.Ledger`, `kt ledger`) with
   *every* run, including regressions, TIMEOUT and NO_SCORE.

## 3. Trust, but verify, the CV→LB transfer

Track the ratio `CV/LB` per family of change. Offline gains often do not transfer.

- BirdCLEF: a CNN student gave +0.034 OOF and **0** LB; Perch Bridge +0.0003 OOF and
  −0.011 LB; BG-aug +0.027 OOF and −0.006 LB. When 3–4 variants of one family are LB-dead,
  **declare the family dead** and write it in a "do NOT retry" list.
- A local "oracle" on the few visible test rows is a do-no-harm check, **not** an LB
  proxy (ROGII: 0.005 locally vs 9.15 LB for the same model).
- Tiny public LBs (BirdCLEF public = 3 windows / 34% slice) are mostly noise;
  ±0.001 is not a signal.

## 4. Choose experiments like a scientist

- **One variable per run.** Two simultaneous changes that regress cannot be
  attributed (Nemotron v28: alpha + lm_head). Isolation pairs (A alone, B alone,
  A+B) found that two BirdCLEF regressions partially cancelled each other.
- **Structural levers beat tweaks.** Plateau-breakers were: a better backbone
  (Perch v2, +0.096), a temporal model (ProtoSSM, +0.018), a second complementary arm
  + rank-blend (+0.011), the right model size (AIMO: GPT-OSS-120B), a novel signal
  (ROGII particle-filter features, −1.48 RMSE CV). Weight sweeps and alpha tuning
  rarely moved the LB.
- **Read the top public notebooks early and often.** Several of the biggest jumps came
  from forking a strong public notebook and adding one's own edge on top.
- **When stuck, retreat to understanding** (EDA, error analysis per group/well/species)
  instead of launching more experiments.
- **When score = 0 or errors appear, check the simplest thing first** (a path, a
  missing wheel, a wrong file) and read the *working* reference code before adding
  complexity. AIMO3 lost 12 versions adding threading/fallbacks while the real bug
  was one wrong tiktoken path.

## 5. Ensembling

- For rank metrics (AUC, Spearman), **rank-blend, never score-blend**
  (`kaggle_tools.ensemble.rank_blend`). Score-blend lets magnitude, not weight, decide.
- Diversity beats strength: an extra model from the same family (AutoGluon on top
  of LGB/CatBoost/XGB) got a stacking weight of ~0.
- Fit blend weights on OOF (`hill_climb`), sanity-check on a held-out fold.
- Post-processing tuned for someone else's model rarely transfers to yours.

## 6. Hyper-parameters

Well-tuned public params are hard to beat: 20 Optuna trials lost to hardcoded
LightGBM params on ROGII. Spend HPO budget only after features/architecture have
plateaued, use a proxy on the *hard* folds, and use TPE + MedianPruner.

## 7. Versioning and quota discipline

- Name every result by its **Kaggle kernel version number**, never a private "v".
- Kernels pin dataset versions at push time: `kaggle datasets version` first,
  then re-push the kernel.
- GPU quota (~30 h/week) goes fast. Minimal changes per run, and sanity-check
  artefacts (e.g. adapter file size > 0) before submitting.
- Scoring workers vary in speed: identical code can score once and TIMEOUT next.
  Keep a healthy time margin and re-submit before concluding code is broken.

## 8. Agentic workflow tips

- Keep a per-competition `CLAUDE.md` with: metric facts, data paths, the LB tracker,
  "what works", and "failed — do NOT retry" with the evidence. It stops re-running
  dead ideas across sessions.
- Put long-running training/kernels in the background and poll status; download the
  kernel log on any ERROR before theorising.
