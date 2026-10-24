---
name: kaggle-competition-playbook
description: Use when starting, planning or steering any Kaggle competition — reading the metric, choosing CV, deciding which experiment to run next, interpreting CV/LB gaps, ensembling, and managing submission/GPU quota
---

# Kaggle Competition Playbook

Cross-competition lessons from AIMO3, BirdCLEF 2026, ROGII Wellbore, Nemotron
reasoning, Biohub cell tracking, March Mania, Stanford RNA 3D and ARC-AGI-3.
Domain-specific skills: `kaggle-code-competitions`, `kaggle-audio-bioacoustics`,
`kaggle-llm-reasoning`, `kaggle-llm-finetuning`, `kaggle-tabular`,
`kaggle-medical-imaging`. Read the latter for weak report labels, anatomical
inputs, teacher dependencies and site-held-out evaluation.

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
- **With noisy training labels, OOF scored against those labels can reward fitting the
  noise.** RSNA Knee 2026 (LLM-extracted report labels): leak-free self-distillation
  raised every local metric (report-label OOF +0.008, confident-label OOF +0.004, a
  58-study radiologist set +0.007) and lowered the LB twice (−0.001 ConvNeXt, −0.002
  CoAtNet). Use a controlled LB A/B alongside leak-free validation to check transfer;
  these small public-LB deltas do not establish that fitting label noise caused it.
- When labels are noisy, score OOF on several label views (all labels, only confident
  labels, the small expert-labelled set) and track each one's correlation with the LB
  across submissions; there, the confident-label subset tracked the LB best (r≈0.95 over
  6 submissions) and the report-label OOF understated a backbone gain by ~3×.
  Treat that six-run correlation as exploratory: selecting a label view repeatedly
  against public LB can overfit it. Preserve an independent holdout for confirmation.
- **Audit the entire target-generation pipeline.** OOF labels can still expose
  a student's validation cases through upstream teachers, calibration or target
  selection. Exclude the outer holdout from each competition-trained dependency.
  Also distinguish duplicate/patient grouping from unseen-site evaluation;
  stratifying sites across folds does not hold them out. The RSNA Knee
  [5th/22nd-place review](../kaggle-medical-imaging/references/rsna-knee-2026.md)
  records explicit teacher leakage in 5th place and a larger, still biased teacher
  reference in 22nd place.

## 4. Choose experiments like a scientist

- **One variable per run.** Two simultaneous changes that regress cannot be
  attributed (Nemotron v28: alpha + lm_head). Isolation pairs (A alone, B alone,
  A+B) found that two BirdCLEF regressions partially cancelled each other.
- With weak labels, pilot better target definitions and complementary inputs
  before adding another similar model. Hold training duration fixed when measuring
  a label change; the reviewed RSNA Knee 5th-place +0.008 private step changed both
  labels and epochs, while a matched crop step added +0.003 private. See the
  [case study](../kaggle-medical-imaging/references/rsna-knee-2026.md) for attribution
  limits and distinct full-context versus restricted-compartment experiments.
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

- For rank metrics (AUC, Spearman), compare rank blending and probability/score
  averaging on matched held-out predictions. `kaggle_tools.ensemble.rank_blend`
  supplies a common scale for differently calibrated arms; it is not universally
  superior. RSNA Knee 5th place used probability averaging followed by rank stages;
  22nd used probability averaging (see the
  [case study](../kaggle-medical-imaging/references/rsna-knee-2026.md)). Re-ranking
  between stages cannot generally be collapsed into one original-score average.
- Diversity beats strength: an extra model from the same family (AutoGluon on top
  of LGB/CatBoost/XGB) got a stacking weight of ~0.
- Fit blend weights on OOF (`hill_climb`), sanity-check on a held-out fold.
  Choose by marginal ensemble gain under the runtime budget, not correlation
  alone. Recheck specialists after labels/base models improve; the reviewed RSNA
  Knee specialists added less on a stronger regional base than on a weaker base.
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

## 8. External data and public weights

- Check each competition's external-data and winner-licensing rules against the
  original data terms and model provenance. A downloadable re-upload or permissive
  model licence does not establish permission to use the upstream training data.
  Gated access alone also does not settle eligibility; use the organisers' rules and
  clarifications for that competition. For RSNA Knee, start with the
  [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
- Inventory top public notebooks with
  `kaggle kernels pull <owner>/<slug> -p /tmp/notebook-audit/ -m`, then read
  `dataset_sources`/`model_sources` and trace upstream weights. Record notebook and
  checkpoint versions, provenance links and scores. Attachments alone cannot prove
  training provenance or establish a performance ceiling for eligible models.
- Weights trained using labels from *all* training rows contaminate CV when reused
  in every fold. Use checkpoints trained without each validation fold or an independent
  holdout absent from both pretraining and fine-tuning. Expert annotations on the same
  exposed studies do not make them untouched. Use public LB only as a transfer check,
  and distinguish supervised leakage from unlabelled pretraining permitted by the rules.

## 9. Agentic workflow tips

- Keep a per-competition `CLAUDE.md` with: metric facts, data paths, the LB tracker,
  "what works", and "failed — do NOT retry" with the evidence. It stops re-running
  dead ideas across sessions.
- Put long-running training/kernels in the background and poll status; download the
  kernel log on any ERROR before theorising.
- Background jobs may be terminated with the agent's execution session. On Linux
  hosts with a systemd user manager, run multi-hour queues as transient services:
  `systemd-run --user --unit <unique-name> --same-dir --service-type=exec /bin/bash /absolute/path/queue.sh`.
  Services use the manager's environment; activate the intended environment in the
  script or use absolute executable paths and explicit `--setenv` values. Inspect
  `journalctl --user -u <unique-name>` and `systemctl --user status <unique-name>`.
  Set memory limits from measured needs, not a fixed 40 GB. If the service must survive
  logout, check user lingering and enable it only when needed. Use the host's scheduler
  where systemd is unavailable. See [systemd-run](https://www.freedesktop.org/software/systemd/man/latest/systemd-run.html).
  Checkpoint model, optimizer, scheduler, RNG and progress state, and implement resume;
  a checkpoint alone does not restart a killed run.
- In the RSNA Knee ZFS setup, memmapped corpora coincided with training OOM kills.
  Linux ZFS can hold mapped data in both page cache and ARC
  ([OpenZFS discussion](https://github.com/openzfs/zfs/issues/225)). Measure process,
  cgroup and ARC memory before changing readers. Bounded `os.pread` reads are an
  alternative to benchmark; `posix_fadvise(DONTNEED)` is advisory, not an ARC memory
  cap or a guarantee against OOM.
- Gate automatic submissions on checks you have validated by hand once; a wrong gate
  silently skips a good submission (see `kaggle-code-competitions`, rehearsal).
