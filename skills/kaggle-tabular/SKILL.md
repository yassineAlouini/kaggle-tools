---
name: kaggle-tabular
description: Use when working on Kaggle tabular, time-series or geospatial regression/classification — GBDT stacks (LightGBM/CatBoost/XGBoost), tabular foundation models (TabICL/TabPFN), spatial/group CV, feature ablations, stacking and hill-climbing, Optuna, local training + Kaggle inference
---

# Kaggle Tabular / Geospatial

From ROGII wellbore geology (RMSE, LB 15.9 → 8.05), March Mania and others.

## Validation first

- Use **spatial K-fold** (K-means on per-entity mean coordinates) or group K-fold for
  anything that uses neighbour/entity information
  (`kaggle_tools.cv.spatial_group_folds`). Random hold-out made neighbour-based methods
  look far better than they were.
- A good CV here was slightly pessimistic but aligned (CV/LB ≈ 1.02). Track the ratio.
- Tune on a proxy of the **hard** folds (e.g. 2 hardest spatial folds), not the average.

## What worked

- **Signals, not more features.** A particle filter (state-space dip/rate model with
  a likelihood on the log signal) turned into 3 features gave −1.48 RMSE CV; it
  became the #1 feature. Look for physics / state-space / matching signals none of the
  existing features carry.
- Feed competing heuristics (offset-well drift, DP drift) **as features** to the GBDT
  rather than blending their predictions.
- Augment across the test-time conditioning distribution (e.g. several random
  "prediction start" draws per entity).
- **Stack diverse families**: 3 LightGBM + 3 CatBoost + a tabular foundation model
  (TabICL: −0.08 stacked CV, ~3× that on LB). Hill-climb weights on OOF
  (`kaggle_tools.ensemble.hill_climb`).
- Blend with the best public physical/heuristic model and **sweep its weight on the LB**
  (0.90 → 0.80 → 0.75 kept improving).

## What didn't

- AutoGluon as an extra family: internally the same GBDTs → OOF highly correlated →
  stack weight ≈ 0.
- 20 Optuna trials vs well-tuned public params: lost. HPO is a late-stage lever.
- Sequence nets (Conv1D, PatchTST, cross-attention) on few hundred entities: won easy
  folds, blew up on geological outliers; overfit in 1–2 epochs.
- Features unavailable (or differently distributed) at test time. Check the test
  schema before engineering from a column.
- Shrinkage/fade post-processing copied from another team's writeup: it corrected
  *their* model's overconfidence, not a universal property.
- Isolate every feature block with a factorial ablation (with/without each block),
  since blocks can be neutral alone and harmful together.

## Train locally, infer on Kaggle

- Train the heavy stack locally (RTX 3090: ~1.5 h vs 8 h on Kaggle), upload artifacts
  as a dataset, keep the Kaggle kernel inference-only (26 min). Expect a tiny LB drift
  from different hardware/float order.
- Ship data (arrays, context sets), not device-bound pickles. For in-context models
  (TabICL/TabPFN), re-fit on the live GPU from shipped context arrays (<1 s/fold).
- `kaggle datasets version` **before** re-pushing the kernel (versions pin at push).
- Pin `"machine_shape": "NvidiaTeslaT4"`; the default P100 can't run current torch.
- Upstream public artifact datasets get restructured without notice. Copy what you
  depend on into your own dataset.

## Leakage and target gotchas

- Know exactly which column the metric scores against (ROGII: hidden `manualTVT`, not
  the train `TVT`). Tricks that pull predictions toward a proxy target can hurt.
- Target encoding and any per-entity statistics must be out-of-fold
  (`kaggle_tools.features.oof_target_encode`).
