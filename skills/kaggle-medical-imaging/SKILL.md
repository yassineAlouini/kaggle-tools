---
name: kaggle-medical-imaging
description: Plan and improve Kaggle MRI or CT classification pipelines with weak report labels, anatomical crops, multi-series inputs, teacher supervision, and site-aware validation. Use for medical-imaging competition experiments and solution reviews, not clinical diagnosis.
---

# Kaggle medical imaging

Start from the competition's target definitions, available annotations and test
inputs. For quantitative evidence and limits, read the
[RSNA Knee 2026 case study](references/rsna-knee-2026.md) when choosing among label,
localization, validation or ensemble changes. It reviews the 5th- and 22nd-place
writeups; it is not a top-three survey or a reproduced benchmark.

## Improve supervision before expanding the ensemble

- Audit per-finding missingness, uncertainty and definition mismatches. A finding
  omitted from a report is not automatically a negative. Preserve soft targets
  and confidence when supported by the label source.
- Compare calibrated report labels, image-assisted adjudication and teacher-fused
  targets in a pilot. Hold backbone, views and training length fixed when
  attributing a gain. Choose permitted teachers within the user's compute/cost
  scope; these lessons do not require a particular model or external service.
- Treat a small expert set as a limited resource. Cases repeatedly used for
  prompts, calibration, blend selection or candidate ranking are development data.
  Preserve an independent audit where feasible; report what influenced selection.
- Track the complete label dependency graph. A teacher trained on a downstream
  student's validation cases can contaminate that student's training targets even
  when the target rows carry OOF predictions. Exclude the outer holdout from
  competition-trained teachers, calibration, target selection and student fitting;
  construct inner cross-fitted targets inside outer training data when needed.

## Test the anatomical hypothesis explicitly

- Distinguish **more detail** (full field plus higher-resolution regional views)
  from **less shortcut information** (a specialist restricted to one anatomical
  compartment). These are different experiments; neither is universally better.
- Match labels, folds and backbone for the input comparison. Audit localization,
  physical crop sizes, slab coverage, oblique series and failures before judging
  classifier quality. Measure the fallback on failed localization, too.
- Normalize orientation using patient geometry and verified laterality. Apply
  only augmentations that preserve target meaning; do not turn one team's no-flip
  setting into a universal ban on anatomically correct transforms.
- Audit sequence/contrast rules against headers and image appearance. Vendor
  exceptions are dataset-specific; do not copy MRI thresholds as general rules.

## Validate domains and marginal additions

- Group repeated reports/patients and separately test held-out site/protocol
  behavior where relevant. Stratifying scanner proxies across folds does not
  measure transfer to an unseen site. Use
  `kaggle_tools.cv.rare_safe_folds` with the intended groups; do not assume it
  reconstructs hospital identities or validates teacher dependencies. Inspect its
  `TRAIN_ONLY` exclusions and coverage; use a different grouped split if pinning
  rare groups would remove the sites the evaluation must test.
- A larger image-teacher reference can reduce some noise but retains teacher bias.
  Keep metric/reference definitions fixed across candidates and track transfer to
  independent evidence. Avoid comparing binary-label correlation directly with
  continuous prediction correlation when diagnosing coupled findings.
- Compare view/finding specialists against another same-view model under a fixed
  runtime budget. Choose by held-out marginal ensemble gain, not correlation alone.
  Recheck the addition after improving labels or the base model, and use an
  independent check after screening many variants.
- Compare probability and rank blends on matched held-out predictions
  (`kaggle_tools.ensemble.rank_blend`, `hill_climb`). Per-finding weights and
  sequential re-ranking can matter; do not flatten repeated rank stages into one
  average of original probabilities.
- Profile decoding, localization, crop generation and inference together. Share
  equivalent preprocessing or features, verify prediction parity, and measure
  whether the saved time enables a useful specialist. Model count is not a goal.

## Review a solution without overstating it

Keep official selected results separate from retrospective best unselected runs;
public from private scores; individual checkpoints from fold-averaged families;
and controlled ablations from bundled changes. Link primary sources and label
unreproduced claims. Extract testable actions, including negative results, rather
than assigning the entire leaderboard gap to a technique.
