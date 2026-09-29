---
name: kaggle-audio-bioacoustics
description: Use when working on Kaggle audio/bioacoustics competitions like BirdCLEF — row_id parsing, pretrained bioacoustic backbones (Perch, BirdNET), per-species probes, temporal models, mel spectrograms, CV/LB gap, rare and zero-sample species, CPU inference budget
---

# Kaggle Audio Bioacoustics (BirdCLEF-style)

Task: multi-label species presence per 5 s window of long soundscapes; metric usually
column-wise macro ROC-AUC (columns without positives skipped). BirdCLEF 2026 path:
0.758 (EffNet) → 0.904 (Perch v2) → 0.927 (+ProtoSSM) → 0.940 (two arms, rank-blend).

## row_id and per-window predictions

```python
from kaggle_tools.audio import parse_row_id, window_row_ids, frame_windows

stem, end_sec = parse_row_id("BC2026_Test_0001_S05_20250227_010002_5")  # window 0 = [0, 5) s
```

Return a separate vector **per row_id**. Averaging over the file raises no error but
costs ~0.1 AUC. Default/prior = uniform `1/n_species` (matches sample_submission).

## What actually moved the LB (in order of impact)

1. **Pretrained bioacoustic backbone embeddings** (Google Perch v2, 1536-d, trained on
   ~15k species): +0.096 over a from-scratch EfficientNet on mels.
2. **Per-species probes** on embeddings (logistic/MLP, PCA 1536→128 + scaler) for
   species with enough labelled data; raw backbone logits for the rest.
3. **Temporal model across windows** (ProtoSSM on 12-window sequences): +0.018 —
   the hidden test has more temporal structure than local OOF suggested.
4. **A second, complementary arm** (distilled SED CNN with clip + frame-max heads,
   Gaussian smoothing across windows σ≈0.65) combined by **per-column rank-blend**:
   +0.011, after a long plateau of score-blending. Sweep the arm weight (0.6/0.4 →
   0.5/0.5 gave +0.001).
5. **ONNX Perch** instead of TF SavedModel: ~10 min of CPU budget freed.

## Dead ends (OOF gains that did NOT transfer — don't retry blindly)

- Fine-tuning the backbone alone: 0.742 LB ≈ raw frozen backbone. Lift comes from
  probes + temporal + ensemble layers, not the backbone.
- Mel-spectrogram CNN students (EffNet-B0): +0.034 OOF, 0 LB, blend or replace.
- Pseudo-labelled extra probes (−0.008), probes for zero-sample species (−0.009),
  BirdNET meta-LR (−0.008), cross-domain contrastive "bridge" (−0.011),
  other people's post-processing stacks (−0.001 to −0.026), k-NN centroid swaps.
- Heuristic: when OOF is measured on few labelled soundscapes, trust *structural*
  changes validated on LB, not OOF deltas.

## Mel spectrograms (if training CNNs)

```python
SR, WIN = 32_000, 160_000  # 5 s
N_FFT, HOP, N_MELS, FMIN, FMAX = 1024, 512, 128, 50, 14_000
mel = amp_to_db(mel_transform(wave).clamp(min=1e-10))  # clamp BEFORE dB → no NaN
mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)  # per-window min-max
```

PANNs/CNN14: use bfloat16 (fp16 overflows from batch 1), clip grads (max_norm=5).

## Rare species

Focal loss (γ=2) or inverse-frequency sampling; soundscape background injection
(p≈0.5) to bridge clip→soundscape shift; mixup with rare-weighted partners; EMA.
Merge prior-year data for overlapping species (per-row `audio_dir` column).

## CV

Split at **file** level (`MultilabelStratifiedKFold` or
`kaggle_tools.cv.rare_safe_folds` with file groups); pin species with <5 clips to
train. Expect a large clip-CV → soundscape-LB gap (0.94 → 0.75 for a clip model):
domain shift, zero-sample species (AUC≈0.5), geography. Validate on labelled
train soundscapes whenever they exist.

## Inference budget (CPU-only, ~90 min)

- Batch all windows of a file (`frame_windows`) through the model; never loop
  per-window on CPU with librosa.
- Progressive submission writes after each stage (uniform → backbone → probes →
  ensemble → final). The scoring environment changed mid-competition and only this
  pattern kept scoring.
- Build the submission with one numpy assignment (per-row iloc was O(n²)).
- 5-fold ensembles of a heavy arm can blow the budget; distil instead.
