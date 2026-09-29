---
name: kaggle-llm-finetuning
description: Use when fine-tuning LLMs (LoRA/QLoRA, SFT, GRPO) inside Kaggle competitions — TRL/PEFT setup, packing, completion-only loss, data quality, adapter submission format, GPU quota and 12 h kernel limits
---

# LLM Fine-Tuning on Kaggle (LoRA / SFT)

From the NVIDIA Nemotron reasoning challenge (hybrid Mamba/MoE/attention model,
LoRA rank ≤ 32, adapters submitted as a zip).

## What works

| Lever | Effect |
|---|---|
| `target_modules="all-linear"` | Strong default; PEFT resolves attention, MoE experts and Mamba projections |
| `packing=True`, `packing_strategy="wrapped"` | ~3× faster. TRL's default `ffd` enables padding-free, which rejects custom collators |
| **Verified chain-of-thought data** (keep only CoTs whose final answer is rule-checked correct) | Biggest data lever (0.59 → 0.65) |
| Soft-balanced sampling across task types (α≈0.5) | Helps weak categories without drowning others |
| **Completion-only loss** (mask all but assistant spans; packing-aware collator) | +0.01 |

The gap to the top (0.66 → 0.84) was **not** LoRA architecture (identical module sets
to the 0.84 adapter). It came from data quantity/quality and more training steps.

## Pitfalls

- **Silent 0-byte adapter** after save (seen with `packing=False` + long seq). Always
  assert `os.path.getsize("adapter_model.safetensors") > 1_000_000` before packaging.
- Adding `lm_head` to LoRA targets balloons the adapter (150 MB → 4.3 GB), can OOM, and
  was neutral at ~6k examples.
- Tuning `target_modules` for *speed* doesn't work: step time is dominated by the base
  forward + gradient checkpointing.
- TRL moves fast (e.g. 0.29 dropped `DataCollatorForCompletionOnlyLM`): pin versions and
  subclass `DataCollatorForLanguageModeling` for custom masking.
- PEFT version mismatch between training and the scorer (new config fields) can make
  scoring error out; train with the version the scorer uses.
- Submission zip: `adapter_config.json` + `adapter_model.safetensors` at the **zip
  root**, not in a folder.

## Quota discipline

- Kaggle GPU ≈ 30 h/week, 12 h per session. Two consecutive 12 h timeouts burned most of
  a week: change **one** variable per run, estimate steps × step-time before pushing.
- Offline at eval: bundle `trl`/`peft` wheels as a dataset.
- Save intermediate checkpoints so a timeout still yields a usable adapter.

## Next levers, ranked

1. Scale verified CoT data: generate with the base model on all train problems,
   rule-verify, keep correct ones (2–3×).
2. Rebalance toward the weakest task type.
3. GRPO after SFT with rule-based rewards (the likely 0.78+ recipe; heavy engineering).
4. Faster training stacks (Unsloth, Tinker) → more steps in the same quota. Tinker
   fused high-rank LoRA can be SVD-compressed back to rank 32 for submission.
