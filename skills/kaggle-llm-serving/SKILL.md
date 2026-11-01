---
name: kaggle-llm-serving
description: Use when an LLM inference notebook on Kaggle GPUs (4x L4, 2x T4, single 24 GB cards) is throughput-bound — tokens/s too low for the runtime limit, choosing tensor vs data parallel, fitting a 20-30B model on one 24 GB card, KV-cache or Mamba/GDN state limiting concurrency, MTP/speculative decoding depth, int4/fp8 weights or KV, vLLM patch stacks (HyperQwen, Merlin), or benchmarking serving configs on RunPod before spending Kaggle quota
---

# Kaggle LLM serving throughput (vLLM on 4x L4)

On Kaggle's 4x L4 (24 GB, ~300 GB/s, PCIe, no P2P) aggregate decode tokens/s =
**running streams x per-stream rate**. The stream count is capped by memory
(KV + per-stream state), so serving work is mostly a memory problem. Evidence below is
from ARC Prize 2026 ARC-AGI-2 (Qwen3.8-27B, vLLM 0.31, Oct 2026).

**Faster serving is not a better score.** Every config change goes through a quality gate
at a fixed token budget before it reaches a submission. See the Quality gate section.

## Measure first

Throughput numbers are only comparable when the bench follows these rules:

- Use the real workload. Continue real traces at spread-out depths (0 to 48k), with temperature as
  in production. Short-prompt probes overstate throughput 2-3x: 8 short streams gave 295 tok/s,
  where the real 24-stream point gave 239.
- Open the window only after **every** stream decodes. The window must have no prefill tokens,
  no waiting requests and no preemptions. Compute aggregate tok/s from the delta of vLLM's
  generation-token counter over a 180-240 s window.
- Log `running`, `waiting`, KV%, the MTP acceptance length (AL) and the context at the start and
  end of the window. When KV is near 100% with requests waiting, you are capacity-bound.
- Before ranking two configs, rerun the baseline. Count a gain only if the slower candidate run
  beats the faster baseline run by 5% or more.
- Profile one decode step (torch profiler, 20 iterations) before choosing a lever. On a 1xL4
  INT4 server the split was attention 45%, Marlin GEMM 44%, GDN 6%, sampler 0.4%.

## Layout: TP4 vs one replica per GPU (DP4)

| layout | what limits it | measured (Qwen3.8-27B) |
|---|---|---|
| FP8, TP4, MTP3, 16 streams | SHM all-reduce: ~80-100 ms per verify step at 32 streams | 345-353 tok/s; 24 streams 378 (+6.6%); 32 streams 393 |
| INT4, TP2 x 2 replicas | KV: ~12 running per replica | 233 per replica (RunPod) |
| **INT4, TP1 x 4 (DP4) + the memory stack below** | KV + state slots | **239 tok/s per L4** at 14.2 running; 2xL4 measured 447.6 (220.7 per server), so 4x is about 900 |

DP4 has no inter-GPU traffic, so the slow link stops mattering. It only works when the
model fits on one 24 GB card with room for KV. A 27B at FP8 (~27 GB) does not fit, so DP4
means 4-bit weights. If 4-bit costs too much accuracy, the remaining options are FP8 TP4 at
more streams or TP2 x 2. "About 900" is projected from the 2xL4 run; the 4xL4 DP4 run never
happened.

One DP4 replica: one server per GPU, each pinned with `CUDA_VISIBLE_DEVICES`, and the client
keeps each session on the same server so prefix-cache hits survive.

```bash
CUDA_VISIBLE_DEVICES=$i VLLM_EMBED_UVA=1 VLLM_SPEC_DECODE_ATTN=1 VLLM_INT4_MQ_3D=1 \
python -m vllm.entrypoints.openai.api_server --model $INT4_CKPT --port $((8000+i)) \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 65536 \
  --max-num-seqs 16 --max-num-batched-tokens 1024 --kv-cache-dtype int4_per_token_head \
  --attention-backend TRITON_ATTN --mamba-ssm-cache-dtype float16 --language-model-only \
  --enable-prefix-caching --use-replayssm --mamba-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Several of these flags and env vars are not in stock vLLM 0.31:

- `int4_per_token_head`, `VLLM_EMBED_UVA` and `VLLM_INT4_MQ_3D` come from the HyperQwen patch series
  (syv-ai/HyperQwen, commit ce9615c: 51 Python files).
- `--use-replayssm` is a backport of the upstream RecoverSSM PRs.

Ship the patched files as a Kaggle dataset and put that copy ahead of the installed package on
`PYTHONPATH`. That leaves the installed package untouched, and you can hash-check the files at boot.

## Fitting a 27B on one 24 GB L4 (stacked; each line was measured)

| lever | effect |
|---|---|
| W4A16 INT4 body (RedHat) + int8 lm_head, embeddings and MTP head; 40,960-row draft vocab | weights 13.9 GiB |
| Embeddings in pinned host RAM (`VLLM_EMBED_UVA=1`, HyperQwen) | frees ~1.3 GB of VRAM |
| `--max-num-batched-tokens 1024` | peak activation 1.5 GiB -> 0.38 GiB |
| `gpu_memory_utilization 0.95` | 0.97 OOMs in the sampler prewarm |
| int4 per-token-head KV + `VLLM_INT4_MQ_3D=1` | KV pool 137k -> 289k tokens. Without MQ_3D, the MTP verify falls back to a serial path: 3.6 vs 29 tok/s at 72k |
| fp16 GDN/Mamba state | halves the per-stream state |
| RecoverSSM (replay; upstream vLLM PRs #58863/#61309/#62625, backported) | drops the K draft state slots: 15 -> 6 blocks per stream; running streams 8 -> 12 |

**Hybrid models (GDN or Mamba layers) pay a hidden per-stream cost.** Each stream holds
`groups x (1+K)` state blocks, or `groups x (2+K)` with prefix caching (align mode). For
Qwen3.8-27B with MTP3 that is 15 blocks, or 435 MB: the same as 12.5k fp8 KV tokens, before any
context. KV quantization shrinks only the context part, so int4 KV gave ~1.5x more streams,
not 2x. On a capacity-bound server, **MTP1 beat MTP3 (197 vs 176 tok/s) until RecoverSSM** freed
the slots. With RecoverSSM, MTP3 won (239 vs 206).

## Kernel fixes worth checking

- **Int4 split-KV attention tiles.** On L4, the Triton verify kernel used tiles tuned for big GPUs.
  `BLOCK_M=32`, fp16 dot (not TF32) and `TILE=32` made it 2.0x faster (2.46 -> 1.23 ms per layer)
  and more accurate against an fp64 reference. Microbench one kernel at N=8, 29k context, 4 rows
  per sequence before and after a change.
- On TP, vLLM 0.31 all-gathers the full 248k-vocab logits on every rank (~64 MB per step). A
  top-K gather and `use_local_argmax_reduction` remove that traffic.
- These did **not** beat baseline by 5% on FP8 TP4: async scheduling, local draft argmax,
  MTP1/MTP5/no MTP, prefill batch 2048.

## Dead ends (don't repeat)

- 2-bit GGUF giants (>200B) with llama.cpp on 4xL4: 43 tok/s aggregate, 3 tok/s per request at
  long context, a 44 min load, 9.6 GPU-h for 0/4 tasks.
- Merlin engine: preflight refuses L4, and its 27B fast paths are gated to sm_86. On sm_89 only its
  memory savers apply.
- A smaller fast model instead (Qwen3-Next "Flash-Next" on 4xL4): 258 tok/s, no accuracy gain.
- `NCCL_SHM_USE_CUDA_MEMCPY=1`: the server hung at load.
- On RunPod L4 hosts, P2P copies return wrong data and NCCL P2P hangs. **Always set
  `NCCL_P2P_DISABLE=1`**, which also matches Kaggle. RunPod 4xL4 ran at 78% of Kaggle's speed
  on the same config, so compare A/Bs only on the same host.

## Quality gate (required before any submit)

- The fully stacked INT4 DP4 agent scored **1.0/16 vs 4.0/16** for FP8 TP4 on the same tasks
  and token budget. The quantization stack, the patches and the KV dtype can each cost accuracy,
  and the cause was never isolated. Ablate them one at a time, starting with INT4 weights + fp8 KV
  and no patches, before stacking.
- Compare at a fixed per-task token budget, on the same tasks, with explicit completion status.
  Treat runs that hit a deadline or time out as failures, not as budget stops.
- Public-eval gains overstate hidden-set gains for LLM agents. On ARC-AGI-2 the agent transferred
  at ~0.56x of its eval score, against ~0.92x for the NVARC baseline. An eval forecast of 34 scored 26.5.

## Quota and R&D logistics

- Do serving R&D on rented L4s (RunPod: about $5 for the whole DP4 study). Spend Kaggle quota only
  on the final validation.
- **Reserve the submission run's GPU hours first.** In ARC-AGI-2, validation notebooks used
  16 h of the 30 h week. The submission push then hit "Maximum batch GPU session count of 2" and
  then the quota limit, which left the best config unsubmitted.
- Give every rented pod an idle-kill guard that pulls results first. Stop each experiment once its
  signal is clear.

REQUIRED BACKGROUND: kaggle-code-competitions (offline wheels, re-run limits) and
kaggle-llm-reasoning (agent and voting side).
