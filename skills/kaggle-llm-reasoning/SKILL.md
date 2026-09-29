---
name: kaggle-llm-reasoning
description: Use when working on Kaggle LLM reasoning/inference competitions like AIMO — model choice, vLLM on H100, tool-integrated reasoning (TIR), self-consistency voting, answer extraction, inference-server API, time budget, offline tokenizer files
---

# Kaggle LLM Reasoning Competitions (AIMO-style)

Recipe that reached 38–44/50 on AIMO3: **the strongest open reasoning model that fits**
(GPT-OSS-120B, 4-bit MoE) + **SC-TIR** (N parallel chains × M code-execute rounds) +
**confidence-weighted voting** + a dynamic per-problem time budget.

## Model choice dominates

7B (AIMO1 winner) → 32B → 120B explains most of the score gap; no amount of
engineering compensates. GPT-OSS-120B (MXFP4 MoE, ~5B active) fits on **one H100 80 GB**:
use `CUDA_VISIBLE_DEVICES=0`, `tensor_parallel_size=1`. Don't switch to TP=2 without an
actual OOM in the vLLM log. Use the "high" reasoning level.

## Offline setup gotchas

- vLLM and tokenizer assets must come from attached datasets/kernels.
- GPT-OSS "harmony" needs its **own tiktoken vocab** files. The generic
  `cl100k_base`/`o200k_base` files are the wrong ones:
  `os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/kaggle/tmp/setup/tiktoken_encodings"`
  after extracting the correct archive. Twelve versions (all 0/50) were lost to this path.
- Copy `kernel-metadata.json` sources (kernel vs dataset, docker image) *exactly* from
  a working public notebook. A kernel listed under `dataset_sources` won't mount.
- When the score is 0, diff against a **working** public notebook
  (`kaggle kernels pull`) before adding complexity (threads, fallbacks, gRPC).

## Inference server

```python
import kaggle_evaluation.aimo_3_inference_server as srv  # match the comp's version

solver = Solver(cfg)  # load vLLM FIRST (page-cache preload → ~60–120 s)
server = srv.AIMO3InferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    server.serve()
else:
    server.run_local_gateway(("reference.csv",))
```

`predict` must always return an int — never `None` (fallback `0`). Apply `% 100000`
(or whatever the answer range is).

## SC-TIR loop

1. N chains at temperature > 0 (N≈16–32 for a 120B; 48 for a 7B).
2. Extract ```python blocks, execute in a **pre-warmed sandbox pool** with a timeout,
   feed stdout/traceback back, continue; M≈4–8 rounds.
3. Extract `\boxed{}` (fallback patterns) — `kaggle_tools.llm.extract_integer_answer`.
4. Vote. Early-stop when k chains agree (`kaggle_tools.llm.ConsensusVoter`).

## Voting

- Plain majority (`majority_vote`) is the baseline.
- **Inverse-entropy weighting** (weight each chain by 1 / mean token entropy from
  vLLM logprobs) beat majority in 29/30 configurations
  (`kaggle_tools.llm.weighted_vote`).

## Time budget

- Hard wall clock (AIMO3: 5 h incl. ~10 min setup). Set the internal limit
  ~1000 s below it (≈17000 s); 28800 s got the kernel killed.
- Allocate per problem dynamically: `TimeBudget.per_item(n_left)`, spend more on
  problems without early consensus, and always leave time for the remaining ones.
- Speculative decoding (EAGLE-3) gave ~+40% tokens/s → more TIR rounds.

## Local iteration

Test the loop with a 7–14B model locally (RTX 3090), measure accuracy vs samples
and vs time per problem, then scale up on Kaggle. Log per-problem answers, votes and
timings to spot budget starvation.
