---
name: kaggle-llm-evolution
description: Use when running an AlphaEvolve-style loop (an LLM writes search programs, an exact verifier scores them) on a Kaggle optimisation or circuit/compiler competition — choosing the representation and fitness, islands and lineage, calibrating on known optima before claiming records, spotting dead ends with counting bounds, sharing LLM quota with other agents, live W&B tracking
---

# LLM-Guided Program Evolution (AlphaEvolve-style)

Loop: an LLM (via `claude -p`) writes a complete program; a harness runs it in a
subprocess with a time and memory cap; an **exact** verifier scores the output; the
result, the printed progress lines and the parent's code go back into the next prompt.
Islands with different research directions, parent selection and migration keep the
population diverse.

Evidence below is from Google Quantum AI "Verifiable Quantum Circuit Compilation"
(GVQCC, Nov 2026), where the scores are Toffoli/qubit counts checked by an official
simulator. Proposer: Claude Fable 5.2 unless noted.

## When it works and when it does not

| target | representation the LLM edited | result |
|---|---|---|
| chemistry loader (212 Toffolis) | gate list, "return an improved circuit" | 20/20 valid, 0 gain: only local edits, flat fitness (Sonnet) |
| same loader | AND-plan IR (products of XOR forms), exact plan check | 5 programs, all valid at 212, 0 gain; the shipped loader sat on a counting bound |
| GF(2^n) multiplier bilinear rank | `search(n, poly, seconds, seed) -> scheme` | rebuilt the known best 24 / 42 / 99 (n = 8 / 12 / 24) from scratch in 0.1-270 s, where a hand-written Rust flip-graph walk stalled at 32 (n = 8) and 111 (n = 12) after 8 x 40 min |
| same, one below the record (41 at n = 12) | same, warm-started | 19/30 programs, all 42: SAT repair UNSAT on small neighbourhoods, best 41-term near miss 16 wrong bits |

What made the difference: the LLM is good at **choosing an architecture** (tower
fields, CRT/interpolation, Chudnovsky-style evaluation, SAT neighbourhoods) and weak
at **micro-editing** a long artefact. Ask for a *search program* that builds the
artefact, never for the artefact itself.

## Setup that worked

- **Program contract**: one function, a time budget argument, "always return your best
  valid answer". A crash, timeout or invalid output scores nothing; give the seed a
  trivially valid fallback (schoolbook scheme, lookup circuit).
- **Exact verifier, fast**: tensor identity, truth tables as 64-bit ints, or the
  official simulator. Grade invalid outputs (wrong bits) so near misses can be selected.
- **Prompt = brief + round**: the brief states the objective, the format, what is
  known (the record, what already failed and why); the round gives the island's
  direction, the parent's code and result, and the last ~14 results with one-line
  ideas. Ask for `IDEA: <one sentence>` + one code block; parse both.
- **Printed progress lines are the best feedback channel**: the child sees
  "41 terms: min wrong bits 16" or "SAT UNSAT for drop 2" and stops repeating it.
- **Islands = research directions**, not random seeds (e.g. structured descent /
  alternative constructions / exact SAT repair). Migrate the best program every few
  generations.
- **Run as a scheduled job** (Dagu queue) with a W&B sidecar that mirrors the JSONL log:
  per-candidate score, best-so-far, per-island score, and a table with id, parent,
  generation, island, idea and program, so lineage can be followed.

## Calibrate before claiming

A search that cannot recover a **known** optimum from a worse start says nothing when
it fails to beat the record. Gate in order: known small case (n = 8: 24) → known target
size (n = 12: 42) → record attempt (41). In GVQCC the calibration passed only for the
LLM loop, not for the hand-written walker; that is what justified spending quota on
the record attempt.

## Spot dead ends early: count degrees of freedom

Before evolving, estimate the counting bound: bits of freedom per unit of cost versus
bits of constraint. GVQCC loaders: one product carries ~212 bits (two low forms, two
high forms, one output set); a generic 15 x 76 x 30 tensor needs ~161 products = the
shipped heavy-loader cost, and the cheap loaders also sat on their bound. When the
shipped artefact is *on* the bound, the data was co-designed with it: no planner or
local edit can win, and the only lever is the data (here a chemistry table, which a
pricing tool showed costs ~4 x 10^4 times the Toffoli saved per change). Check the
"free" slack too (padded rows, don't-care entries) — the shipped circuits already used
the padded rows.

## Operational traps

- **Quota is shared.** `claude -p` on a subscription shares the five-hour limit with any
  other agent on the account: a 24-candidate run hit the limit after 3 candidates and
  stalled the main research agent; 21 candidates failed in 30 s. Retry failed calls
  every ~10 min instead of burning candidates, chain runs one after another, and keep
  the main agent's critical jobs independent of the LLM.
- **Relative paths break in the runner**: the subprocess runs with `cwd` = the work
  dir, so resolve program paths to absolute before passing them (a warm start silently
  failed this way).
- **Third-party scoring tools may assume defaults**: a public pricing tool read the word
  width as 8 bits when the npz had no `brot` field, giving lambda_eff 802 instead of
  46.9. Check the base metrics against the official verifier before trusting prices.
- **Python package shadowed by a run folder**: a `./wandb/` output directory in the repo
  root shadows the `wandb` package; start the sidecar from another directory.

## Brainstorming with the same model

When stuck, send the LLM a standalone brief with the facts and **not your conclusion**
("attack this counting argument; rank loopholes by expected value with a one-hour
test each"). In GVQCC this produced the counting-bound reading and the joint
plan+table test that closed the loader line; one of its claims (loaders "on the
bound") was wrong as first stated and had to be checked against measurements.
