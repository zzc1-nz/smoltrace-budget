# Design: Cost-Aware Evaluation Scheduling Layer for SMOLTRACE

## The problem this project attacks

SMOLTRACE is a good agent-evaluation engine. It instruments runs with
OpenTelemetry, tracks tokens / cost / latency, and publishes a leaderboard.
But it has one structural assumption baked in:

> **Every task is evaluated at the same fidelity, with the same model, at the
> same effort.** There is no way to say "spend a lot of measurement budget on
> this hard GAIA task and almost nothing on this easy arithmetic one."

That assumption hides a real problem:

> **Evaluation has a cost, and the evaluation strategy distorts model
> selection.**

If you evaluate a pool of models cheaply — few agent steps, no tools, a loose
grader — you get a leaderboard that is cheap to produce and *wrong*. This
project makes the evaluation budget an explicit, optimisable variable and
quantifies exactly that distortion.

## Core concepts

### Evaluation fidelity tier

A **tier** is a bundle of measurement-fidelity knobs. It is named after the
model class you would have to pay for to obtain that fidelity, which is how
evaluation cost is usually reasoned about in practice.

| tier | label (model class) | max_steps | tools | grader | relative cost |
|------|--------------------|-----------|-------|--------|---------------|
| `low` | `gpt-4o-mini` | 2 | disabled | keyword match | cheapest |
| `mid` | `gpt-4o` | 4 | enabled (web/file) | keyword match + LLM-ish semantic | mid |
| `high` | `claude-sonnet` | 6 | enabled (all) | strict keyword + semantic | most expensive |

Cost differences are real and large: agent steps dominate token consumption
(each step re-sends the growing conversation), and tool use adds observation
tokens. A cheap tier therefore emits far fewer tokens per task than a premium
tier.

### The three evaluation arms

The **same pool of candidate models** is evaluated in every arm, so every arm
produces a leaderboard over an identical row set and the rankings are directly
comparable.

- **(a) `all_high`** — every task measured at the `high` tier. Gold standard.
- **(b) `all_low`** — every task measured at the `low` tier. Cheapest.
- **(c) `allocated`** — the budget allocator assigns a tier per task.

### The budget allocator

Reads the task manifest (SMOLTRACE GAIA / Math / SimpleQA benchmark) plus
historical difficulty signals, and assigns a tier to every task. Three
policies ship:

1. **`by_category`** — static mapping from task category to tier.
   GAIA (multi-hop, tool-heavy) → `high`; Math → `mid`; SimpleQA → `low`.
2. **`by_pass_rate`** — uses historical per-task pass rates as a difficulty
   signal: tasks the pool historically fails are measured at higher fidelity,
   because that is where a cheap measurement is most likely to mis-rank models.
3. **`greedy_budget`** — under an explicit token budget, greedily spend on the
   tasks with the highest *ranking-risk* per token, where ranking-risk is
   estimated from historical pass rates and category.

## What is measured per task

Every evaluated task records:

```
task_id, category, difficulty, agent_type,
allocated_tier,          # low | mid | high
allocated_model,         # the model class label behind that tier
candidate_model,         # the model under test (leaderboard row owner)
prompt_tokens, completion_tokens, total_tokens,
latency_ms, steps,
passed,                  # graded outcome
```

The allocation decision is attached to SMOLTRACE's OTEL span as a **custom
attribute `allocation_tier`**, alongside the existing GenAI semantic
convention attributes, so allocation-aware analysis is possible downstream
without parsing our own side files.

## Analysis outputs

1. **Three leaderboards** — `all_high`, `all_low`, `allocated` — ranked by
   pass rate over identical model sets.
2. **Cost deltas** — total tokens / total USD per arm, and the percentage
   saved by the allocated arm versus the gold standard.
3. **Ranking stability** — top-3 membership and ordering agreement versus the
   gold-standard arm, per round and aggregated over 3 rounds. This is the
   "Y%" in *"cost down X%, top-3 rank stability changes Y%"*.
4. **Cost report** — total tokens, total cost, cost distribution by task
   category, and the headline finding: **which model the cheap arm wrongly
   crowns first**.

## Why 3 rounds

A single run cannot separate a real ranking shift from measurement noise. Three
independent rounds give a per-arm distribution of the top-3 ordering, so
stability is reported as agreement *across rounds* rather than as a single
comparison.

## Reuse, not rewrite

The layer is built strictly on SMOLTRACE primitives:

| capability | SMOLTRACE API reused |
|---|---|
| task loading | `smoltrace.core.load_test_cases_from_hf` semantics (frozen JSONL snapshot here) |
| OTEL setup | `smoltrace.otel.setup_inmemory_otel` |
| agent construction | `smoltrace.core.initialize_agent` |
| single-task evaluation | `smoltrace.core.evaluate_single_test` |
| trace extraction | `smoltrace.core.extract_traces` |
| metrics extraction | `smoltrace.core.extract_metrics` |
| leaderboard row | `smoltrace.utils.compute_leaderboard_row` |

Nothing in `smoltrace` is modified. The scheduler drives these functions with a
per-task model instance instead of one model for the whole run — which is
precisely the capability SMOLTRACE lacks.

## Model backend

The evaluator runs against a **calibrated simulated LM backend** that implements
`smolagents.Model`. It is deterministic given a seed and reproduces the
token / latency / correctness profile of each model class. This keeps the whole
132-task × 3-arm × 3-round matrix reproducible and free to run, while every
millisecond of token accounting, OTEL span emission, and leaderboard computation
is real. Swapping in `LiteLLMModel` for a live API is a one-line provider
change; the scheduler is backend-agnostic.
