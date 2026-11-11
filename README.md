# Cost-Aware Evaluation Scheduling for SMOLTRACE

**Evaluation itself has a cost — and the evaluation strategy distorts model selection.**

A cost-aware scheduling layer on top of [SMOLTRACE](https://github.com/Mandark-droid/SMOLTRACE),
the OpenTelemetry-native evaluation engine for Hugging Face `smolagents`. It treats
**evaluation budget as a first-class, optimisable variable** instead of spending the
same measurement effort on every task.

SMOLTRACE evaluates every task at the same fidelity with the same model. This project
adds the layer it lacks: a **pre-evaluation budget allocator** that assigns a
measurement-fidelity tier per task, then quantifies what that choice costs and what it
does to the leaderboard.

---

## The headline result

Measured on the full SMOLTRACE comprehensive benchmark — **132 tasks × 5 candidate
models × 3 arms × 3 rounds = 5,940 real evaluations, 30.9M tokens, 6 minutes**:

| | |
|---|---|
| **Total evaluation cost** | $20.35 (all_high) → **$8.79 (allocated)** |
| **Cost reduction** | **−56.79%** |
| **Top-3 rank stability** | 100% → **68.33%** (**−31.66%**) |
| **Models that keep their top-3 slot** | 2 of 3 |
| **Wrongly crowned by cheap evaluation** | **`balanced-13b`** (true rank: **#2**) |

> ### The insight
>
> **Spending less on evaluation is not free — it costs you the ranking.**
>
> The gold standard (`all_high`) crowns **`thorough-32b`** at 71.97%.
> Measuring the *same five models* cheaply (`all_low`) crowns **`balanced-13b`** at
> 56.82% instead — a model that is genuinely **#2**, while `thorough-32b` falls to
> **#4 of 5** (47.22%). The cheap evaluation is 29× cheaper and gets the answer wrong.
>
> The cause is structural, not statistical. A cheap tier gives the model fewer agent
> steps and no tool budget. That barely hurts a model that answers immediately
> (`quickthink-7b`), and badly hurts one that needs tools and deliberation to reach its
> true ability (`thorough-32b`). **Cheap evaluation systematically rewards models that
> are cheap to evaluate** — which is a different property from being good.

---

## Three leaderboards, same model pool, same tasks

Pooled over 3 rounds (396 evaluations per model per arm). Identical task set in every
arm, so the tables are directly comparable.

### (a) `all_high` — every task at full fidelity — **gold standard**

| rank | model | pass rate | passed / evaluated | tokens | cost |
|---:|---|---:|---:|---:|---:|
| 1 | **`thorough-32b`** | **71.97%** | 285 / 396 | 5,281,154 | $9.8742 |
| 2 | `balanced-13b` | 62.63% | 248 / 396 | 1,943,620 | $3.3887 |
| 3 | `quickthink-7b` | 58.08% | 230 / 396 | 1,079,030 | $1.7447 |
| 4 | `mathwhiz-9b` | 56.57% | 224 / 396 | 2,192,268 | $3.9365 |
| 5 | `tinyfast-3b` | 33.08% | 131 / 396 | 879,840 | $1.4055 |

**Total: 11,375,912 tokens · $20.3497**

### (b) `all_low` — every task at the cheapest fidelity

| rank | model | pass rate | passed / evaluated | tokens | cost |
|---:|---|---:|---:|---:|---:|
| 1 | **`balanced-13b`** ← *wrong winner* | **56.82%** | 225 / 396 | 1,705,826 | $0.1437 |
| 2 | `mathwhiz-9b` | 56.06% | 222 / 396 | 1,924,326 | $0.1660 |
| 3 | `quickthink-7b` | 53.28% | 211 / 396 | 1,078,954 | $0.0856 |
| 4 | `thorough-32b` ← *true winner* | 47.22% | 187 / 396 | 2,580,564 | $0.2329 |
| 5 | `tinyfast-3b` | 36.36% | 144 / 396 | 879,874 | $0.0692 |

**Total: 8,169,544 tokens · $0.6975** (3.4% of gold-standard cost — and it gets rank 1 wrong)

### (c) `allocated` — the budget allocator assigns a tier per task

| rank | model | pass rate | passed / evaluated | tokens | cost |
|---:|---|---:|---:|---:|---:|
| 1 | `balanced-13b` | 63.89% | 253 / 396 | 1,943,598 | $1.3806 |
| 2 | `mathwhiz-9b` | 63.38% | 251 / 396 | 2,192,272 | $1.5986 |
| 3 | `thorough-32b` | 61.87% | 245 / 396 | 5,281,018 | $4.7744 |
| 4 | `quickthink-7b` | 60.86% | 241 / 396 | 1,079,044 | $0.5750 |
| 5 | `tinyfast-3b` | 35.86% | 142 / 396 | 879,854 | $0.4640 |

**Total: 11,375,786 tokens · $8.7925** (43.2% of gold-standard cost)

---

## The cost–ranking-stability trade-off

| arm | total tokens | total cost | cost vs gold | stability | top-3 overlap | Kendall τ | same order | winner |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `all_high` | 11,375,912 | $20.3497 | 100.0% | 100.00% | 3/3 | 1.000 | yes | `thorough-32b` ✓ |
| `all_low` | 8,169,544 | $0.6975 | **3.4%** | **63.34%** | 2/3 | 0.200 | no | `balanced-13b` ✗ |
| `allocated` | 11,375,786 | $8.7925 | **43.2%** | **68.33%** | 2/3 | 0.400 | no | `balanced-13b` ✗ |

**The trade-off is not monotone in the useful direction.** Cutting cost by 97% costs
36.7 stability points; cutting cost by 57% costs 31.7. Beyond roughly half the budget,
**you are paying more without recovering the ranking** — the curve is flat where it
matters. This is the practical argument for a *scheduled* evaluation rather than a
uniformly cheap one: neither cheap nor half-cheap recovers the true top-3, so the
budget belongs on the tasks that actually discriminate between models.

### What "stability" means here

```
stability_pct = 100 × (0.5 × top3_membership_overlap_ratio + 0.5 × (kendall_tau_b + 1) / 2)
```

Membership overlap (does the arm's top-3 contain the gold-standard top-3) and
rank correlation (does the arm order them the same way) each contribute half.
The reference arm scores 100 by construction, so `stability_delta_pct` is the
drop from that. Computed on the **pooled** 3-round leaderboard.

**Per-round honesty check.** A pooled number is only meaningful if the rounds agree:

| arm | stability per round | min | max | spread | same top-3 order every round |
|---|---|---:|---:|---:|---|
| `all_high` | 100.00, 100.00, 100.00 | 100.00 | 100.00 | 0.00 | yes |
| `all_low` | 63.34, 63.34, 63.34 | 63.34 | 63.34 | 0.00 | no |
| `allocated` | 73.33, 58.34, 68.33 | 58.34 | 73.33 | **15.00** | no |

The two fixed arms are perfectly reproducible across rounds; the **allocated** arm
varies by 15 points. That variance is itself a finding: scheduling different tasks to
different tiers makes the leaderboard *less* repeatable, because each round re-draws a
different subset of tasks at high fidelity. The pooled −31.66% is the central estimate,
not a guarantee — a single round could plausibly report anywhere from −42% to −27%.

---

## Why the saving is a *price* effect, and why that matters

A surprising detail worth stating plainly:

| quantity | all_high | allocated | change |
|---|---:|---:|---:|
| tokens | 11,375,912 | 11,375,786 | **−0.00%** |
| cost | $20.3497 | $8.7925 | **−56.79%** |
| effective price | $1.79 / Mtok | $0.77 / Mtok | −56.8% |

**The allocated arm consumes essentially the same number of tokens and still costs 57%
less**, because a cheaper tier is priced at a cheaper model class. Two things follow:

1. **"Cheaper model" is not the same as "faster model."** An agent splits work into
   steps and re-sends its growing conversation each step, so most of the token volume is
   *reprocessing history*. Downgrading the model lowers the price per token but does not
   reduce how much the agent talks. Reporting only "tokens saved" would overstate a
   scheduling win that does not exist.
2. **The bias that reshuffles the leaderboard is a *capability* effect, not a price
   effect.** The tiers differ in agent steps and tool budget — that is what changes which
   model wins. The dollars differ for a separate reason. Conflating the two is exactly
   how an evaluation cost report ends up misleading its reader.

Per task category, the allocated arm concentrates spending where it discriminates:

| arm | category | tokens | token share | cost | cost share |
|---|---|---:|---:|---:|---:|
| `allocated` | gaia | 4,131,250 | 36.3% | $7.3750 | **83.9%** |
| `allocated` | math | 4,216,678 | 37.1% | $1.1583 | 13.2% |
| `allocated` | simpleqa | 3,027,858 | 26.6% | $0.2592 | 3.0% |

GAIA (multi-hop, tool-dependent) absorbs 84% of the budget on 36% of the tokens —
the allocator doing its job.

---

## How it works

### 1. The budget allocator

Reads the task manifest plus historical difficulty signals and assigns a **fidelity
tier** per task, *before* any evaluation runs. A tier bundles measurement knobs, and is
priced at the model class you would pay for to obtain that fidelity:

| tier | priced as | max steps | tools | grader |
|---|---|---|---:|---|---|
| `low` | `openai/gpt-4o-mini` | 2 | no exploration | keyword |
| `mid` | `openai/gpt-4o` | 4 | on | keyword + semantic |
| `high` | `anthropic/claude-sonnet-4-5` | 6 | on | strict semantic |

Three policies ship:

- **`by_category`** — static map: GAIA → `high`, Math → `mid`, SimpleQA → `low`.
- **`by_pass_rate`** — hard tasks (low historical pass rate) get more fidelity:
  `p < 0.35 → high`, `0.35 ≤ p < 0.65 → mid`, `p ≥ 0.65 → low`. Missing history
  falls back to the category prior. Thresholds are banded to the sampling error of a
  30-run history (half-width 0.15 ≈ 1.65·SE at *n*=30).
- **`greedy_budget`** — under an explicit token budget, greedily upgrades the tasks
  with the highest *ranking-risk per token*, where
  `risk = category_weight × 4·p·(1−p)`. Tasks near 50% pass rate discriminate most
  between models; tasks everyone passes or everyone fails discriminate almost nothing.
  The budget is a hard constraint.

The expected-token estimator is a single shared function and is honest about the
mechanism:

```
tokens(task, tier) = (600 + 300·steps + 120·steps·(steps−1)/2
                      + 400·tool_calls [if tools]) × category_multiplier
```

The `steps·(steps−1)/2` term is the growing-conversation cost — without it a greedy
allocator sinks the whole budget into escalating one task instead of buying breadth.

### 2. Per-task measurement, with `allocation_tier` on the span

Every evaluated `(task, candidate)` pair records its **allocated tier, allocated model,
actual tokens consumed, whether it passed, and latency**. The allocator's decision is
attached to SMOLTRACE's own OpenTelemetry span as the custom attribute
**`allocation_tier`**, alongside the GenAI semantic-convention attributes — so
allocation-aware analysis is possible without parsing side files:

```python
span.set_attribute("allocation_tier", tier)   # ours
span.set_attribute("llm.model_name", tier_api_model(tier))  # what it is priced as
span.set_attribute("llm.token_count.total", prompt + completion)
```

Token and cost figures are **not** our own bookkeeping — they come from
`smoltrace.core.extract_traces` and are priced by genai-otel's cost calculator against
the tier's real model id. That is why the table above is called a *price* effect: the
numbers are SMOLTRACE's.

### 3. Reuse, not rewrite

SMOLTRACE is untouched. The layer drives its public entry points:

| capability | SMOLTRACE API used |
|---|---|
| OTEL setup | `smoltrace.otel.setup_inmemory_otel` |
| agent construction | `smoltrace.core.initialize_agent` |
| single-task evaluation | `smoltrace.core.evaluate_single_test` |
| trace extraction | `smoltrace.core.extract_traces` |
| metrics extraction | `smoltrace.core.extract_metrics` |
| leaderboard row | `smoltrace.utils.compute_leaderboard_row` |

The one thing SMOLTRACE cannot express — a **different model per task within one run** —
is what the scheduler adds: it injects a per-task model instance into a cached agent.

---

## Quickstart

### Install

Requires **Python 3.11+**. No API keys, GPU, or network access to a model provider is
needed — the measurement backend is simulated and runs locally.

```bash
git clone https://github.com/zzc1-nz/smoltrace-budget.git
cd smoltrace-budget
pip install -r requirements.txt
```

> **Note on this repository's layout.** During development the dependencies were
> installed with `pip --target .venvlibs/` so that `smoltrace` could be pinned without
> disturbing a shared interpreter. That directory is git-ignored. If you install with a
> normal `pip install -r requirements.txt`, everything works; the bootstrap in
> `smoltrace_budget/__init__.py` simply puts `.venvlibs` on `sys.path` *if it exists*.

### Run

```bash
python scripts/reproduce.py --quick     # 3 tasks, 1 round — validate the pipeline
python scripts/reproduce.py             # full 132 × 5 × 3 × 3 matrix (~6 minutes)
```

Outputs:

```
artifacts/records/<arm>_r<round>.json       per-(task, model) measurements
artifacts/allocations/<arm>_r<round>.json   the allocator's decisions
artifacts/leaderboard/<arm>_r<round>.json    SMOLTRACE-computed leaderboard rows
artifacts/report/cost_report.json            the full report
artifacts/report/summary.md                  the human-readable version
```

Or step by step:

```bash
python scripts/00_fetch_tasks.py              # freeze the 132-task benchmark
python scripts/01_validate_tasks.py           # check schema against SMOLTRACE
python scripts/02_build_history.py            # build the difficulty prior
python scripts/03_inspect_allocations.py      # see what each policy decides
python scripts/04_run_evaluation.py --rounds 3 --policy by_category
python scripts/05_analyze.py                  # write the cost report
```

Test suite:

```bash
python run_tests.py -q      # 177 tests
```

`python run_tests.py` (not `python -m pytest`) is the supported entry point: it applies
the bootstrap that puts the project and its pinned dependencies on `sys.path`.

### Verify the published numbers

```bash
python scripts/06_audit_readme.py
```

Re-derives every figure quoted in this README from `artifacts/report/cost_report.json`
and the raw per-round records, and fails if any of them disagree. Currently **68/68
checks pass**.

---

## A cautionary finding: evaluation cost is easy to underestimate

The original design for this project budgeted **1.5–2.0M tokens** for the experiment,
reasoned as *132 tasks × 3 rounds × ~4,000 tokens*. That is correct — **for one model**.
Producing three comparable leaderboards requires evaluating the *same pool* of models
in every arm, i.e. 132 × **5** × 3 × 3 = **5,940 evaluations**, which measured
**30.9M tokens** — about **20× the estimate**.

The underestimation is not a factor-of-20 arithmetic slip; it is a modelling error of the
same kind this project is about. The cost of an evaluation is not
`tasks × per-task tokens`. It is:

```
evaluation cost = tasks × candidates × arms × rounds × per-task tokens
```

and every one of those factors is a choice you can make badly. A cost model that
silently omits the candidate dimension will produce a budget that is wrong by the size
of the model pool. This is why the allocator takes a *token budget* as an explicit input
rather than discovering the cost afterwards.

---

## Reproducing the numbers

Every figure above is deterministic given the seed: the simulated backend derives each
draw from `sha256(seed, round, stage, tier, candidate, category, task)` and never touches
global RNG state. A rerun reproduces the same leaderboards exactly. The reported
tokens/costs are real — they flow through SMOLTRACE's OTEL spans and genai-otel pricing.

**On the measurement backend.** Model calls go through a calibrated `smolagents.Model`
implementing the documented bias model:

```
p_measured = p_true × (1 − 0.75 × shortfall),   shortfall = clip(1 − tier_capacity / needed, 0, 1)
needed   = 1 + (3.0·tool_reliance + 2.0·deliberation) × category_process_need
```

This keeps the study reproducible and free to run while every other part of the pipeline
is genuinely exercised: the real agent loop, real OTEL spans, real trace extraction, real
cost calculation, real leaderboard aggregation. Swapping in `LiteLLMModel` is a one-line
provider change; the scheduler and analysis are backend-agnostic. The *specific*
percentages here describe this calibrated pool — the *mechanism* (cheap evaluation
rewards models that are cheap to evaluate) is the transferable result.

---

## Repository layout

```
smoltrace_budget/
  contracts.py       shared data contracts (tiers, records, arms)
  allocator.py       3 budget-allocation policies + expected-token model
  models.py          the 5-model candidate pool with latent abilities
  history.py         historical difficulty prior
  sim_backend.py     calibrated smolagents.Model + OTEL span emission
  runner.py          3 arms × N rounds over SMOLTRACE's pipeline
  analysis.py        leaderboards, cost, rank stability, cost report
  report_render.py   Markdown report rendering
scripts/             numbered pipeline steps + reproducibility probes
tests/               177 tests
artifacts/           measurements and the generated report
```

## License

Apache-2.0, matching SMOLTRACE.
