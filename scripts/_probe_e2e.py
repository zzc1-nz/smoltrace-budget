"""End-to-end probe: does the backend produce real passes and real tokens?

Runs a handful of tasks through SMOLTRACE's real pipeline for each tier and
prints per-(task, candidate) outcomes, so the calibration can be inspected
before committing to the full 5,940-evaluation matrix.
"""

from _common import PROJECT_ROOT  # noqa: F401  (bootstrap first)

import json
from collections import Counter, defaultdict

import smoltrace_budget  # noqa: F401
from smoltrace_budget.models import CANDIDATES
from smoltrace_budget.runner import run_arm
from smoltrace_budget.tasks import load_tasks


def summarize(arm: str, records) -> None:
    print(f"\n=== arm: {arm} ({len(records)} records) ===")
    by_model = defaultdict(lambda: {"passed": 0, "n": 0, "tokens": 0, "cost": 0.0, "tiers": Counter()})
    for record in records:
        bucket = by_model[record["candidate_model"]]
        bucket["n"] += 1
        bucket["passed"] += 1 if record["passed"] else 0
        bucket["tokens"] += record["total_tokens"]
        bucket["cost"] += record["cost_usd"]
        bucket["tiers"][record["allocated_tier"]] += 1

    print(f"  {'model':<16} {'pass':>6} {'rate':>7} {'tokens':>9} {'cost':>10}  tiers")
    for model, bucket in sorted(by_model.items(), key=lambda kv: -kv[1]["passed"] / max(1, kv[1]["n"])):
        rate = bucket["passed"] / bucket["n"] * 100
        print(
            f"  {model:<16} {bucket['passed']:>3}/{bucket['n']:<3} {rate:>6.1f}% "
            f"{bucket['tokens']:>9,} ${bucket['cost']:>9.5f}  {dict(bucket['tiers'])}"
        )
    total_tokens = sum(r["total_tokens"] for r in records)
    total_cost = sum(r["cost_usd"] for r in records)
    print(f"  TOTAL tokens={total_tokens:,} cost=${total_cost:.5f}")
    return total_tokens, total_cost


def main() -> None:
    tasks = load_tasks(limit=6)
    print(f"tasks: {[t.id + '/' + t.category for t in tasks]}")
    print(f"candidates: {[c.model_id for c in CANDIDATES]}")

    grand_tokens = 0
    for arm in ("all_high", "all_low", "allocated"):
        result = run_arm(arm, tasks, CANDIDATES, round_index=0, policy="by_category", seed=0)
        records = result.records if hasattr(result, "records") else result["records"]
        tokens, _ = summarize(arm, records)
        grand_tokens += tokens

    print()
    print(f"grand total tokens for this probe: {grand_tokens:,}")
    n_pairs = len(tasks) * len(CANDIDATES)
    print(f"per (task,candidate) pair, 3 arms: {grand_tokens / (n_pairs * 3):,.0f} tokens")
    print(
        f"extrapolated to 132 tasks x 5 candidates x 3 arms x 3 rounds: "
        f"{grand_tokens / (n_pairs * 3) * 132 * 5 * 3 * 3 / 1e6:.2f}M tokens"
    )


if __name__ == "__main__":
    main()
