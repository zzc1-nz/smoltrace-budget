"""Inspect what the budget allocator decides for the real 132-task manifest.

This is the human-facing check on the allocator: it prints the tier histogram,
the resulting token split, and a few example decisions with their reasons.
"""

from _common import PROJECT_ROOT  # noqa: F401  (bootstrap first)

from smoltrace_budget.allocator import estimate_task_tokens, make_allocator
from smoltrace_budget.tasks import load_tasks

HISTORY = PROJECT_ROOT / "data" / "history" / "benchmark_v1_history.json"


def report(policy: str, tasks, history, **kwargs) -> None:
    allocator = make_allocator(policy, history=history, **kwargs)
    allocations = allocator.allocate(tasks)
    tokens = sum(a.expected_tokens for a in allocations)
    print()
    print(f"=== policy: {policy} " + " ".join(f"{k}={v}" for k, v in kwargs.items()) + " ===")
    counts: dict[str, int] = {}
    cost: dict[str, float] = {}
    for alloc in allocations:
        counts[alloc.tier] = counts.get(alloc.tier, 0) + 1
        cost[alloc.tier] = cost.get(alloc.tier, 0.0) + alloc.expected_tokens
    print(f"  allocations: {len(allocations)}")
    for tier in ("low", "mid", "high"):
        n = counts.get(tier, 0)
        share = n / len(allocations) * 100 if allocations else 0
        print(f"    {tier:<5} {n:>4} tasks ({share:5.1f}%)  ~{cost.get(tier, 0):>12,.0f} expected tokens")
    print(f"  total expected tokens: {tokens:,.0f}")
    return allocations


def main() -> None:
    import json

    tasks = load_tasks()
    history = json.loads(HISTORY.read_text(encoding="utf-8"))

    baseline_high = sum(estimate_task_tokens(t, "high") for t in tasks)
    baseline_low = sum(estimate_task_tokens(t, "low") for t in tasks)
    print(f"manifest: {len(tasks)} tasks")
    print(f"  all-high expected tokens: {baseline_high:,.0f}")
    print(f"  all-low  expected tokens: {baseline_low:,.0f}")

    for policy, kwargs in (
        ("by_category", {}),
        ("by_pass_rate", {}),
        ("greedy_budget", {"token_budget": int(baseline_high * 0.45)}),
        ("greedy_budget", {"token_budget": int(baseline_high * 0.25)}),
    ):
        allocations = report(policy, tasks, history, **kwargs)
        if allocations and policy == "by_category":
            print("  sample decisions:")
            seen = set()
            for alloc in allocations:
                task = next(t for t in tasks if t.id == alloc.task_id)
                if task.category in seen:
                    continue
                seen.add(task.category)
                print(f"    {alloc.task_id:<14} {task.category:<10} -> {alloc.tier:<5}  {alloc.reason}")


if __name__ == "__main__":
    main()
