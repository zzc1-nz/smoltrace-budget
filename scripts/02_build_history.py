"""Build the historical difficulty signal used by the budget allocator.

Writes data/history/benchmark_v1_history.json — a per-task pass-rate prior in
the shape BudgetAllocator expects:

    {"<task_id>": {"pass_rate": 0.42, "n": 5, "category": "gaia"}, ...}

In a real deployment this file is produced from previous SMOLTRACE runs (the
leaderboard's `smoltrace-results-*` datasets contain per-task outcomes). Here it
is synthesised from the candidate pool's latent ability with deterministic noise,
so the allocator gets a realistic *prior* rather than an oracle.
"""

from _common import DATA_DIR, write_json
from smoltrace_budget.history import CATEGORY_DIFFICULTY, build_history
from smoltrace_budget.models import CANDIDATES
from smoltrace_budget.tasks import load_tasks

OUT = DATA_DIR / "history" / "benchmark_v1_history.json"


def main() -> None:
    tasks = load_tasks()
    history = build_history(tasks, CANDIDATES, seed=0)
    write_json(OUT, history)

    print(f"wrote {len(history)} task priors -> {OUT}")
    print(f"category difficulty offsets: {CATEGORY_DIFFICULTY}")
    print()

    by_category: dict[str, list[float]] = {}
    for task_id, entry in history.items():
        by_category.setdefault(entry["category"], []).append(entry["pass_rate"])

    print(f"{'category':<12} {'n':>4} {'mean prior':>11} {'min':>7} {'max':>7}")
    print("-" * 46)
    for category in sorted(by_category):
        values = by_category[category]
        print(
            f"{category:<12} {len(values):>4} {sum(values) / len(values):>11.3f} "
            f"{min(values):>7.3f} {max(values):>7.3f}"
        )
    print()
    hardest = sorted(history.items(), key=lambda kv: kv[1]["pass_rate"])[:5]
    easiest = sorted(history.items(), key=lambda kv: kv[1]["pass_rate"])[-5:]
    print("hardest 5 (lowest prior pass rate):")
    for task_id, entry in hardest:
        print(f"  {task_id:<14} {entry['pass_rate']:.3f}  ({entry['category']})")
    print("easiest 5 (highest prior pass rate):")
    for task_id, entry in easiest:
        print(f"  {task_id:<14} {entry['pass_rate']:.3f}  ({entry['category']})")


if __name__ == "__main__":
    main()
