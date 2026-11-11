"""CLI: run the full three-arm x N-round evaluation matrix.

Usage (from the ``smoltrace-budget`` directory)::

    python scripts/04_run_evaluation.py
    python scripts/04_run_evaluation.py --rounds 3 --policy by_category
    python scripts/04_run_evaluation.py --limit 12 --rounds 1        # smoke run
    python scripts/04_run_evaluation.py --policy greedy_budget --token-budget 1500000
    python scripts/04_run_evaluation.py --force                     # re-run everything

The default configuration is the published one: the frozen 132-task manifest,
the full five-model candidate pool, all three arms, three rounds. Artifacts land
in ``artifacts/`` (see ``smoltrace_budget.runner`` for the exact layout).

Rounds are resumable: a round whose artifacts already exist is loaded instead
of re-evaluated, so an interrupted run continues where it stopped. Pass
``--force`` to redo them.
"""

from _common import PROJECT_ROOT  # noqa: F401  (bootstrap + sys.path first)

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from smoltrace_budget import ARTIFACTS_DIR
from smoltrace_budget.contracts import ARMS, REFERENCE_ARM, TIER_LABELS
from smoltrace_budget.models import CANDIDATES
from smoltrace_budget.runner import (
    DEFAULT_AGENT_TYPE,
    experiment_totals,
    load_round_artifacts,
    run_experiment,
)
from smoltrace_budget.tasks import load_tasks, summarize_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SMOLTRACE cost-aware evaluation matrix (3 arms x N rounds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="path to the task manifest JSONL (default: the frozen data/tasks/benchmark_v1.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N tasks (smoke runs; never use for published numbers)",
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="number of independent repetitions per arm"
    )
    parser.add_argument(
        "--policy",
        default="by_category",
        choices=["by_category", "by_pass_rate", "greedy_budget"],
        help="tier-assignment policy used by the allocated arm",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="token budget for --policy greedy_budget (sum of expected_tokens stays under it)",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated subset of arms to run",
    )
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument(
        "--agent-type",
        default=DEFAULT_AGENT_TYPE,
        choices=["tool", "code"],
        help="smolagents agent loop used for every task",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"artifacts root (default: {ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-evaluate rounds whose artifacts already exist"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print only the per-round summary lines"
    )
    return parser


def _parse_arms(raw: str) -> List[str]:
    arms = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; expected a subset of {ARMS}")
    if not arms:
        raise SystemExit(f"--arms is empty; expected a subset of {ARMS}")
    return arms


def _print_table(experiment: dict) -> None:
    print("\n" + "=" * 78)
    print("PER-ARM TOTALS")
    print("=" * 78)
    print(f"{'arm':<12}{'records':>9}{'tokens':>12}{'cost_usd':>13}{'ms':>12}")
    for arm, bucket in sorted(experiment["totals"]["per_arm"].items()):
        wall = sum(
            r["wall_ms"] for r in experiment["arms"] if r["arm"] == arm
        )
        print(
            f"{arm:<12}{bucket['n_records']:>9}{bucket['total_tokens']:>12}"
            f"{bucket['total_cost_usd']:>13.6f}{wall:>12.0f}"
        )
    totals = experiment["totals"]
    print("-" * 78)
    print(
        f"{'ALL':<12}{totals['n_records']:>9}{totals['total_tokens']:>12}"
        f"{totals['total_cost_usd']:>13.6f}{experiment['wall_ms']:>12.0f}"
    )

    # The headline comparison the report quotes.
    per_arm = experiment["totals"]["per_arm"]
    reference = per_arm.get(REFERENCE_ARM)
    allocated = per_arm.get("allocated")
    if reference and allocated and reference["total_tokens"]:
        saved = 100.0 * (reference["total_tokens"] - allocated["total_tokens"]) / reference["total_tokens"]
        print(
            f"\nallocated vs {REFERENCE_ARM}: {saved:.1f}% fewer tokens "
            f"({reference['total_tokens']} -> {allocated['total_tokens']})"
        )


def _print_ranking(experiment: dict) -> None:
    """Final-round ranking per arm — the shape the analysis layer consumes."""
    final: dict = {}
    for arm_result in experiment["arms"]:
        current = final.get(arm_result["arm"])
        if current is None or arm_result["round_index"] >= current["round_index"]:
            final[arm_result["arm"]] = arm_result
    if not final:
        return
    print("\n" + "=" * 78)
    print(f"RANKING (final round)")
    print("=" * 78)
    for arm in sorted(final):
        arm_result = final[arm]
        print(f"\n{arm}  (round {arm_result['round_index']})")
        print(f"  {'rank':<6}{'model':<22}{'pass_rate':>11}{'tokens':>12}{'cost_usd':>13}")
        for row in arm_result["ranking"]:
            print(
                f"  {row['rank']:<6}{row['model']:<22}{row['pass_rate']:>10.2f}%"
                f"{row['total_tokens']:>12}{row['total_cost_usd']:>13.6f}"
            )


def _print_tiers(experiment: dict) -> None:
    allocations = [
        a for arm_result in experiment["arms"] if arm_result["arm"] == "allocated"
        for a in arm_result["allocations"]
    ]
    if not allocations:
        return
    counts: dict = {}
    for allocation in allocations:
        counts[allocation["tier"]] = counts.get(allocation["tier"], 0) + 1
    rounds = len({r["round_index"] for r in experiment["arms"] if r["arm"] == "allocated"})
    print("\nallocated-arm tier distribution (per round):")
    for tier in TIER_LABELS:
        if tier in counts:
            print(f"  {tier:<6} ({TIER_LABELS[tier]:<14}) {counts[tier] / max(rounds, 1):>7.1f} tasks")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    arms = _parse_arms(args.arms)
    tasks = load_tasks(args.tasks, limit=args.limit)
    if not tasks:
        print("no tasks loaded; run scripts/00_fetch_tasks.py first", file=sys.stderr)
        return 2

    summary = summarize_tasks(tasks)
    print("=" * 78)
    print("SMOLTRACE cost-aware evaluation")
    print("=" * 78)
    print(f"  tasks        : {summary['n_tasks']}  {summary['by_category']}")
    print(f"  candidates   : {len(CANDIDATES)}  {[c.model_id for c in CANDIDATES]}")
    print(f"  arms         : {arms}")
    print(f"  rounds       : {args.rounds}")
    print(f"  policy       : {args.policy}" + (f" (budget {args.token_budget:,})" if args.token_budget else ""))
    print(f"  agent loop   : {args.agent_type}")
    print(f"  artifacts    : {args.out_dir or ARTIFACTS_DIR}")
    print(
        "  evaluate calls: "
        f"{summary['n_tasks'] * len(CANDIDATES) * len(arms) * args.rounds}"
    )

    started = time.perf_counter()
    experiment = run_experiment(
        tasks,
        CANDIDATES,
        arms=arms,
        n_rounds=args.rounds,
        policy=args.policy,
        token_budget=args.token_budget,
        seed=args.seed,
        out_dir=args.out_dir,
        force=args.force,
        progress=not args.quiet,
        quiet=True,
        agent_type=args.agent_type,
    )
    wall = time.perf_counter() - started

    _print_table(experiment)
    _print_ranking(experiment)
    _print_tiers(experiment)

    print("\nartifacts written under", experiment["paths"]["experiment"])
    loaded = load_round_artifacts(args.out_dir)
    print(
        "re-read from disk:",
        {arm: sorted(rounds) for arm, rounds in sorted(loaded.items())},
    )
    print(f"\ntotal wall clock: {wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
