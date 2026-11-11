"""Validate the frozen 132-task manifest against SMOLTRACE's own expectations.

Checks that the snapshot we evaluate is shape-compatible with what
``smoltrace.core.load_test_cases_from_hf`` returns, so ``evaluate_single_test``
cannot fail on a missing field at run time.
"""

from _common import PROJECT_ROOT  # noqa: F401  (bootstrap first)

import json
from collections import Counter

import smoltrace_budget  # noqa: F401
from smoltrace_budget.contracts import TaskSpec
from smoltrace_budget.tasks import DEFAULT_MANIFEST, load_tasks, summarize_tasks

# Fields smoltrace.core.evaluate_single_test / _filter_tests read directly.
REQUIRED = ("id", "prompt", "difficulty", "agent_type")


def main() -> None:
    tasks = load_tasks()
    print(f"loaded {len(tasks)} tasks from {DEFAULT_MANIFEST}")
    print(json.dumps(summarize_tasks(tasks), indent=2))

    problems = []
    for task in tasks:
        payload = task.to_dict()
        for field in REQUIRED:
            if not payload.get(field):
                problems.append(f"{task.id}: missing {field}")
        if not task.expected_keywords:
            problems.append(f"{task.id}: no expected keywords (grader would auto-pass)")
        if task.expected_tool_calls < 1:
            problems.append(f"{task.id}: expected_tool_calls < 1")

    print()
    print("--- keyword sanity ---")
    keyword_counts = Counter(len(t.expected_keywords) for t in tasks)
    print("keywords per task histogram:", dict(sorted(keyword_counts.items())))
    empties = [t.id for t in tasks if any(not k.strip() for k in t.expected_keywords)]
    print("tasks with blank keyword strings:", empties or "none")

    print()
    print("--- agent_type coverage (SMOLTRACE runs tool and code separately) ---")
    print(dict(Counter(t.agent_type for t in tasks)))
    for agent_type in ("tool", "code"):
        runnable = [t for t in tasks if t.agent_type in (agent_type, "both")]
        print(f"  {agent_type}: {len(runnable)} runnable tasks")

    print()
    print("--- SMOLTRACE-compatible dict shape ---")
    first = tasks[0].to_dict()
    print("keys:", sorted(first))
    print("sample:", json.dumps(
        {k: (str(v)[:70] + "...") if isinstance(v, str) and len(str(v)) > 70 else v
         for k, v in first.items()},
        indent=2,
    ))

    print()
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for problem in problems[:20]:
            print("  -", problem)
        raise SystemExit(1)
    print("OK: manifest is shape-compatible with SMOLTRACE's evaluator")


if __name__ == "__main__":
    main()
