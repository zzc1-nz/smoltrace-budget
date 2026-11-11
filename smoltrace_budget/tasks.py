"""Task-manifest loading for the benchmark.

The manifest is the frozen JSONL snapshot produced by
``scripts/00_fetch_tasks.py`` from the SMOLTRACE comprehensive benchmark
(``kshitijthakkar/smoltrace-benchmark-v1`` — 132 tasks across GAIA, Math and
SimpleQA). Freezing it locally keeps every experiment reproducible and means a
flaky network cannot change a published number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import TaskSpec

__all__ = ["DEFAULT_MANIFEST", "load_tasks", "load_manifest_meta", "summarize_tasks"]

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "data" / "tasks" / "benchmark_v1.jsonl"
DEFAULT_META = Path(__file__).resolve().parent.parent / "data" / "tasks" / "benchmark_v1.meta.json"


def load_tasks(
    path: Optional[Path | str] = None,
    *,
    limit: Optional[int] = None,
    categories: Optional[List[str]] = None,
    agent_type: Optional[str] = None,
) -> List[TaskSpec]:
    """Load the frozen benchmark manifest.

    Args:
        path: manifest JSONL; defaults to the bundled 132-task snapshot.
        limit: keep only the first N tasks (after filtering). Useful for smoke
            runs; never use it for numbers that go into the report.
        categories: keep only these categories (``gaia`` / ``math`` / ``simpleqa``).
        agent_type: keep only tasks runnable by this agent type (``tool``/``code``).
    """
    manifest = Path(path) if path else DEFAULT_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(
            f"task manifest not found: {manifest}\n"
            "Run `python scripts/00_fetch_tasks.py` to fetch and freeze it."
        )

    tasks: List[TaskSpec] = []
    with open(manifest, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tasks.append(TaskSpec.from_dict(json.loads(line)))

    if categories:
        wanted = {c.lower() for c in categories}
        tasks = [t for t in tasks if t.category.lower() in wanted]

    if agent_type:
        tasks = [t for t in tasks if t.agent_type in (agent_type, "both")]

    if limit is not None and limit > 0:
        tasks = tasks[:limit]

    return tasks


def load_manifest_meta(path: Optional[Path | str] = None) -> Dict[str, Any]:
    meta_path = Path(path) if path else DEFAULT_META
    if not meta_path.is_file():
        return {}
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_tasks(tasks: List[TaskSpec]) -> Dict[str, Any]:
    """Category / difficulty / agent-type counts for report headers."""
    by_category: Dict[str, int] = {}
    by_difficulty: Dict[str, int] = {}
    by_agent_type: Dict[str, int] = {}
    for task in tasks:
        by_category[task.category] = by_category.get(task.category, 0) + 1
        by_difficulty[task.difficulty] = by_difficulty.get(task.difficulty, 0) + 1
        by_agent_type[task.agent_type] = by_agent_type.get(task.agent_type, 0) + 1
    return {
        "n_tasks": len(tasks),
        "by_category": dict(sorted(by_category.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_agent_type": dict(sorted(by_agent_type.items())),
    }
