"""Fetch the SMOLTRACE comprehensive benchmark (132 tasks) into data/tasks/.

Downloads the parquet file straight from the Hub (the datasets-server API is not
reachable from this network) and freezes it as JSONL so every later step is
reproducible even when the network is not.

Usage: python scripts/00_fetch_tasks.py
"""

from __future__ import annotations

import io
from collections import Counter

from _common import DATA_DIR, fetch_json, http_get, write_json, write_jsonl

REPO = "kshitijthakkar/smoltrace-benchmark-v1"
SPLIT = "train"
OUT = DATA_DIR / "tasks" / "benchmark_v1.jsonl"
META = DATA_DIR / "tasks" / "benchmark_v1.meta.json"
FIELDS = [
    "id",
    "prompt",
    "difficulty",
    "agent_type",
    "expected_tool",
    "expected_tool_calls",
    "expected_keywords",
    "category",
    "source",
    "true_answer",
    "true_reasoning",
]


def normalize(value):
    """Parquet round-trips numpy scalars; convert them into plain JSON types."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    for attr in ("item",):  # numpy scalar
        if hasattr(value, attr) and not isinstance(value, (str, bytes)):
            try:
                return normalize(value.item())
            except (TypeError, ValueError):
                break
    return value


def main() -> None:
    info = fetch_json(f"https://huggingface.co/api/datasets/{REPO}?blobs=true")
    files = [s["rfilename"] for s in info.get("siblings", []) if s["rfilename"].endswith(".parquet")]
    print(f"{REPO}: {len(files)} parquet file(s)")
    for name in files:
        print("  ", name)

    import pyarrow.parquet as pq

    tables = []
    for name in files:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{name}"
        print(f"downloading {url}")
        blob = http_get(url, timeout=180)
        print(f"  {len(blob):,} bytes")
        tables.append(pq.read_table(io.BytesIO(blob)))

    import pyarrow as pa

    table = pa.concat_tables(tables)
    print(f"rows: {table.num_rows}, columns: {table.column_names}")

    tasks = []
    for record in table.to_pylist():
        task = {field: normalize(record.get(field)) for field in FIELDS}
        task["expected_keywords"] = [str(k) for k in (task.get("expected_keywords") or [])]
        task["expected_tool_calls"] = task.get("expected_tool_calls") or 1
        task["difficulty"] = task.get("difficulty") or "unknown"
        task["agent_type"] = task.get("agent_type") or "both"
        task["category"] = task.get("category") or "unknown"
        tasks.append(task)

    write_jsonl(OUT, tasks)

    meta = {
        "dataset": REPO,
        "split": SPLIT,
        "n_tasks": len(tasks),
        "fields": FIELDS,
        "by_category": dict(Counter(t["category"] for t in tasks)),
        "by_difficulty": dict(Counter(t["difficulty"] for t in tasks)),
        "by_agent_type": dict(Counter(t["agent_type"] for t in tasks)),
        "by_expected_tool": dict(Counter(str(t["expected_tool"]) for t in tasks)),
        "by_source": dict(Counter(str(t["source"]) for t in tasks)),
        "keywords_with": sum(1 for t in tasks if t["expected_keywords"]),
        "keywords_without": sum(1 for t in tasks if not t["expected_keywords"]),
        "prompt_chars": {
            "min": min(len(t["prompt"]) for t in tasks),
            "max": max(len(t["prompt"]) for t in tasks),
            "mean": round(sum(len(t["prompt"]) for t in tasks) / len(tasks), 1),
        },
    }
    write_json(META, meta)

    import json

    print()
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print()
    print(f"wrote {len(tasks)} tasks -> {OUT}")


if __name__ == "__main__":
    main()
