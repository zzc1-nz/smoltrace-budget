"""Build the analysis report from the runner's artifacts.

Reads ``artifacts/records/<arm>_r<round>.json`` and writes:

    artifacts/report/cost_report.json   full JSON document (everything the
                                        analysis layer computed)
    artifacts/report/summary.md         human-readable report: three leaderboard
                                        tables, the cost table, the category
                                        breakdown, rank stability, the cheap-arm
                                        finding and the X%/Y% trade-off

Usage:
    python scripts/05_analyze.py
    python scripts/05_analyze.py --records scratch/records --out scratch/report

Exits non-zero with a clear message (never a stack trace) when the artifacts are
not there yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import ARTIFACTS_DIR, PROJECT_ROOT, ensure_dir, write_json

from smoltrace_budget.analysis import (
    build_cost_report,
    flatten_artifacts,
    load_artifacts,
    render_summary_markdown,
)
from smoltrace_budget.contracts import ARMS

DEFAULT_RECORDS = ARTIFACTS_DIR / "records"
DEFAULT_OUT = ARTIFACTS_DIR / "report"


def _fail(message: str, hint: str | None = None) -> int:
    """Print a diagnosable error and return the process exit code."""
    print(f"ERROR: {message}", file=sys.stderr)
    if hint:
        print(f"       {hint}", file=sys.stderr)
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse the 3-arm evaluation artifacts and write the cost report."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS,
        help=f"directory of <arm>_r<round>.json files (default: {DEFAULT_RECORDS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"report output directory (default: {DEFAULT_OUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records_dir: Path = args.records
    out_dir: Path = args.out

    # --- load -------------------------------------------------------------
    if not records_dir.is_dir():
        return _fail(
            f"records directory not found: {records_dir}",
            "Run the evaluation first: python scripts/04_run_experiment.py",
        )

    artifacts = load_artifacts(records_dir)
    if not artifacts:
        return _fail(
            f"no record artifacts found in {records_dir}",
            "Expected files named <arm>_r<round>.json (for example "
            f"all_high_r0.json). Run the evaluation first: {ARMS} x rounds.",
        )

    total_records = sum(len(rows) for per_arm in artifacts.values() for rows in per_arm.values())
    if total_records == 0:
        return _fail(
            f"record files in {records_dir} are all empty",
            "The evaluation produced no TaskRecords; check the runner log.",
        )

    declared = [arm for arm in ARMS if arm in artifacts]
    missing = [arm for arm in ARMS if arm not in artifacts]
    print(f"Loaded {total_records} records from {records_dir}")
    print(f"  arms found : {', '.join(declared)}")
    if missing:
        print(f"  arms missing: {', '.join(missing)} (report will be partial)")
    for arm in sorted(artifacts):
        rounds = ", ".join(
            f"r{round_index}({len(rows)})" for round_index, rows in sorted(artifacts[arm].items())
        )
        print(f"  {arm:10s} {rounds}")

    # --- analyse ----------------------------------------------------------
    records_by_arm = flatten_artifacts(artifacts)
    report = build_cost_report(records_by_arm)
    report["source"] = {
        "records_dir": str(records_dir.relative_to(PROJECT_ROOT))
        if records_dir.is_relative_to(PROJECT_ROOT)
        else str(records_dir),
        "n_record_files": sum(len(per_arm) for per_arm in artifacts.values()),
        "rounds": sorted({r for per_arm in artifacts.values() for r in per_arm}),
        "arms_present": list(report["arms"]),
        "arms_missing": missing,
    }

    # --- write ------------------------------------------------------------
    ensure_dir(out_dir)
    json_path = Path(write_json(out_dir / "cost_report.json", report))
    markdown_path = out_dir / "summary.md"
    markdown_path.write_text(render_summary_markdown(report), encoding="utf-8")

    tradeoff = report["cost_rank_tradeoff"]
    hallucination = report["cheap_arm_hallucination"]
    print("")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print("")
    print(f"Headline: {report['headline']}")
    cost_reduction = tradeoff["cost_reduction_pct"]
    token_reduction = tradeoff["token_reduction_pct"]
    print(
        "  cost reduction      : "
        + ("n/a (arm missing)" if cost_reduction is None else f"{cost_reduction:.2f}%")
    )
    print(
        "  token reduction     : "
        + ("n/a (arm missing)" if token_reduction is None else f"{token_reduction:.2f}%")
    )
    print(f"  rank stability      : {tradeoff['rank_stability_change_pct']:+.2f}%")
    if hallucination["wrongly_crowned"]:
        print(f"  wrongly crowned     : {hallucination['wrongly_crowned']}")
    else:
        print("  wrongly crowned     : none (cheap and gold arms agree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
