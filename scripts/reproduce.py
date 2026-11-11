#!/usr/bin/env python
"""Reproduce the whole study end to end.

    python scripts/reproduce.py            # full 132-task x 5-model x 3-arm x 3-round run
    python scripts/reproduce.py --quick    # small smoke configuration
    python scripts/reproduce.py --force    # ignore existing artifacts and re-run

Steps, in order:
  1. freeze the benchmark manifest            (scripts/00_fetch_tasks.py)
  2. validate it against SMOLTRACE's schema   (scripts/01_validate_tasks.py)
  3. build the historical difficulty prior    (scripts/02_build_history.py)
  4. run the three evaluation arms x rounds   (scripts/04_run_evaluation.py)
  5. analyse and write the cost report        (scripts/05_analyze.py)

Every step is idempotent: re-running skips work whose output already exists
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def run_step(label: str, args: list[str]) -> None:
    print()
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)
    started = time.time()
    env = dict(os.environ)
    # The report is written with non-ASCII typography; make sure the child's
    # stdout encoding can carry it on a Chinese-locale Windows console.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run([sys.executable, *args], cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise SystemExit(f"step failed ({completed.returncode}): {label}")
    print(f"  -> {label} finished in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small smoke configuration (3 tasks, 1 round)")
    parser.add_argument("--force", action="store_true", help="ignore existing artifacts and re-run")
    parser.add_argument("--rounds", type=int, default=3, help="rounds per arm (default 3)")
    parser.add_argument("--limit", type=int, default=None, help="limit number of tasks")
    parser.add_argument("--policy", default="by_category", help="allocator policy for the allocated arm")
    parser.add_argument("--token-budget", type=int, default=None, help="token budget for greedy_budget policy")
    parser.add_argument("--out-dir", default=None, help="artifacts root (default: artifacts/)")
    parser.add_argument("--skip-fetch", action="store_true", help="use the already-frozen manifest")
    args = parser.parse_args()

    limit = args.limit
    rounds = args.rounds
    if args.quick:
        limit = limit or 3
        rounds = 1

    if not args.skip_fetch:
        run_step("1/5  freeze benchmark manifest", [str(SCRIPTS / "00_fetch_tasks.py")])
    run_step("2/5  validate manifest", [str(SCRIPTS / "01_validate_tasks.py")])
    run_step("3/5  build historical difficulty prior", [str(SCRIPTS / "02_build_history.py")])

    eval_args = [str(SCRIPTS / "04_run_evaluation.py"), "--rounds", str(rounds), "--policy", args.policy]
    if limit:
        eval_args += ["--limit", str(limit)]
    if args.token_budget:
        eval_args += ["--token-budget", str(args.token_budget)]
    if args.out_dir:
        eval_args += ["--out-dir", str(args.out_dir)]
    if args.force:
        eval_args.append("--force")
    run_step("4/5  run evaluation arms", eval_args)

    analyze_args = [str(SCRIPTS / "05_analyze.py")]
    if args.out_dir:
        analyze_args += ["--records", str(Path(args.out_dir) / "records")]
        analyze_args += ["--out", str(Path(args.out_dir) / "report")]
    run_step("5/5  analyse and write cost report", analyze_args)

    report_dir = Path(args.out_dir) / "report" if args.out_dir else PROJECT_ROOT / "artifacts" / "report"
    print()
    print("=" * 78)
    print("  done")
    print("=" * 78)
    for path in (report_dir / "cost_report.json", report_dir / "summary.md"):
        marker = "ok " if path.is_file() else "MISSING "
        print(f"  {marker}{path}")


if __name__ == "__main__":
    main()
