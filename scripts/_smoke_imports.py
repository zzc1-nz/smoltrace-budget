"""Smoke test: can we import SMOLTRACE and drive its building blocks?"""

from _common import PROJECT_ROOT  # noqa: F401  (bootstrap + sys.path first)

import inspect

import smoltrace
from smoltrace import core, otel, utils

print("smoltrace version:", getattr(smoltrace, "__version__", "unknown"))
print("file:", smoltrace.__file__)
print()
for label, fn in [
    ("run_evaluation", core.run_evaluation),
    ("evaluate_single_test", core.evaluate_single_test),
    ("initialize_agent", core.initialize_agent),
    ("compute_leaderboard_row", utils.compute_leaderboard_row),
    ("setup_inmemory_otel", otel.setup_inmemory_otel),
    ("load_test_cases_from_hf", core.load_test_cases_from_hf),
    ("extract_traces", core.extract_traces),
    ("extract_metrics", core.extract_metrics),
]:
    try:
        print(f"{label}: {inspect.signature(fn)}")
    except Exception as exc:  # pragma: no cover
        print(f"{label}: <unavailable: {exc}>")
    print()
