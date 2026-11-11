"""Run pytest with the project's vendored dependencies on sys.path.

The interpreter is externally managed, so the project's dependencies (including
pytest itself) live in ``.venvlibs`` and are only visible after the bootstrap in
``smoltrace_budget/__init__.py`` has run. ``python -m pytest`` therefore fails
with "No module named pytest"; this wrapper is the supported entry point.

Usage:
    python run_tests.py                     # whole suite
    python run_tests.py tests/test_x.py -q  # like pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import smoltrace_budget  # noqa: E402,F401  (bootstrap: sys.path + tempfile patch)

import pytest  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:] or ["tests"]
    raise SystemExit(pytest.main(args))
