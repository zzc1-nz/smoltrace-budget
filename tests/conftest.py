"""Shared pytest fixtures for the smoltrace-budget test-suite.

Importing this module runs the project bootstrap (dependency path + the
``tempfile.mkdtemp`` fix the Windows sandbox requires) before any test module is
imported.

pytest's built-in ``tmp_path`` / ``tmp_path_factory`` fixtures are unusable in
this sandbox: they create their base directory through ``tempfile.mkdtemp``,
whose DACL ends up denying our own process (DSH denies even ``os.scandir`` on
it), so the fixture errors at setup rather than at use. Both fixtures are
therefore redefined here on top of the bootstrap's patched ``mkdtemp``. Test
files should just use ``tmp_path`` normally.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import smoltrace_budget  # noqa: E402,F401  (bootstrap: sys.path + tempfile patch)

from smoltrace_budget.contracts import ModelProfile, TaskRecord, TaskSpec  # noqa: E402


# ---------------------------------------------------------------------------
# tmp_path replacements (must be defined before any test needs them)
# ---------------------------------------------------------------------------


class _SandboxTmpPathFactory:
    """Minimal stand-in for pytest's ``tmp_path_factory``."""

    def __init__(self, base: Path) -> None:
        self._base = base
        self._counter = 0

    @property
    def basetemp(self) -> Path:
        return self._base

    def mktemp(self, name: str, numbered: bool = True) -> Path:
        self._counter += 1
        suffix = f"{self._counter}-" if numbered else ""
        path = self._base / f"{suffix}{name}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def mkdtemp(self, name: str = "tmp", numbered: bool = True) -> Path:
        return self.mktemp(name, numbered=numbered)

    def getbasetemp(self) -> Path:
        return self._base


@pytest.fixture(scope="session")
def tmp_path_factory():
    base = Path(tempfile.mkdtemp(prefix="pytest-of-user-"))
    yield _SandboxTmpPathFactory(base)
    # Leave the directory in place: removing it runs into the same DACL problem.


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("test")


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_tasks() -> list[TaskSpec]:
    """One task per category, with a distinctive expected keyword each."""
    return [
        TaskSpec(
            id="gaia_0",
            prompt="If Eliud Kipchoge could maintain his marathon pace, how many hours to the Moon?",
            difficulty="hard",
            agent_type="both",
            category="gaia",
            source="GAIA",
            expected_tool="multiple",
            expected_tool_calls=1,
            expected_keywords=["17"],
        ),
        TaskSpec(
            id="math_0",
            prompt="What is 2 + 3 * 4?",
            difficulty="easy",
            agent_type="both",
            category="math",
            source="MATH",
            expected_tool="multiple",
            expected_tool_calls=1,
            expected_keywords=["14"],
        ),
        TaskSpec(
            id="simpleqa_0",
            prompt="Name the capital of France.",
            difficulty="easy",
            agent_type="both",
            category="simpleqa",
            source="SimpleQA",
            expected_tool="multiple",
            expected_tool_calls=1,
            expected_keywords=["Paris"],
        ),
    ]


@pytest.fixture
def sample_candidates() -> list[ModelProfile]:
    """A two-model pool: one cheap-favoured, one tool/deliberation-favoured."""
    return [
        ModelProfile(
            model_id="quickthink-7b",
            display_name="QuickThink-7B",
            api_model_id="openai/quickthink-7b",
            true_pass_rate={"gaia": 0.30, "math": 0.60, "simpleqa": 0.70, "default": 0.55},
            tool_reliance=0.15,
            deliberation=0.10,
            verbosity=0.55,
        ),
        ModelProfile(
            model_id="thorough-32b",
            display_name="Thorough-32B",
            api_model_id="openai/thorough-32b",
            true_pass_rate={"gaia": 0.62, "math": 0.74, "simpleqa": 0.70, "default": 0.68},
            tool_reliance=0.92,
            deliberation=0.90,
            verbosity=1.25,
        ),
    ]


@pytest.fixture
def make_record():
    """Factory for synthetic TaskRecords in analysis tests."""

    def _make(
        *,
        arm: str = "all_high",
        candidate: str = "model-a",
        task_id: str = "t1",
        category: str = "math",
        tier: str = "high",
        passed: bool = True,
        total_tokens: int = 4000,
        cost_usd: float = 0.01,
        round_index: int = 0,
        latency_ms: float = 800.0,
        steps: int = 4,
    ) -> TaskRecord:
        return TaskRecord(
            round_index=round_index,
            arm=arm,
            task_id=task_id,
            category=category,
            difficulty="easy",
            agent_type="tool",
            candidate_model=candidate,
            allocated_tier=tier,
            allocated_model={"low": "gpt-4o-mini", "mid": "gpt-4o", "high": "claude-sonnet"}[tier],
            prompt_tokens=total_tokens - 500,
            completion_tokens=500,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            steps=steps,
            passed=passed,
            cost_usd=cost_usd,
            stop_reason="final_answer",
        )

    return _make
