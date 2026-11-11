"""Shared data contracts for the cost-aware evaluation scheduling layer.

Every component (allocator, backend, runner, analysis) speaks these shapes, so
they can be developed and tested independently. Keep this module dependency-free
beyond the standard library: it is imported by everything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Fidelity tiers
# ---------------------------------------------------------------------------

TIER_ORDER: List[str] = ["low", "mid", "high"]

#: A tier is a *measurement fidelity* bundle, not a model. The label records the
#: model class you would have to pay for to obtain that fidelity, which is how
#: evaluation cost is usually reasoned about in practice.
TIER_LABELS: Dict[str, str] = {
    "low": "gpt-4o-mini",
    "mid": "gpt-4o",
    "high": "claude-sonnet",
}

#: Measurement-fidelity knobs. `max_steps` dominates cost: each agent step
#: re-sends the whole growing conversation, so token use is roughly quadratic in
#: steps before truncation.
TIER_SPEC: Dict[str, Dict[str, Any]] = {
    "low": {
        "max_steps": 2,
        "tools_enabled": False,
        "grader": "keyword",
        "label": TIER_LABELS["low"],
        "description": "single-shot, no tools, keyword grader",
    },
    "mid": {
        "max_steps": 4,
        "tools_enabled": True,
        "grader": "keyword+semantic",
        "label": TIER_LABELS["mid"],
        "description": "multi-step with tools, keyword+semantic grader",
    },
    "high": {
        "max_steps": 6,
        "tools_enabled": True,
        "grader": "strict+semantic",
        "label": TIER_LABELS["high"],
        "description": "full-depth with all tools, strict semantic grader",
    },
}


def tier_rank(tier: str) -> int:
    """0 for the cheapest tier, 2 for the most expensive."""
    return TIER_ORDER.index(tier)


def validate_tier(tier: str) -> str:
    if tier not in TIER_SPEC:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIER_ORDER}")
    return tier


# ---------------------------------------------------------------------------
# Candidate models under evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """A model in the leaderboard pool, plus its *true* latent ability.

    ``true_pass_rate`` is the ground truth the evaluation is trying to recover.
    It is per category because ability is not uniform across GAIA / Math /
    SimpleQA, and it is what a high-fidelity evaluation should measure.
    """

    model_id: str
    display_name: str
    #: provider-facing id handed to the model backend (litellm-style)
    api_model_id: str
    true_pass_rate: Dict[str, float]
    #: how much a model relies on tools; low-fidelity tiers disable them and
    #: therefore under-measure this model disproportionately
    tool_reliance: float
    #: how much extra reasoning the model does before answering; cheap tiers
    #: truncate at max_steps and cut these models off mid-deliberation
    deliberation: float
    #: token multiplier relative to a baseline model (verbose models cost more)
    verbosity: float = 1.0

    def pass_rate_for(self, category: str) -> float:
        return float(self.true_pass_rate.get(category, self.true_pass_rate.get("default", 0.0)))


@dataclass(frozen=True)
class TaskSpec:
    """One benchmark task, as frozen in data/tasks/benchmark_v1.jsonl."""

    id: str
    prompt: str
    difficulty: str
    agent_type: str
    category: str
    source: str
    expected_tool: Optional[str] = None
    expected_tool_calls: int = 1
    expected_keywords: List[str] = field(default_factory=list)
    true_answer: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaskSpec":
        keywords = payload.get("expected_keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        return cls(
            id=str(payload["id"]),
            prompt=str(payload["prompt"]),
            difficulty=str(payload.get("difficulty") or "unknown"),
            agent_type=str(payload.get("agent_type") or "both"),
            category=str(payload.get("category") or "unknown"),
            source=str(payload.get("source") or payload.get("category") or "unknown"),
            expected_tool=payload.get("expected_tool"),
            expected_tool_calls=int(payload.get("expected_tool_calls") or 1),
            expected_keywords=[str(k) for k in keywords],
            true_answer=payload.get("true_answer"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


@dataclass
class Allocation:
    """The allocator's decision for one task.

    ``tier`` is what the runner must honour; the other fields exist so the
    decision is explainable in the report ("why did this task get measured at
    high fidelity?").
    """

    task_id: str
    tier: str
    policy: str
    #: expected token outlay assumed when the decision was made
    expected_tokens: float
    #: the difficulty signal the policy used, if any (historical pass rate)
    historical_pass_rate: Optional[float] = None
    #: free-form, shown in the report
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        validate_tier(self.tier)
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-task measurement record
# ---------------------------------------------------------------------------


@dataclass
class TaskRecord:
    """What one evaluated (task, candidate model) pair produced.

    This is the join key across everything: the leaderboards are aggregations of
    these, and the cost report sums them.
    """

    round_index: int
    arm: str
    task_id: str
    category: str
    difficulty: str
    agent_type: str
    candidate_model: str
    allocated_tier: str
    allocated_model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    steps: int
    passed: bool
    cost_usd: float
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaskRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

ARM_ALL_HIGH = "all_high"
ARM_ALL_LOW = "all_low"
ARM_ALLOCATED = "allocated"
ARMS: List[str] = [ARM_ALL_HIGH, ARM_ALL_LOW, ARM_ALLOCATED]

ARM_DESCRIPTIONS: Dict[str, str] = {
    ARM_ALL_HIGH: "every task measured at the high tier (gold standard)",
    ARM_ALL_LOW: "every task measured at the low tier (cheapest possible)",
    ARM_ALLOCATED: "the budget allocator assigns a tier per task",
}

#: The reference arm every comparison is made against.
REFERENCE_ARM = ARM_ALL_HIGH


# ---------------------------------------------------------------------------
# Leaderboard rows
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    """One row of a per-arm leaderboard."""

    arm: str
    round_index: int
    model: str
    passed: int
    evaluated: int
    pass_rate: float
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "TIER_ORDER",
    "TIER_LABELS",
    "TIER_SPEC",
    "tier_rank",
    "validate_tier",
    "ModelProfile",
    "TaskSpec",
    "Allocation",
    "TaskRecord",
    "ARM_ALL_HIGH",
    "ARM_ALL_LOW",
    "ARM_ALLOCATED",
    "ARMS",
    "ARM_DESCRIPTIONS",
    "REFERENCE_ARM",
    "LeaderboardEntry",
]
