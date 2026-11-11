"""Pre-evaluation budget allocator for the cost-aware SMOLTRACE scheduling layer.

The allocator answers one question, per task, before anything is measured:

    *At what measurement-fidelity tier should this task be evaluated?*

Three policies ship. All of them return exactly one :class:`~smoltrace_budget.contracts.Allocation`
per input task, ordered by ``task_id`` so the output is byte-for-byte
reproducible for identical inputs.

1. ``"by_category"`` -- a static map from category to tier
   (``gaia -> high``, ``math -> mid``, ``simpleqa -> low``). The category is a
   proxy for how much fidelity a task *needs*: GAIA tasks are multi-hop and
   tool-heavy, so a cheap measurement systematically under-measures them; a
   SimpleQA lookup is answerable in one shot. This policy ignores history
   entirely. This is the default policy for the ``allocated`` arm.

2. ``"by_pass_rate"`` -- a difficulty-driven policy built on historical per-task
   pass rates (``history = {task_id: {"pass_rate": float, "n": int}}``). The pool
   failing a task is a *bias* signal: it means the task sits where the cheap
   measurement is most likely to mis-rank models, so it is measured at higher
   fidelity. Thresholds (documented and justified in ``_tier_for_pass_rate``):

   ===========================  ======
   historical pass rate ``p``   tier
   ===========================  ======
   ``p < 0.35``                 high
   ``0.35 <= p < 0.65``         mid
   ``p >= 0.65``                low
   ===========================  ======

   The raw historical rate is used unshrunk, so the mapping is *strictly
   monotone*: a lower pass rate can never produce a cheaper tier. Tasks with no
   usable history fall back to the category prior: such a task takes the
   ``by_category`` tier for its category (see :func:`category_tier`), because
   with no per-task evidence the category is the only signal there is. That
   keeps an unmeasured GAIA task at ``high`` rather than downgrading it to
   whatever the pool-wide baseline pass rate happens to round to.

3. ``"greedy_budget"`` -- an explicit token budget buys measurement fidelity
   where it discriminates model rankings most cheaply.

   **Ranking-risk.** A task only informs the leaderboard to the extent that its
   outcome differs between models. If every model passes (or every model fails)
   the task, it carries no ranking signal no matter how precisely it is
   measured. With ``p`` the pass rate the model pool achieves on the task, the
   Bernoulli variance ``p * (1 - p)`` is exactly that notion of information; it
   is scaled to a convenient 0..1 range and weighted by how much a category
   depends on the fidelity knobs the cheap tier removes::

       discrimination(p) = 4 * p * (1 - p)        # 1.0 at p = 0.5, 0 at p in {0, 1}
       risk(task)        = w(category) * discrimination(p_eff(task))

   ``w`` is ``1.0`` for ``gaia`` (tool-heavy: the low tier disables tools, so
   cheap measurement distorts it most), ``0.8`` for ``math``, ``0.6`` for
   ``simpleqa`` (a one-shot lookup survives the low tier almost intact), and
   ``0.8`` for any unrecognised category. ``p_eff`` is the historical pass rate
   when available, else the category prior.

   **Greedy.** Start from an all-``low`` baseline (the cheapest feasible
   configuration), then repeatedly take the affordable single-step upgrade
   (``low -> mid`` or ``mid -> high``) with the highest *marginal* efficiency
   ``risk(task) / marginal_tokens``. Ties break on higher risk, then on
   ``task_id`` ascending, so the result is fully deterministic. The loop stops
   when no remaining upgrade fits the budget.

   **Budget safety.** The sum of ``expected_tokens`` never exceeds
   ``token_budget``: the baseline is checked first, and each upgrade is only
   taken if it fits. If the budget cannot even cover the all-``low`` baseline,
   the policy *degrades gracefully to all-low* rather than raising -- a run must
   still produce a leaderboard -- and every ``reason`` string says so.

**Expected-token model.** :func:`estimate_task_tokens` is the single estimator
used by the allocator, the runner and the analysis layer::

    tokens(task, tier) = (
        BASE_PROMPT_TOKENS
        + max_steps * PER_STEP_TOKENS
        + STEP_HISTORY_GROWTH_TOKENS * max_steps * (max_steps - 1) / 2
        + (expected_tool_calls * TOOL_OBSERVATION_TOKENS_PER_CALL if tools_enabled else 0)
    ) * category_token_multiplier(category)

``max_steps`` and ``tools_enabled`` are read from
``contracts.TIER_SPEC`` -- the tier definitions are *not* redefined here. Agent
steps dominate cost (every step re-sends the growing conversation, hence the
``n*(n-1)/2`` history term, which DESIGN.md calls out as making token use
roughly quadratic in steps) and tool use adds observation tokens, which is why
GAIA (tool-heavy) carries a larger category multiplier than SimpleQA. The result
is rounded to whole tokens so all budget arithmetic in floating point is exact.
See :func:`estimate_task_tokens` for the calibration the constants were chosen
against.

**Purity.** Standard library plus :mod:`smoltrace_budget.contracts` only: no
network, no filesystem access, no ``smoltrace`` import, no global state. Repeated
calls with identical inputs return identical output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    TIER_ORDER,
    TIER_SPEC,
    Allocation,
    TaskSpec,
    validate_tier,
)

__all__ = [
    "POLICIES",
    "CATEGORY_TIER",
    "DEFAULT_CATEGORY_TIER",
    "CATEGORY_RISK_WEIGHT",
    "DEFAULT_CATEGORY_RISK_WEIGHT",
    "CATEGORY_PRIOR_PASS_RATE",
    "DEFAULT_PRIOR_PASS_RATE",
    "PASS_RATE_HIGH_TIER_THRESHOLD",
    "PASS_RATE_LOW_TIER_THRESHOLD",
    "BASE_PROMPT_TOKENS",
    "PER_STEP_TOKENS",
    "STEP_HISTORY_GROWTH_TOKENS",
    "TOOL_OBSERVATION_TOKENS_PER_CALL",
    "CATEGORY_TOKEN_MULTIPLIER",
    "DEFAULT_CATEGORY_TOKEN_MULTIPLIER",
    "History",
    "category_tier",
    "category_risk_weight",
    "category_prior_pass_rate",
    "category_token_multiplier",
    "estimate_task_tokens",
    "discrimination_weight",
    "effective_pass_rate",
    "ranking_risk_score",
    "history_pass_rate",
    "summarize_allocations",
    "BudgetAllocator",
    "make_allocator",
]

#: The three allocation policies the runner may select.
POLICIES: Tuple[str, ...] = ("by_category", "by_pass_rate", "greedy_budget")

#: ``history`` shape accepted by every policy:
#: ``{task_id: {"pass_rate": float in [0, 1], "n": int >= 0}}``.
History = Mapping[str, Mapping[str, Any]]

# ---------------------------------------------------------------------------
# Policy constants (documented, deliberately module-level so the report can cite
# them and the analysis layer can reuse them verbatim).
# ---------------------------------------------------------------------------

#: Static category -> tier map behind the ``by_category`` policy. Mirrors
#: DESIGN.md: GAIA is multi-hop/tool-heavy, Math is mid-depth, SimpleQA is a
#: one-shot lookup.
CATEGORY_TIER: Dict[str, str] = {
    "gaia": "high",
    "math": "mid",
    "simpleqa": "low",
}

#: Tier for a category that is not in :data:`CATEGORY_TIER`. ``mid`` is the
#: neutral choice: with no evidence either way, neither starve nor over-spend.
DEFAULT_CATEGORY_TIER: str = "mid"

#: How much a category's *ranking* signal depends on the fidelity knobs the
#: cheap tier removes. GAIA loses tools and multi-hop depth at ``low``, so cheap
#: measurement distorts it most; SimpleQA survives a one-shot answer.
CATEGORY_RISK_WEIGHT: Dict[str, float] = {
    "gaia": 1.0,
    "math": 0.8,
    "simpleqa": 0.6,
}

DEFAULT_CATEGORY_RISK_WEIGHT: float = 0.8

#: Category prior pass rate used when a task has no usable history. These are
#: the pool baseline: the arithmetic mean of the five candidate models'
#: ``true_pass_rate`` for the category in ``smoltrace_budget/models.py``
#: (gaia: mean(0.34, 0.46, 0.62, 0.30, 0.16) = 0.376; math: 0.634; simpleqa:
#: 0.618). Re-derive them if the candidate pool changes.
CATEGORY_PRIOR_PASS_RATE: Dict[str, float] = {
    "gaia": 0.376,
    "math": 0.634,
    "simpleqa": 0.618,
}

DEFAULT_PRIOR_PASS_RATE: float = 0.5

#: ``by_pass_rate`` thresholds. They are symmetric about 0.5 and 0.15 wide
#: because that is roughly where a pass rate stops being distinguishable from
#: 50/50 given realistic sample sizes: the standard error of a pass rate is
#: ``sqrt(p(1-p)/n) <= 0.091`` for ``n = 30`` at ``p = 0.5``, so differences
#: inside ~1.65 standard errors (0.15) are close to noise. Inside that band a
#: mid-tier measurement is enough; outside it the pass/fail direction is clear
#: and the fidelity question is about bias, not noise. Boundary semantics:
#: ``p == 0.35`` is mid and ``p == 0.65`` is low (thresholds are exclusive).
PASS_RATE_HIGH_TIER_THRESHOLD: float = 0.35
PASS_RATE_LOW_TIER_THRESHOLD: float = 0.65

#: Expected-token model constants. Tokens are only an *estimate* of the outlay
#: the decision commits to; the runner prices the real run from OTEL.
#:
#: The estimator keeps the prescribed shape (a prompt base plus a
#: ``max_steps``-driven term) but makes the per-step term grow with the step
#: index, because DESIGN.md is explicit that "each step re-sends the growing
#: conversation, so token use is roughly quadratic in steps"::
#:
#:     tokens = (BASE_PROMPT_TOKENS
#:               + max_steps * PER_STEP_TOKENS
#:               + STEP_HISTORY_GROWTH_TOKENS * max_steps * (max_steps - 1) / 2
#:               + tool_tokens) * category_token_multiplier(category)
#:
#: ``BASE_PROMPT_TOKENS``: system prompt + tool schemas + task instruction +
#:     answer-format scaffolding. Paid at the first call and re-sent thereafter.
#: ``PER_STEP_TOKENS``: the fixed part of a step -- that step's own generation.
#: ``STEP_HISTORY_GROWTH_TOKENS``: extra tokens the step adds to the history that
#:     every *later* step re-sends, so the term scales with ``n*(n-1)/2``. This
#:     is what makes the second fidelity upgrade (mid -> high) cost more than the
#:     first (low -> mid), and hence what makes ``greedy_budget`` buy breadth
#:     across tasks instead of sinking the whole budget into one task.
#: ``TOOL_OBSERVATION_TOKENS_PER_CALL``: a tool call's arguments plus the
#:     observation it returns, charged per ``TaskSpec.expected_tool_calls`` and
#:     only at tiers whose ``TIER_SPEC[...]["tools_enabled"]`` is true (``low``
#:     has tools disabled, so it pays nothing here).
#:
#: Calibration: with ``max_steps`` 2/4/6 from TIER_SPEC, a category multiplier of
#: 1.0 and no tool calls this yields 1,320 / 2,520 / 4,200 tokens for
#: low/mid/high. A GAIA task with one tool call costs 1,518 / 3,358 / 5,290.
#: Measured on the bundled 132-task manifest, the tier totals are ~164k tokens
#: for all-low and ~660k for all-high. Those figures sit in the magnitude band
#: the runner's accounting is designed to produce, so allocator estimates and
#: measured spend agree -- the premise ``greedy_budget`` relies on.
BASE_PROMPT_TOKENS: float = 600.0
PER_STEP_TOKENS: float = 300.0
STEP_HISTORY_GROWTH_TOKENS: float = 120.0
TOOL_OBSERVATION_TOKENS_PER_CALL: float = 400.0

#: Category scaling of the token model. GAIA is tool-heavy and its prompts are
#: long (mean prompt 202 chars, max 2062 in the manifest), so it costs more than
#: the coefficients alone suggest; SimpleQA answers are one short word.
CATEGORY_TOKEN_MULTIPLIER: Dict[str, float] = {
    "gaia": 1.15,
    "math": 1.0,
    "simpleqa": 0.75,
}

DEFAULT_CATEGORY_TOKEN_MULTIPLIER: float = 1.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _category_key(category: str) -> str:
    """Normalise a category label for lookup (``"GAIA "`` -> ``"gaia"``)."""
    return (category or "").strip().lower()


def category_tier(category: str) -> str:
    """The ``by_category`` tier for ``category`` (static map, neutral default)."""
    return CATEGORY_TIER.get(_category_key(category), DEFAULT_CATEGORY_TIER)


def category_risk_weight(category: str) -> float:
    """The ``greedy_budget`` category weight for ``category``."""
    return CATEGORY_RISK_WEIGHT.get(_category_key(category), DEFAULT_CATEGORY_RISK_WEIGHT)


def category_prior_pass_rate(category: str) -> float:
    """The category prior pass rate for ``category`` (baseline pool difficulty)."""
    return CATEGORY_PRIOR_PASS_RATE.get(_category_key(category), DEFAULT_PRIOR_PASS_RATE)


def category_token_multiplier(category: str) -> float:
    """The expected-token category multiplier for ``category``."""
    return CATEGORY_TOKEN_MULTIPLIER.get(
        _category_key(category), DEFAULT_CATEGORY_TOKEN_MULTIPLIER
    )


def estimate_task_tokens(task: TaskSpec, tier: str) -> float:
    """Expected token outlay for measuring ``task`` at ``tier``.

    The single source of truth for the cost side of the allocation decision --
    the runner and the analysis layer import this rather than re-deriving it::

        tokens = (BASE_PROMPT_TOKENS
                  + max_steps * PER_STEP_TOKENS
                  + STEP_HISTORY_GROWTH_TOKENS * max_steps * (max_steps - 1) / 2
                  + tool_tokens) * category_token_multiplier(category)

    with ``max_steps`` / ``tools_enabled`` taken from :data:`TIER_SPEC` and
    ``tool_tokens = expected_tool_calls * TOOL_OBSERVATION_TOKENS_PER_CALL``
    for tool-enabled tiers, ``0`` otherwise. The ``n*(n-1)/2`` term models the
    growing conversation each step re-sends, which DESIGN.md identifies as the
    reason step count dominates evaluation cost.

    The value is rounded to whole tokens, which keeps every sum in floating
    point exact and therefore makes the ``greedy_budget`` budget check strict.

    Args:
        task: the task being measured.
        tier: one of ``contracts.TIER_ORDER``; anything else raises ``ValueError``.
    """
    spec = TIER_SPEC[validate_tier(tier)]
    max_steps = float(spec["max_steps"])

    tool_tokens = 0.0
    if spec.get("tools_enabled"):
        tool_calls = int(getattr(task, "expected_tool_calls", 0) or 0)
        tool_tokens = max(0, tool_calls) * TOOL_OBSERVATION_TOKENS_PER_CALL

    raw = (
        BASE_PROMPT_TOKENS
        + max_steps * PER_STEP_TOKENS
        + STEP_HISTORY_GROWTH_TOKENS * max_steps * (max_steps - 1.0) / 2.0
        + tool_tokens
    ) * category_token_multiplier(task.category)
    return float(round(raw))


def discrimination_weight(pass_rate: float) -> float:
    """Ranking-information content of a task with pass rate ``pass_rate``.

    ``4 * p * (1 - p)``: the Bernoulli variance scaled so it peaks at a clean
    ``1.0`` when models are split 50/50 on the task, and decays to ``0.0`` when
    every model passes (``p = 1``) or every model fails (``p = 0``) -- both of
    which are uninformative about *relative* model quality. ``p`` is clamped to
    ``[0, 1]``.
    """
    p = min(1.0, max(0.0, float(pass_rate)))
    return 4.0 * p * (1.0 - p)


def history_pass_rate(task_id: str, history: Optional[History]) -> Optional[float]:
    """The historical pass rate for ``task_id``, or ``None`` if unavailable.

    Missing task ids and missing/``None`` ``pass_rate`` values are *unavailable*
    (``None``) and are meant to be handled by the caller's fallback path. A
    present-but-malformed entry (non-numeric, or outside ``[0, 1]``) is a data
    bug and raises ``ValueError`` rather than being silently clamped -- the
    allocation must not quietly lie in a published cost report.
    """
    if not history:
        return None
    entry = history.get(task_id)
    if entry is None:
        return None
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"history[{task_id!r}] must be a mapping like {{'pass_rate': .., 'n': ..}}, "
            f"got {type(entry).__name__}"
        )
    raw = entry.get("pass_rate")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"history[{task_id!r}]['pass_rate'] is not a number: {raw!r}"
        ) from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"history[{task_id!r}]['pass_rate']={value} is outside [0, 1]")
    return value


def _history_sample_size(task_id: str, history: Optional[History]) -> Optional[int]:
    """The ``n`` reported alongside a historical pass rate, if present."""
    if not history:
        return None
    entry = history.get(task_id)
    if not isinstance(entry, Mapping):
        return None
    raw = entry.get("n")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def effective_pass_rate(task: TaskSpec, history: Optional[History]) -> float:
    """The pass rate a policy should reason about for ``task``.

    Historical when available, otherwise the category prior from
    :data:`CATEGORY_PRIOR_PASS_RATE`. This is what makes the ``greedy_budget``
    fallback *category-based*: an unmeasured task enters at the pool's baseline
    difficulty for its category rather than at an arbitrary 0.5.

    Note that ``"by_pass_rate"`` does not use this helper for its tier decision:
    with no history it prefers the category's static tier outright, because a
    prior pass rate near the 0.35/0.65 thresholds can straddle a band and throw
    away the category signal. See ``_allocate_by_pass_rate``.
    """
    observed = history_pass_rate(task.id, history)
    if observed is not None:
        return observed
    return category_prior_pass_rate(task.category)


def ranking_risk_score(task: TaskSpec, history: Optional[History] = None) -> float:
    """``greedy_budget`` ranking-risk of ``task``: ``w(category) * 4*p*(1-p)``.

    See the module docstring for the formula and the category weights.
    """
    return category_risk_weight(task.category) * discrimination_weight(
        effective_pass_rate(task, history)
    )


# ---------------------------------------------------------------------------
# The allocator
# ---------------------------------------------------------------------------


class BudgetAllocator:
    """Assigns a measurement-fidelity tier to every task before evaluation.

    Args:
        policy: one of :data:`POLICIES`.
        token_budget: the token ceiling for ``"greedy_budget"``. Strictly
            required (``ValueError``) for that policy and ignored by the other
            two, which are budget-free by construction.
        history: optional historical pass rates, ``{task_id: {"pass_rate":
            float, "n": int}}``. Consumed by ``"by_pass_rate"`` and
            ``"greedy_budget"``; ``"by_category"`` is history-free and reports
            ``historical_pass_rate=None``.

    The instance is immutable after construction (attributes are only read), so
    one allocator may be shared by every round of a run.
    """

    def __init__(
        self,
        policy: str,
        *,
        token_budget: Optional[int] = None,
        history: Optional[History] = None,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {list(POLICIES)}")
        if token_budget is not None and (
            isinstance(token_budget, bool) or not isinstance(token_budget, (int, float))
        ):
            raise ValueError(f"token_budget must be a number or None, got {token_budget!r}")
        if token_budget is not None and token_budget < 0:
            raise ValueError(f"token_budget must be >= 0, got {token_budget!r}")
        if policy == "greedy_budget" and token_budget is None:
            raise ValueError(
                "policy 'greedy_budget' requires an explicit token_budget "
                "(use policy='by_category' or 'by_pass_rate' for budget-free allocation)"
            )

        self.policy: str = policy
        self.token_budget: Optional[float] = (
            float(token_budget) if token_budget is not None else None
        )
        # Deep-copy the nested entries: a shallow dict() would leave the caller's
        # per-task dicts aliased, so a later mutation upstream could silently
        # change an allocation decision that was already made. Non-mapping
        # entries are copied through untouched so that ``history_pass_rate``
        # still raises on them instead of the validation being skipped here.
        self.history: Dict[str, Any] = (
            {
                str(task_id): dict(entry) if isinstance(entry, Mapping) else entry
                for task_id, entry in history.items()
            }
            if history
            else {}
        )

    # -- public API ---------------------------------------------------------

    def allocate(self, tasks: Sequence[TaskSpec]) -> List[Allocation]:
        """Return one :class:`Allocation` per task, sorted by ``task_id``.

        Deterministic: identical inputs always produce an identical list, in the
        same order. The input sequence is never mutated.
        """
        task_list: List[TaskSpec] = list(tasks)

        if self.policy == "by_category":
            allocations = self._allocate_by_category(task_list)
        elif self.policy == "by_pass_rate":
            allocations = self._allocate_by_pass_rate(task_list)
        else:  # greedy_budget -- the constructor guarantees this branch
            allocations = self._allocate_greedy_budget(task_list)

        return sorted(allocations, key=lambda allocation: allocation.task_id)

    # -- shared construction helper ----------------------------------------

    def _make_allocation(
        self,
        task: TaskSpec,
        tier: str,
        *,
        reason: str,
        historical_pass_rate: Optional[float] = None,
    ) -> Allocation:
        """Build a validated :class:`Allocation`, filling in the token estimate."""
        return Allocation(
            task_id=task.id,
            tier=validate_tier(tier),
            policy=self.policy,
            expected_tokens=estimate_task_tokens(task, tier),
            historical_pass_rate=historical_pass_rate,
            reason=reason,
        )

    # -- policy 1: by_category ---------------------------------------------

    def _allocate_by_category(self, tasks: List[TaskSpec]) -> List[Allocation]:
        mapping = ", ".join(f"{category}->{tier}" for category, tier in CATEGORY_TIER.items())
        allocations: List[Allocation] = []
        for task in tasks:
            key = _category_key(task.category)
            if key in CATEGORY_TIER:
                tier = CATEGORY_TIER[key]
                reason = (
                    f"category {task.category!r} maps to tier {tier!r} under the static "
                    f"policy ({mapping})"
                )
            else:
                tier = DEFAULT_CATEGORY_TIER
                reason = (
                    f"category {task.category!r} is not in the static map ({mapping}); "
                    f"using the neutral default tier {tier!r}"
                )
            allocations.append(self._make_allocation(task, tier, reason=reason))
        return allocations

    # -- policy 2: by_pass_rate --------------------------------------------

    def _tier_for_pass_rate(self, pass_rate: float) -> str:
        """Thresholds documented in :data:`PASS_RATE_HIGH_TIER_THRESHOLD`."""
        if pass_rate < PASS_RATE_HIGH_TIER_THRESHOLD:
            return "high"
        if pass_rate < PASS_RATE_LOW_TIER_THRESHOLD:
            return "mid"
        return "low"

    def _allocate_by_pass_rate(self, tasks: List[TaskSpec]) -> List[Allocation]:
        allocations: List[Allocation] = []
        for task in tasks:
            observed = history_pass_rate(task.id, self.history)

            if observed is None:
                # No per-task evidence: fall back to what the category implies.
                # GAIA is multi-hop/tool-heavy by construction, so an unmeasured
                # GAIA task is assumed to need full fidelity; SimpleQA is not.
                tier = category_tier(task.category)
                pass_rate = category_prior_pass_rate(task.category)
                reason = (
                    f"no historical pass rate for this task; category-based prior for "
                    f"{task.category!r} is tier {tier!r} (baseline pool pass rate "
                    f"{pass_rate:.2f}, under the same thresholds "
                    f"high<{PASS_RATE_HIGH_TIER_THRESHOLD:.2f}<="
                    f"mid<{PASS_RATE_LOW_TIER_THRESHOLD:.2f}<=low)"
                )
            else:
                tier = self._tier_for_pass_rate(observed)
                samples = _history_sample_size(task.id, self.history)
                n_note = f" (n={samples})" if samples is not None else ""
                if tier == "high":
                    reason = (
                        f"historical pass rate {observed:.2f}{n_note} is below "
                        f"{PASS_RATE_HIGH_TIER_THRESHOLD:.2f}: the pool usually fails this "
                        f"task, so cheap measurement is the most likely place to mis-rank "
                        f"models -> {tier!r}"
                    )
                elif tier == "mid":
                    reason = (
                        f"historical pass rate {observed:.2f}{n_note} sits in the contested "
                        f"band [{PASS_RATE_HIGH_TIER_THRESHOLD:.2f}, "
                        f"{PASS_RATE_LOW_TIER_THRESHOLD:.2f}) -> {tier!r} fidelity is enough"
                    )
                else:
                    reason = (
                        f"historical pass rate {observed:.2f}{n_note} is at or above "
                        f"{PASS_RATE_LOW_TIER_THRESHOLD:.2f}: the pool reliably passes this "
                        f"task, so extra fidelity buys little -> {tier!r}"
                    )

            allocations.append(
                self._make_allocation(
                    task, tier, reason=reason, historical_pass_rate=observed
                )
            )
        return allocations

    # -- policy 3: greedy_budget -------------------------------------------

    def _allocate_greedy_budget(self, tasks: List[TaskSpec]) -> List[Allocation]:
        """Greedy fidelity upgrades under a hard token ceiling.

        Baseline: everything at ``low``. Then, in repeated rounds, take the
        affordable single-step upgrade with the highest ``risk / marginal
        tokens`` ratio. The token sum is tracked in whole tokens so the
        ``<= token_budget`` guarantee is exact, not approximately true.
        """
        budget = self.token_budget
        if budget is None:  # pragma: no cover - guarded in __init__
            raise RuntimeError("greedy_budget requires token_budget")

        ordered: List[TaskSpec] = sorted(tasks, key=lambda task: task.id)
        n = len(ordered)
        if n == 0:
            return []

        max_tier_index = len(TIER_ORDER) - 1
        tier_index: List[int] = [0] * n
        # Token cost of every task at every tier, indexed like TIER_ORDER.
        costs: List[List[float]] = [
            [estimate_task_tokens(task, tier) for tier in TIER_ORDER] for task in ordered
        ]
        risks: List[float] = [ranking_risk_score(task, self.history) for task in ordered]
        # Marginal costs paid when upgrading task i into its current tier.
        paid: List[List[float]] = [[] for _ in range(n)]

        baseline_tokens = sum(costs[i][0] for i in range(n))
        spent = baseline_tokens
        degraded = baseline_tokens > budget

        while not degraded:
            # Pick the best affordable single-step upgrade. Efficiency is
            # risk-per-marginal-token; ties break on higher risk, then on the
            # lower task index (tasks are pre-sorted by id, so this is stable).
            best_index = -1
            best_key: Optional[Tuple[float, float, int]] = None
            for i in range(n):
                if tier_index[i] >= max_tier_index:
                    continue
                marginal = costs[i][tier_index[i] + 1] - costs[i][tier_index[i]]
                if spent + marginal > budget:
                    continue
                efficiency = float("inf") if marginal <= 0.0 else risks[i] / marginal
                candidate_key = (efficiency, risks[i], -i)
                if best_key is None or candidate_key > best_key:
                    best_index, best_key = i, candidate_key
            if best_index < 0:
                break
            marginal = (
                costs[best_index][tier_index[best_index] + 1]
                - costs[best_index][tier_index[best_index]]
            )
            tier_index[best_index] += 1
            spent += marginal
            paid[best_index].append(marginal)

        # The hard guarantee: never over budget when the budget is attainable.
        if not degraded:
            total = sum(costs[i][tier_index[i]] for i in range(n))
            if total > budget:  # pragma: no cover - defensive, loop enforces it
                raise AssertionError(
                    f"greedy_budget overspent: {total} tokens > budget {budget}"
                )

        allocations: List[Allocation] = []
        for i, task in enumerate(ordered):
            tier = TIER_ORDER[tier_index[i]]
            risk = risks[i]
            observed = history_pass_rate(task.id, self.history)
            source = (
                f"historical p, n={_history_sample_size(task.id, self.history)}"
                if observed is not None
                else f"{task.category!r} prior p, no history"
            )

            if degraded:
                reason = (
                    f"budget {budget:,.0f} < {baseline_tokens:,.0f} tokens needed to measure "
                    f"every task even at 'low'; degraded to all-low rather than dropping "
                    f"coverage (allocation is over budget by "
                    f"{baseline_tokens - budget:,.0f} tokens)"
                )
            elif paid[i]:
                steps = " -> ".join(
                    ["low"]
                    + [TIER_ORDER[k] for k in range(1, tier_index[i] + 1)]
                )
                step_costs = ", ".join(f"+{cost:,.0f}" for cost in paid[i])
                reason = (
                    f"ranking-risk {risk:.4f} = 4*p*(1-p) at p={effective_pass_rate(task, self.history):.3f} "
                    f"({source}) x category weight {category_risk_weight(task.category):.2f}; "
                    f"greedy upgrades {steps} ({step_costs} tokens) fit the "
                    f"{budget:,.0f}-token budget, total spend {costs[i][tier_index[i]]:,.0f}"
                )
            else:
                reason = (
                    f"ranking-risk {risk:.4f} = 4*p*(1-p) at p={effective_pass_rate(task, self.history):.3f} "
                    f"({source}) x category weight {category_risk_weight(task.category):.2f}; "
                    f"kept the low baseline (no affordable upgrade left at "
                    f"{budget:,.0f} tokens of budget)"
                )

            allocations.append(
                self._make_allocation(
                    task, tier, reason=reason, historical_pass_rate=observed
                )
            )
        return allocations


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def make_allocator(policy: str, **kwargs: Any) -> BudgetAllocator:
    """Build a :class:`BudgetAllocator`; ``kwargs`` are forwarded verbatim.

    Present so callers (runner, CLI, tests) can construct an allocator from a
    string policy name without importing the class.
    """
    return BudgetAllocator(policy, **kwargs)


def summarize_allocations(allocations: Sequence[Allocation]) -> Dict[str, Any]:
    """Deterministic tier counts and token totals for a batch of allocations.

    Handy for run metadata and report headers; not part of the allocation
    decision itself.
    """
    by_tier: Dict[str, int] = {tier: 0 for tier in TIER_ORDER}
    total_tokens = 0.0
    policies = set()
    for allocation in allocations:
        by_tier[allocation.tier] = by_tier.get(allocation.tier, 0) + 1
        total_tokens += float(allocation.expected_tokens)
        policies.add(allocation.policy)
    return {
        "n_allocations": len(allocations),
        "by_tier": by_tier,
        "expected_tokens_total": float(round(total_tokens)),
        "policies": sorted(policies),
    }
