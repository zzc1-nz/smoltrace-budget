"""Tests for :mod:`smoltrace_budget.allocator`.

Covers the contract the runner and the published cost report rely on:

* every policy returns exactly one valid :class:`Allocation` per task, sorted by
  ``task_id``;
* ``by_category`` maps the three benchmark categories to the DESIGN.md tiers;
* ``by_pass_rate`` is monotone in difficulty and has a category-based fallback
  for tasks with no history;
* ``greedy_budget`` never exceeds its budget, spends it when it can, and
  degrades gracefully to all-low on an impossible budget;
* the module is deterministic and pure.

The project's pytest lives in ``.venvlibs`` and only becomes importable after
``smoltrace_budget/__init__.py`` has bootstrapped ``sys.path``; ``tests/conftest.py``
does that for us, so a plain ``from smoltrace_budget...`` is correct here.
"""

from __future__ import annotations

import pytest

from smoltrace_budget import allocator as allocator_module
from smoltrace_budget.allocator import (
    BASE_PROMPT_TOKENS,
    CATEGORY_TIER,
    PASS_RATE_HIGH_TIER_THRESHOLD,
    PASS_RATE_LOW_TIER_THRESHOLD,
    PER_STEP_TOKENS,
    POLICIES,
    STEP_HISTORY_GROWTH_TOKENS,
    TOOL_OBSERVATION_TOKENS_PER_CALL,
    BudgetAllocator,
    category_prior_pass_rate,
    category_risk_weight,
    discrimination_weight,
    effective_pass_rate,
    estimate_task_tokens,
    history_pass_rate,
    make_allocator,
    ranking_risk_score,
    summarize_allocations,
)
from smoltrace_budget.contracts import (
    TIER_ORDER,
    TIER_SPEC,
    Allocation,
    TaskSpec,
    tier_rank,
)

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _task(
    task_id: str,
    category: str,
    *,
    difficulty: str = "medium",
    expected_tool_calls: int = 1,
) -> TaskSpec:
    """A minimal but valid TaskSpec for allocator tests."""
    return TaskSpec(
        id=task_id,
        prompt=f"prompt for {task_id}",
        difficulty=difficulty,
        agent_type="both",
        category=category,
        source=category.upper(),
        expected_tool="multiple",
        expected_tool_calls=expected_tool_calls,
        expected_keywords=["answer"],
    )


@pytest.fixture
def mixed_tasks() -> list[TaskSpec]:
    """Two tasks per benchmark category, so tier differences are visible."""
    return [
        _task("gaia_0", "gaia", difficulty="hard"),
        _task("gaia_1", "gaia", difficulty="hard"),
        _task("math_0", "math", difficulty="medium"),
        _task("math_1", "math", difficulty="medium"),
        _task("simpleqa_0", "simpleqa", difficulty="easy"),
        _task("simpleqa_1", "simpleqa", difficulty="easy"),
    ]


def _by_id(allocations: list[Allocation]) -> dict[str, Allocation]:
    return {allocation.task_id: allocation for allocation in allocations}


# ---------------------------------------------------------------------------
# Policy 1: by_category
# ---------------------------------------------------------------------------


def test_by_category_maps_each_category_to_the_designed_tier(mixed_tasks):
    allocations = make_allocator("by_category").allocate(mixed_tasks)
    tiers = _by_id(allocations)

    assert len(allocations) == len(mixed_tasks)
    assert tiers["gaia_0"].tier == "high"
    assert tiers["gaia_1"].tier == "high"
    assert tiers["math_0"].tier == "mid"
    assert tiers["math_1"].tier == "mid"
    assert tiers["simpleqa_0"].tier == "low"
    assert tiers["simpleqa_1"].tier == "low"

    # The static map itself is the documented DESIGN.md mapping.
    assert CATEGORY_TIER == {"gaia": "high", "math": "mid", "simpleqa": "low"}


def test_by_category_same_category_gets_identical_tier(mixed_tasks):
    tiers = _by_id(make_allocator("by_category").allocate(mixed_tasks))
    assert tiers["gaia_0"].tier == tiers["gaia_1"].tier
    assert tiers["math_0"].tier == tiers["math_1"].tier
    assert tiers["simpleqa_0"].tier == tiers["simpleqa_1"].tier


def test_by_category_unknown_category_uses_neutral_default():
    allocation = make_allocator("by_category").allocate([_task("weird_0", "chemistry")])[0]
    assert allocation.tier == "mid"
    assert "chemistry" in allocation.reason


def test_by_category_records_no_historical_pass_rate(mixed_tasks):
    # by_category must not invent a difficulty signal it never used.
    for allocation in make_allocator("by_category").allocate(mixed_tasks):
        assert allocation.historical_pass_rate is None
        assert allocation.policy == "by_category"
        assert allocation.reason


# ---------------------------------------------------------------------------
# Policy 2: by_pass_rate
# ---------------------------------------------------------------------------


def test_by_pass_rate_harder_tasks_never_get_a_cheaper_tier():
    """Monotonicity: lower historical pass rate => tier rank must not decrease."""
    rates = [0.00, 0.05, 0.20, 0.34, 0.35, 0.36, 0.50, 0.64, 0.65, 0.66, 0.95, 1.00]
    tasks = [_task(f"t{i:02d}", "math") for i in range(len(rates))]
    history = {
        task.id: {"pass_rate": rate, "n": 40} for task, rate in zip(tasks, rates)
    }

    tiers = _by_id(make_allocator("by_pass_rate", history=history).allocate(tasks))

    ranks = [tier_rank(tiers[task.id].tier) for task in tasks]
    # rates are ascending, so the tier ranks must be non-increasing.
    assert ranks == sorted(ranks, reverse=True), (
        f"tier ranks {ranks} are not monotonically non-increasing in pass rate"
    )
    # And the extremes must land on the documented tiers.
    assert tiers["t00"].tier == "high"  # p = 0.00, pool always fails
    assert tiers["t11"].tier == "low"  # p = 1.00, pool always passes


def test_by_pass_rate_uses_documented_thresholds():
    high_below = PASS_RATE_HIGH_TIER_THRESHOLD - 0.01
    mid_inside = (PASS_RATE_HIGH_TIER_THRESHOLD + PASS_RATE_LOW_TIER_THRESHOLD) / 2
    low_above = PASS_RATE_LOW_TIER_THRESHOLD

    tasks = [
        _task("p_high", "math"),
        _task("p_mid", "math"),
        _task("p_low", "math"),
    ]
    history = {
        "p_high": {"pass_rate": high_below, "n": 50},
        "p_mid": {"pass_rate": mid_inside, "n": 50},
        "p_low": {"pass_rate": low_above, "n": 50},
    }
    tiers = _by_id(make_allocator("by_pass_rate", history=history).allocate(tasks))

    assert tiers["p_high"].tier == "high"
    assert tiers["p_mid"].tier == "mid"
    # The low threshold is inclusive (>= 0.65 => low).
    assert tiers["p_low"].tier == "low"


def test_by_pass_rate_populates_the_signal_it_used(mixed_tasks):
    history = {
        "gaia_0": {"pass_rate": 0.1, "n": 30},
        "math_0": {"pass_rate": 0.5, "n": 30},
        "simpleqa_0": {"pass_rate": 0.9, "n": 30},
    }
    tiers = _by_id(make_allocator("by_pass_rate", history=history).allocate(mixed_tasks))

    assert tiers["gaia_0"].historical_pass_rate == pytest.approx(0.1)
    assert tiers["math_0"].historical_pass_rate == pytest.approx(0.5)
    assert tiers["simpleqa_0"].historical_pass_rate == pytest.approx(0.9)
    assert tiers["gaia_0"].policy == "by_pass_rate"
    assert all(allocation.reason for allocation in tiers.values())


def test_by_pass_rate_missing_history_falls_back_to_category(mixed_tasks):
    """Tasks absent from `history` must not crash and must use a category prior."""
    # gaia_1 / math_1 / simpleqa_1 are deliberately absent.
    history = {
        "gaia_0": {"pass_rate": 0.02, "n": 30},
        "math_0": {"pass_rate": 0.02, "n": 30},
        "simpleqa_0": {"pass_rate": 0.02, "n": 30},
    }
    tiers = _by_id(make_allocator("by_pass_rate", history=history).allocate(mixed_tasks))

    for task_id in ("gaia_1", "math_1", "simpleqa_1"):
        allocation = tiers[task_id]
        assert allocation.tier in TIER_ORDER
        # No per-task evidence, so the difficulty signal is reported as absent...
        assert allocation.historical_pass_rate is None
        # ...and the reason says the category prior drove the decision.
        assert "prior" in allocation.reason

    # The fallback is the category's designed tier, not a blanket default.
    assert tiers["gaia_1"].tier == "high"
    assert tiers["simpleqa_1"].tier == "low"


def test_by_pass_rate_empty_history_is_the_pure_category_prior(mixed_tasks):
    tiers = _by_id(make_allocator("by_pass_rate").allocate(mixed_tasks))
    for task in mixed_tasks:
        allocation = tiers[task.id]
        assert allocation.historical_pass_rate is None
        assert allocation.tier == CATEGORY_TIER[task.category]
        assert "prior" in allocation.reason


def test_by_pass_rate_handles_a_null_pass_rate_entry():
    """A history entry whose pass_rate is None is 'no evidence', not a crash."""
    tasks = [_task("t_none", "gaia")]
    history = {"t_none": {"pass_rate": None, "n": 0}}
    allocation = make_allocator("by_pass_rate", history=history).allocate(tasks)[0]
    assert allocation.historical_pass_rate is None
    assert allocation.tier == "high"


@pytest.mark.parametrize("bad_rate", [1.5, -0.2, "high", object()])
def test_by_pass_rate_rejects_corrupt_history(bad_rate):
    """A malformed pass rate is a data bug; silently clamping would publish a lie."""
    tasks = [_task("t_bad", "math")]
    with pytest.raises(ValueError):
        make_allocator(
            "by_pass_rate", history={"t_bad": {"pass_rate": bad_rate, "n": 10}}
        ).allocate(tasks)


# ---------------------------------------------------------------------------
# Policy 3: greedy_budget
# ---------------------------------------------------------------------------


def test_greedy_budget_respects_the_budget_strictly():
    tasks = [
        _task(f"gaia_{i}", "gaia", difficulty="hard") for i in range(6)
    ] + [_task(f"math_{i}", "math") for i in range(6)] + [
        _task(f"simpleqa_{i}", "simpleqa", difficulty="easy") for i in range(6)
    ]
    baseline = sum(estimate_task_tokens(task, "low") for task in tasks)

    # Walk a range of budgets, including ones exactly on a task boundary.
    for budget in range(int(baseline), int(baseline) + 40_000, 137):
        allocations = make_allocator("greedy_budget", token_budget=budget).allocate(tasks)
        spent = sum(allocation.expected_tokens for allocation in allocations)
        assert spent <= budget, f"budget {budget} overspent: {spent} tokens"
        assert len(allocations) == len(tasks)
        assert {allocation.tier for allocation in allocations} <= set(TIER_ORDER)


def test_greedy_budget_matches_the_reported_token_sum():
    tasks = [_task(f"math_{i}", "math") for i in range(10)]
    allocations = make_allocator("greedy_budget", token_budget=25_000).allocate(tasks)

    recomputed = sum(
        estimate_task_tokens(task, allocation.tier)
        for task, allocation in zip(tasks, allocations)
    )
    assert sum(allocation.expected_tokens for allocation in allocations) == recomputed


def test_greedy_budget_actually_spends_a_generous_budget():
    """With a big budget the allocator must upgrade, not sit at the baseline."""
    tasks = [_task(f"math_{i}", "math") for i in range(10)]
    baseline = sum(estimate_task_tokens(task, "low") for task in tasks)
    all_high = sum(estimate_task_tokens(task, "high") for task in tasks)

    allocations = make_allocator(
        "greedy_budget", token_budget=int((baseline + all_high) // 2)
    ).allocate(tasks)

    spent = sum(allocation.expected_tokens for allocation in allocations)
    assert spent > baseline, "a budget above the low baseline must buy upgrades"
    assert any(allocation.tier != "low" for allocation in allocations)


def test_greedy_budget_spends_most_of_the_available_budget():
    tasks = [_task(f"math_{i}", "math") for i in range(10)]
    baseline = sum(estimate_task_tokens(task, "low") for task in tasks)
    budget = int(baseline + 9_000)

    allocations = make_allocator("greedy_budget", token_budget=budget).allocate(tasks)
    spent = sum(allocation.expected_tokens for allocation in allocations)

    # Greedy should get within one smallest upgrade step of the ceiling; it can
    # only stop early when nothing else fits.
    assert spent <= budget
    assert budget - spent < 3_000, f"left {budget - spent} tokens unspent"


def test_greedy_budget_degrades_to_all_low_on_an_impossible_budget():
    tasks = [_task(f"gaia_{i}", "gaia", difficulty="hard") for i in range(5)]
    baseline = sum(estimate_task_tokens(task, "low") for task in tasks)

    allocations = make_allocator("greedy_budget", token_budget=1).allocate(tasks)

    assert len(allocations) == len(tasks)
    assert {allocation.tier for allocation in allocations} == {"low"}
    # The over-spend is disclosed, not hidden.
    assert all("degraded" in allocation.reason for allocation in allocations)
    # Nothing is silently dropped: coverage is preserved even when over budget.
    assert sum(a.expected_tokens for a in allocations) == pytest.approx(baseline)


def test_greedy_budget_at_exactly_the_baseline_stays_low():
    tasks = [_task(f"math_{i}", "math") for i in range(5)]
    baseline = sum(estimate_task_tokens(task, "low") for task in tasks)

    allocations = make_allocator("greedy_budget", token_budget=int(baseline)).allocate(tasks)

    assert {allocation.tier for allocation in allocations} == {"low"}
    assert sum(a.expected_tokens for a in allocations) == pytest.approx(baseline)


def test_greedy_budget_zero_budget_still_returns_allocations():
    tasks = [_task(f"math_{i}", "math") for i in range(3)]
    allocations = make_allocator("greedy_budget", token_budget=0).allocate(tasks)
    assert len(allocations) == 3
    assert {allocation.tier for allocation in allocations} == {"low"}


def test_greedy_budget_prefers_high_risk_tasks_per_token():
    """A task near 50% pass rate is upgraded before a task everyone passes."""
    easy = _task("easy_task", "simpleqa")  # prior ~0.62, low risk
    contested = _task("contested_task", "math")  # prior ~0.63 ... see history below
    history = {
        "easy_task": {"pass_rate": 0.99, "n": 100},
        "contested_task": {"pass_rate": 0.50, "n": 100},
    }
    tasks = [easy, contested]

    # Budget for exactly one low->mid upgrade: the contested task must win.
    upgrade_cost = estimate_task_tokens(contested, "mid") - estimate_task_tokens(
        contested, "low"
    )
    baseline = sum(estimate_task_tokens(task, "low") for task in tasks)
    allocations = make_allocator(
        "greedy_budget", token_budget=int(baseline + upgrade_cost), history=history
    ).allocate(tasks)

    tiers = _by_id(allocations)
    assert tiers["contested_task"].tier == "mid"
    assert tiers["easy_task"].tier == "low"


def test_greedy_budget_requires_a_token_budget():
    with pytest.raises(ValueError, match="token_budget"):
        BudgetAllocator("greedy_budget")


def test_greedy_budget_records_history_and_reasons():
    tasks = [_task(f"math_{i}", "math") for i in range(4)]
    history = {"math_0": {"pass_rate": 0.4, "n": 20}}
    allocations = make_allocator(
        "greedy_budget", token_budget=20_000, history=history
    ).allocate(tasks)
    tiers = _by_id(allocations)

    assert tiers["math_0"].historical_pass_rate == pytest.approx(0.4)
    assert tiers["math_1"].historical_pass_rate is None
    for allocation in allocations:
        assert allocation.policy == "greedy_budget"
        assert allocation.reason
        assert "ranking-risk" in allocation.reason or "degraded" in allocation.reason


# ---------------------------------------------------------------------------
# Cross-policy invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_returns_one_valid_allocation_per_task(policy, mixed_tasks):
    kwargs = {"token_budget": 50_000} if policy == "greedy_budget" else {}
    allocations = make_allocator(policy, **kwargs).allocate(mixed_tasks)

    assert isinstance(allocations, list)
    assert len(allocations) == len(mixed_tasks)
    # Exactly one allocation per task, no duplicates.
    assert [allocation.task_id for allocation in allocations] == sorted(
        task.id for task in mixed_tasks
    )
    for allocation in allocations:
        assert isinstance(allocation, Allocation)
        assert allocation.tier in TIER_ORDER
        assert allocation.policy == policy
        assert allocation.expected_tokens > 0
        assert allocation.reason


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_sorts_by_task_id_regardless_of_input_order(policy, mixed_tasks):
    kwargs = {"token_budget": 50_000} if policy == "greedy_budget" else {}
    forward = make_allocator(policy, **kwargs).allocate(mixed_tasks)
    backward = make_allocator(policy, **kwargs).allocate(list(reversed(mixed_tasks)))

    assert [a.task_id for a in forward] == sorted(task.id for task in mixed_tasks)
    assert forward == backward


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_is_deterministic(policy, mixed_tasks):
    kwargs = {"token_budget": 40_000} if policy == "greedy_budget" else {}
    first = make_allocator(policy, **kwargs).allocate(mixed_tasks)
    second = make_allocator(policy, **kwargs).allocate(mixed_tasks)

    assert first == second
    assert [a.to_dict() for a in first] == [a.to_dict() for a in second]


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_handles_an_empty_task_list(policy):
    kwargs = {"token_budget": 10_000} if policy == "greedy_budget" else {}
    assert make_allocator(policy, **kwargs).allocate([]) == []


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_does_not_mutate_its_input(policy, mixed_tasks):
    kwargs = {"token_budget": 40_000} if policy == "greedy_budget" else {}
    snapshot = [task.to_dict() for task in mixed_tasks]
    make_allocator(policy, **kwargs).allocate(mixed_tasks)
    assert [task.to_dict() for task in mixed_tasks] == snapshot


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        BudgetAllocator("by_vibes")


def test_negative_token_budget_is_rejected():
    with pytest.raises(ValueError, match="token_budget"):
        BudgetAllocator("greedy_budget", token_budget=-1)


def test_summarize_allocations_is_consistent_with_the_allocations(mixed_tasks):
    allocations = make_allocator("by_category").allocate(mixed_tasks)
    summary = summarize_allocations(allocations)

    assert summary["n_allocations"] == len(mixed_tasks)
    assert summary["policies"] == ["by_category"]
    assert sum(summary["by_tier"].values()) == len(mixed_tasks)
    assert summary["expected_tokens_total"] == pytest.approx(
        sum(allocation.expected_tokens for allocation in allocations)
    )
    assert summary["by_tier"] == {"low": 2, "mid": 2, "high": 2}


# ---------------------------------------------------------------------------
# Expected-token estimator (shared with the runner and the analysis layer)
# ---------------------------------------------------------------------------


def test_estimate_task_tokens_grows_with_tier():
    gaia = _task("gaia_0", "gaia")
    estimates = [estimate_task_tokens(gaia, tier) for tier in TIER_ORDER]
    assert estimates == sorted(estimates)
    assert len(set(estimates)) == len(TIER_ORDER)


def test_estimate_task_tokens_uses_tier_spec_max_steps():
    """The estimator must read its step counts from contracts.TIER_SPEC."""
    task = _task("math_0", "math")
    for tier in TIER_ORDER:
        max_steps = TIER_SPEC[tier]["max_steps"]
        expected = (
            BASE_PROMPT_TOKENS
            + max_steps * PER_STEP_TOKENS
            + STEP_HISTORY_GROWTH_TOKENS * max_steps * (max_steps - 1) / 2
            + (
                task.expected_tool_calls * TOOL_OBSERVATION_TOKENS_PER_CALL
                if TIER_SPEC[tier]["tools_enabled"]
                else 0
            )
        ) * allocator_module.category_token_multiplier(task.category)
        assert estimate_task_tokens(task, tier) == pytest.approx(round(expected))


def test_estimate_task_tokens_charges_tools_only_when_enabled():
    """`low` disables tools, so its estimate must not include tool observations."""
    task = _task("math_0", "math", expected_tool_calls=5)
    low = estimate_task_tokens(task, "low")
    mid = estimate_task_tokens(task, "mid")
    no_tools_task = _task("math_1", "math", expected_tool_calls=0)
    mid_no_tools = estimate_task_tokens(no_tools_task, "mid")

    assert TIER_SPEC["low"]["tools_enabled"] is False
    assert mid - mid_no_tools == pytest.approx(5 * TOOL_OBSERVATION_TOKENS_PER_CALL)


def test_estimate_task_tokens_is_category_scaled():
    """GAIA is tool-heavy and priced above a SimpleQA lookup at the same tier."""
    gaia = _task("gaia_0", "gaia")
    simpleqa = _task("simpleqa_0", "simpleqa")
    assert estimate_task_tokens(gaia, "high") > estimate_task_tokens(simpleqa, "high")


def test_estimate_task_tokens_lands_in_the_designed_magnitude():
    """DESIGN.md targets 3000-5000 tokens per task at the high tier."""
    for category in ("gaia", "math", "simpleqa"):
        tokens = estimate_task_tokens(_task(f"{category}_0", category), "high")
        assert 2_500 <= tokens <= 7_000, f"{category} high-tier estimate {tokens} out of range"
    # Not every category may be cheap: GAIA must be the most expensive one.
    assert estimate_task_tokens(_task("g", "gaia"), "high") >= estimate_task_tokens(
        _task("m", "math"), "high"
    )


def test_estimate_task_tokens_returns_whole_tokens():
    for category in ("gaia", "math", "simpleqa"):
        for tier in TIER_ORDER:
            tokens = estimate_task_tokens(_task("t", category), tier)
            assert isinstance(tokens, float)
            assert tokens == int(tokens), "fractional tokens would break exact budgeting"


def test_estimate_task_tokens_rejects_an_unknown_tier():
    with pytest.raises(ValueError):
        estimate_task_tokens(_task("t", "math"), "ultra")


def test_estimate_task_tokens_makes_later_upgrades_more_expensive():
    """mid->high must cost more than low->mid, or greedy sinks into one task.

    The growing-conversation term means a 6-step run costs more than linearly
    more than a 4-step run, which is what lets ``greedy_budget`` spread its
    budget over several tasks instead of escalating a single one to ``high``.
    """
    task = _task("math_0", "math")
    low = estimate_task_tokens(task, "low")
    mid = estimate_task_tokens(task, "mid")
    high = estimate_task_tokens(task, "high")

    assert (mid - low) < (high - mid)


# ---------------------------------------------------------------------------
# Ranking-risk formula
# ---------------------------------------------------------------------------


def test_discrimination_weight_peaks_at_fifty_percent():
    assert discrimination_weight(0.5) == pytest.approx(1.0)
    assert discrimination_weight(0.0) == pytest.approx(0.0)
    assert discrimination_weight(1.0) == pytest.approx(0.0)
    # Symmetric about 0.5.
    assert discrimination_weight(0.3) == pytest.approx(discrimination_weight(0.7))
    # Monotone increasing on [0, 0.5].
    assert discrimination_weight(0.1) < discrimination_weight(0.3) < discrimination_weight(0.5)


def test_discrimination_weight_clamps_out_of_range_input():
    assert discrimination_weight(-3.0) == pytest.approx(0.0)
    assert discrimination_weight(4.0) == pytest.approx(0.0)


def test_ranking_risk_score_combines_pass_rate_and_category():
    history = {"t": {"pass_rate": 0.5, "n": 10}}
    task = _task("t", "gaia")
    assert ranking_risk_score(task, history) == pytest.approx(
        category_risk_weight("gaia") * discrimination_weight(0.5)
    )
    # GAIA outweighs SimpleQA at equal difficulty.
    assert ranking_risk_score(_task("t", "gaia"), history) > ranking_risk_score(
        _task("t", "simpleqa"), history
    )


def test_ranking_risk_score_falls_back_to_the_category_prior():
    task = _task("t", "math")
    expected = category_risk_weight("math") * discrimination_weight(
        category_prior_pass_rate("math")
    )
    assert ranking_risk_score(task) == pytest.approx(expected)
    assert effective_pass_rate(task, None) == pytest.approx(
        category_prior_pass_rate("math")
    )


def test_history_lookup_ignores_missing_entries():
    assert history_pass_rate("nope", {}) is None
    assert history_pass_rate("nope", None) is None
    assert history_pass_rate("t", {"t": {"pass_rate": 0.25, "n": 5}}) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# make_allocator
# ---------------------------------------------------------------------------


def test_make_allocator_returns_a_budget_allocator():
    instance = make_allocator("by_category")
    assert isinstance(instance, BudgetAllocator)
    assert instance.policy == "by_category"


def test_make_allocator_forwards_keyword_arguments():
    history = {"t": {"pass_rate": 0.1, "n": 5}}
    instance = make_allocator("greedy_budget", token_budget=12_345, history=history)
    assert instance.token_budget == pytest.approx(12_345.0)
    assert instance.history == history


def test_history_is_copied_not_aliased():
    """Mutating the caller's dict afterwards must not change the allocator."""
    history = {"t": {"pass_rate": 0.1, "n": 5}}
    instance = make_allocator("by_pass_rate", history=history)
    history["t"]["pass_rate"] = 0.99
    history["t2"] = {"pass_rate": 0.5, "n": 5}
    assert history_pass_rate("t", instance.history) == pytest.approx(0.1)
    assert history_pass_rate("t2", instance.history) is None
