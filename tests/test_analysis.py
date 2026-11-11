"""Tests for :mod:`smoltrace_budget.analysis`.

Every record here is synthetic: nothing in this file reads ``artifacts/`` or
requires the runner to have executed. ``make_record`` comes from
``tests/conftest.py``; the local helpers below only add the few shapes that
fixture does not cover (a labelled task set, and leaderboards built by hand).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from smoltrace_budget.analysis import (
    MEMBERSHIP_WEIGHT,
    POOLED_ROUND_INDEX,
    STABILITY_TOP_K,
    TAU_WEIGHT,
    arm_cost,
    build_cost_report,
    build_leaderboard,
    cost_by_category,
    cost_rank_tradeoff,
    detect_cheap_arm_hallucination,
    flatten_artifacts,
    kendall_tau,
    leaderboards_by_round,
    load_artifacts,
    pooled_leaderboards,
    rank_stability,
    render_summary_markdown,
)
from smoltrace_budget.contracts import (
    ARM_ALL_HIGH,
    ARM_ALL_LOW,
    ARM_ALLOCATED,
    ARMS,
    REFERENCE_ARM,
    LeaderboardEntry,
    TaskRecord,
)

#: Five candidates, best first — the canonical pool shape of the real study.
POOL = ["thorough-32b", "balanced-13b", "mathwhiz-9b", "quickthink-7b", "tinyfast-3b"]


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def make_model_records(
    make_record,
    *,
    arm: str = ARM_ALL_HIGH,
    passed_by_model: dict,
    round_index: int = 0,
    tokens_by_model: dict | None = None,
    cost_by_model: dict | None = None,
    tier: str = "high",
    category: str = "math",
    n_tasks: int = 4,
):
    """``passed_by_model`` maps model -> how many of ``n_tasks`` tasks it passed."""
    tokens_by_model = tokens_by_model or {}
    cost_by_model = cost_by_model or {}
    records = []
    for model, n_passed in passed_by_model.items():
        for task_index in range(n_tasks):
            records.append(
                make_record(
                    arm=arm,
                    candidate=model,
                    task_id=f"{category}_{task_index}",
                    category=category,
                    tier=tier,
                    passed=task_index < n_passed,
                    total_tokens=tokens_by_model.get(model, 4000),
                    cost_usd=cost_by_model.get(model, 0.01),
                    round_index=round_index,
                    latency_ms=500.0 + task_index,
                )
            )
    return records


def board(*models):
    """A leaderboard given model names in rank order (rank 1 first)."""
    return [
        LeaderboardEntry(
            arm="synthetic",
            round_index=0,
            model=model,
            passed=len(models) - index,
            evaluated=len(models),
            pass_rate=float(len(models) - index),
            total_tokens=1000,
            total_cost_usd=0.01,
            avg_latency_ms=100.0,
            rank=index + 1,
        )
        for index, model in enumerate(models)
    ]


@pytest.fixture
def artifacts_tmpdir():
    """A scratch directory that this sandbox can actually write to.

    pytest's ``tmp_path`` is unusable here: DSH's Windows sandbox denies
    ``os.scandir`` on the pytest basetemp, so the fixture errors before the test
    body runs. ``tempfile.mkdtemp`` is patched by ``smoltrace_budget`` to create
    directories our process owns, so we build on that instead.
    """
    path = Path(tempfile.mkdtemp(prefix="analysis-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# build_leaderboard
# ---------------------------------------------------------------------------


def test_leaderboard_ranks_by_pass_rate(make_record):
    records = make_model_records(
        make_record,
        passed_by_model={"model-a": 4, "model-b": 3, "model-c": 1},
    )
    entries = build_leaderboard(records, ARM_ALL_HIGH, 0)

    assert [entry.model for entry in entries] == ["model-a", "model-b", "model-c"]
    assert [entry.rank for entry in entries] == [1, 2, 3]
    assert entries[0].pass_rate == 100.0
    assert entries[1].pass_rate == 75.0
    assert entries[2].pass_rate == 25.0
    assert entries[0].passed == 4
    assert entries[0].evaluated == 4
    assert all(entry.arm == ARM_ALL_HIGH and entry.round_index == 0 for entry in entries)


def test_leaderboard_aggregates_tokens_cost_and_latency(make_record):
    records = make_model_records(
        make_record,
        passed_by_model={"model-a": 2},
        tokens_by_model={"model-a": 4000},
        cost_by_model={"model-a": 0.5},
        n_tasks=3,
    )
    (entry,) = build_leaderboard(records, ARM_ALL_HIGH, 0)

    assert entry.total_tokens == 12000
    assert entry.total_cost_usd == pytest.approx(1.5)
    assert entry.avg_latency_ms == pytest.approx((500.0 + 501.0 + 502.0) / 3)


def test_leaderboard_tie_break_prefers_cheaper_model_then_name(make_record):
    # Identical pass rates: the cheaper model must win, then the name decides.
    records = []
    for model, cost in (("zebra", 0.02), ("alpha", 0.02), ("cheap", 0.001)):
        for task_index in range(2):
            records.append(
                make_record(
                    candidate=model,
                    task_id=f"t{task_index}",
                    passed=task_index == 0,
                    cost_usd=cost,
                )
            )
    entries = build_leaderboard(records, ARM_ALL_HIGH, 0)

    assert [entry.pass_rate for entry in entries] == [50.0, 50.0, 50.0]
    assert [entry.model for entry in entries] == ["cheap", "alpha", "zebra"]
    assert [entry.rank for entry in entries] == [1, 2, 3]


def test_leaderboard_tie_break_is_deterministic_under_shuffling(make_record):
    records = make_model_records(
        make_record,
        passed_by_model={"b": 2, "a": 2, "c": 2},
        cost_by_model={"a": 0.01, "b": 0.01, "c": 0.01},
    )
    forward = build_leaderboard(records, ARM_ALL_HIGH, 0)
    backward = build_leaderboard(list(reversed(records)), ARM_ALL_HIGH, 0)

    assert [entry.model for entry in forward] == ["a", "b", "c"]
    assert [(e.model, e.rank) for e in forward] == [(e.model, e.rank) for e in backward]


def test_leaderboard_empty_input():
    assert build_leaderboard([], ARM_ALL_HIGH, 0) == []


def test_leaderboard_accepts_record_dicts(make_record):
    record = make_record(candidate="model-a", passed=True)
    as_dict = record.to_dict()

    from_objects = build_leaderboard([record], ARM_ALL_HIGH, 3)
    from_dicts = build_leaderboard([as_dict], ARM_ALL_HIGH, 3)

    assert [e.to_dict() for e in from_objects] == [e.to_dict() for e in from_dicts]


# ---------------------------------------------------------------------------
# arm_cost / cost_by_category
# ---------------------------------------------------------------------------


def test_arm_cost_totals(make_record):
    records = make_model_records(
        make_record,
        passed_by_model={"model-a": 2, "model-b": 1},
        n_tasks=3,
        tokens_by_model={"model-a": 4000, "model-b": 1000},
        cost_by_model={"model-a": 0.02, "model-b": 0.01},
    )
    totals = arm_cost(records)

    assert totals == {
        "total_tokens": 15000,
        "total_cost_usd": pytest.approx(0.09),
        "n_records": 6,
        "n_tasks": 3,
        "n_models": 2,
    }


def test_arm_cost_empty():
    assert arm_cost([]) == {
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "n_records": 0,
        "n_tasks": 0,
        "n_models": 0,
    }


def test_cost_by_category_shares(make_record):
    records = [
        make_record(category="gaia", total_tokens=3000, cost_usd=0.03),
        make_record(category="math", total_tokens=1000, cost_usd=0.01),
    ]
    breakdown = cost_by_category(records)

    assert set(breakdown) == {"gaia", "math"}
    assert breakdown["gaia"]["tokens"] == 3000
    assert breakdown["gaia"]["cost"] == pytest.approx(0.03)
    assert breakdown["gaia"]["share_of_tokens"] == pytest.approx(75.0)
    assert breakdown["gaia"]["share_of_cost"] == pytest.approx(75.0)
    assert breakdown["math"]["share_of_tokens"] == pytest.approx(25.0)
    assert breakdown["gaia"]["n_records"] == 1
    # Aliases report_render reads must agree with the canonical keys.
    assert breakdown["gaia"]["total_tokens"] == breakdown["gaia"]["tokens"]
    assert breakdown["gaia"]["total_cost_usd"] == breakdown["gaia"]["cost"]


def test_cost_by_category_empty():
    assert cost_by_category([]) == {}


# ---------------------------------------------------------------------------
# kendall_tau
# ---------------------------------------------------------------------------


def test_kendall_tau_identical_is_one():
    assert kendall_tau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_kendall_tau_reversed_is_minus_one():
    assert kendall_tau([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == -1.0


def test_kendall_tau_known_permutation():
    # One adjacent swap in 5 elements: 9 concordant, 1 discordant of 10 pairs.
    assert kendall_tau([1, 2, 3, 4, 5], [2, 1, 3, 4, 5]) == pytest.approx(0.8)


def test_kendall_tau_handles_ties():
    # All pairs tied on one side => denominator zero => undefined (None), not 0.0.
    assert kendall_tau([1, 1, 1], [1, 2, 3]) is None
    assert kendall_tau([1, 1, 1], [3, 3, 3]) is None
    # A single tie on each side stays well defined.
    value = kendall_tau([1, 1, 2, 3], [1, 2, 2, 3])
    assert value is not None and -1.0 <= value <= 1.0


def test_kendall_tau_needs_two_points():
    assert kendall_tau([], []) is None
    assert kendall_tau([1], [2]) is None


def test_kendall_tau_rejects_ragged_input():
    with pytest.raises(ValueError):
        kendall_tau([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# rank_stability
# ---------------------------------------------------------------------------


def test_rank_stability_reference_arm_is_perfect():
    leaderboards = {"all_high": board(*POOL)}
    result = rank_stability(leaderboards)["arms"][ARM_ALL_HIGH]

    assert result["stability_pct"] == 100.0
    assert result["stability_delta_pct"] == 0.0
    assert result["top3_membership_overlap"] == STABILITY_TOP_K
    assert result["top3_ordering_identical"] is True
    assert result["kendall_tau"] == 1.0
    assert result["rank_changes"] == {
        model: {
            "reference_rank": position,
            "arm_rank": position,
            "delta": 0,
        }
        for position, model in enumerate(POOL[:5], start=1)
    }


def test_rank_stability_detects_top3_swap():
    reference = board(*POOL)
    # Swap the top two: same membership, different order.
    swapped = board(POOL[1], POOL[0], POOL[2], POOL[3], POOL[4])
    # Swap rank 3 into rank 1: membership actually changes.
    replaced = board(POOL[2], POOL[1], POOL[0], POOL[3], POOL[4])

    result = rank_stability({"all_high": reference, "allocated": swapped})
    swapped_stats = result["arms"]["allocated"]
    assert swapped_stats["top3_membership_overlap"] == 3
    assert swapped_stats["top3_ordering_identical"] is False
    assert swapped_stats["stability_pct"] < 100.0
    assert swapped_stats["stability_delta_pct"] < 0.0

    result2 = rank_stability({"all_high": reference, "allocated": replaced})
    replaced_stats = result2["arms"]["allocated"]
    assert replaced_stats["top3_membership_overlap"] == 3
    assert replaced_stats["top3_ordering_identical"] is False


def test_rank_stability_membership_drops_when_top3_loses_a_model():
    reference = board(*POOL)
    arm = board(POOL[0], POOL[1], POOL[3], POOL[2], POOL[4])  # quickthink displaces mathwhiz
    stats = rank_stability({"all_high": reference, "allocated": arm})["arms"]["allocated"]

    assert stats["top3_membership_overlap"] == 2
    assert stats["top3_membership_overlap_ratio"] == pytest.approx(2 / 3, abs=1e-4)


def test_rank_stability_formula_is_exact():
    """stability_pct must equal 100 * (w_m * ratio + w_t * (tau+1)/2)."""
    reference = board(*POOL)
    arm = board(POOL[0], POOL[2], POOL[1], POOL[3], POOL[4])
    stats = rank_stability({"all_high": reference, "allocated": arm})["arms"]["allocated"]

    expected = 100.0 * (
        MEMBERSHIP_WEIGHT * stats["top3_membership_overlap_ratio"]
        + TAU_WEIGHT * stats["kendall_tau_normalised"]
    )
    assert stats["stability_pct"] == pytest.approx(expected, abs=1e-3)
    assert stats["kendall_tau"] == pytest.approx((stats["kendall_tau_normalised"] * 2) - 1)


def test_rank_stability_rank_changes_cover_top5_of_either():
    reference = board(*POOL)
    arm = board(POOL[0], POOL[1], POOL[2], POOL[4], POOL[3])
    stats = rank_stability({"all_high": reference, "allocated": arm})["arms"]["allocated"]

    assert set(stats["rank_changes"]) == set(POOL)
    assert stats["rank_changes"]["tinyfast-3b"] == {
        "reference_rank": 5,
        "arm_rank": 4,
        "delta": 1,
    }
    assert stats["rank_changes"][POOL[0]]["delta"] == 0


def test_rank_stability_without_reference_is_empty():
    result = rank_stability({ARM_ALLOCATED: board(*POOL)})

    assert result["arms"] == {}
    assert result["mean_stability_pct"] is None
    assert result["reference_arm"] == REFERENCE_ARM


def test_rank_stability_empty_leaderboards():
    result = rank_stability({})
    assert result["arms"] == {}
    assert result["reference_top3"] == []


def test_rank_stability_handles_disjoint_model_sets():
    reference = board("a", "b", "c")
    arm = board("x", "y", "z")
    stats = rank_stability({"all_high": reference, "allocated": arm})["arms"]["allocated"]

    assert stats["top3_membership_overlap"] == 0
    assert stats["kendall_tau"] is None
    assert stats["stability_pct"] == 0.0


def test_rank_stability_ranks_unranked_rows():
    """Rows with the default rank=0 must be ranked, not compared as all-zero."""
    def unranked(*models):
        return [
            LeaderboardEntry(
                arm="a",
                round_index=0,
                model=model,
                passed=5 - index,
                evaluated=5,
                pass_rate=100.0 - 10 * index,
                total_tokens=1,
                total_cost_usd=0.01,
                avg_latency_ms=1.0,
                # rank deliberately left at the 0 default
            )
            for index, model in enumerate(models)
        ]

    stats = rank_stability(
        {"all_high": unranked("a", "b", "c"), "allocated": unranked("b", "a", "c")}
    )["arms"]["allocated"]

    assert stats["kendall_tau"] is not None
    assert stats["rank_changes"]["a"] == {
        "reference_rank": 1,
        "arm_rank": 2,
        "delta": -1,
    }
    assert stats["rank_changes"]["b"] == {
        "reference_rank": 2,
        "arm_rank": 1,
        "delta": 1,
    }


# ---------------------------------------------------------------------------
# detect_cheap_arm_hallucination
# ---------------------------------------------------------------------------


def test_hallucination_fires_when_low_arm_crowns_a_different_model(make_record):
    high = make_model_records(
        make_record,
        arm=ARM_ALL_HIGH,
        passed_by_model={"thorough-32b": 4, "balanced-13b": 3, "quickthink-7b": 2},
    )
    low = make_model_records(
        make_record,
        arm=ARM_ALL_LOW,
        passed_by_model={"thorough-32b": 1, "balanced-13b": 2, "quickthink-7b": 4},
        tier="low",
    )
    leaderboards = pooled_leaderboards({ARM_ALL_HIGH: high, ARM_ALL_LOW: low})
    finding = detect_cheap_arm_hallucination(leaderboards)

    assert finding["all_high_winner"] == "thorough-32b"
    assert finding["all_low_winner"] == "quickthink-7b"
    assert finding["same"] is False
    assert finding["wrongly_crowned"] == "quickthink-7b"
    assert finding["true_winner_pass_rate_all_high"] == pytest.approx(100.0)
    # quickthink-7b passes 2 of 4 tasks under the gold standard.
    assert finding["wrongly_crowned_pass_rate_all_high"] == pytest.approx(50.0)
    assert finding["wrongly_crowned_true_rank_all_high"] == 3
    assert "quickthink-7b" in finding["explanation"]
    assert "thorough-32b" in finding["explanation"]


def test_hallucination_does_not_fire_when_arms_agree():
    reference = board(*POOL)
    finding = detect_cheap_arm_hallucination(
        {"all_high": reference, "all_low": board(*POOL)}
    )

    assert finding["same"] is True
    assert finding["wrongly_crowned"] is None
    assert "No distortion" in finding["explanation"]


def test_hallucination_without_data():
    finding = detect_cheap_arm_hallucination({})

    assert finding["same"] is False
    assert finding["wrongly_crowned"] is None
    assert finding["all_high_winner"] is None
    assert "No comparison possible" in finding["explanation"]


# ---------------------------------------------------------------------------
# cost_rank_tradeoff
# ---------------------------------------------------------------------------


def test_cost_reduction_pct_math(make_record):
    baseline = [
        make_record(arm=ARM_ALL_HIGH, cost_usd=0.03, total_tokens=3000),
        make_record(arm=ARM_ALL_HIGH, cost_usd=0.02, total_tokens=2000),
    ]
    allocated = [
        make_record(arm=ARM_ALLOCATED, cost_usd=0.02, total_tokens=1500),
        make_record(arm=ARM_ALLOCATED, cost_usd=0.01, total_tokens=1500),
    ]
    tradeoff = cost_rank_tradeoff(
        {ARM_ALL_HIGH: baseline, ARM_ALLOCATED: allocated}
    )

    # baseline = 0.05 USD / 5000 tokens, allocated = 0.03 USD / 3000 tokens.
    assert tradeoff["baseline_cost_usd"] == pytest.approx(0.05)
    assert tradeoff["allocated_cost_usd"] == pytest.approx(0.03)
    assert tradeoff["cost_reduction_pct"] == pytest.approx(40.0)
    assert tradeoff["token_reduction_pct"] == pytest.approx(40.0)
    assert tradeoff["cost_saving_usd"] == pytest.approx(0.02)
    assert "40.00%" in tradeoff["interpretation"]


def test_cost_reduction_pct_zero_baseline_is_safe():
    # Both arms present but zero cost: a real (zero) saving, not a missing one.
    tradeoff = cost_rank_tradeoff({ARM_ALL_HIGH: [], ARM_ALLOCATED: []})

    assert tradeoff["cost_reduction_pct"] == 0.0
    assert tradeoff["baseline_cost_usd"] == 0.0
    assert tradeoff["baseline_available"] is True
    assert tradeoff["interpretation"]


def test_cost_reduction_is_none_when_an_arm_is_missing(make_record):
    only_allocated = {
        ARM_ALLOCATED: [make_record(arm=ARM_ALLOCATED, cost_usd=0.01, total_tokens=1000)]
    }
    tradeoff = cost_rank_tradeoff(only_allocated)

    # "No baseline produced" must never be reported as "no money saved".
    assert tradeoff["cost_reduction_pct"] is None
    assert tradeoff["token_reduction_pct"] is None
    assert tradeoff["cost_saving_usd"] is None
    assert tradeoff["baseline_available"] is False
    assert tradeoff["allocated_available"] is True
    assert "not computable" in tradeoff["interpretation"]


def test_cost_rank_tradeoff_reports_stability_change(make_record):
    high = make_model_records(
        make_record,
        arm=ARM_ALL_HIGH,
        passed_by_model={"a": 4, "b": 3, "c": 2, "d": 1},
        cost_by_model={"a": 0.02, "b": 0.02, "c": 0.02, "d": 0.02},
    )
    allocated = make_model_records(
        make_record,
        arm=ARM_ALLOCATED,
        passed_by_model={"b": 4, "a": 3, "c": 2, "d": 1},
        cost_by_model={"a": 0.005, "b": 0.005, "c": 0.005, "d": 0.005},
    )
    tradeoff = cost_rank_tradeoff(
        {ARM_ALL_HIGH: high, ARM_ALLOCATED: allocated}
    )

    assert tradeoff["cost_reduction_pct"] == pytest.approx(75.0)
    # Top-3 membership is identical, ordering is not => below 100 but still high.
    assert 0.0 < tradeoff["allocated_stability_pct"] < 100.0
    assert tradeoff["rank_stability_change_pct"] == pytest.approx(
        tradeoff["allocated_stability_pct"] - 100.0
    )


# ---------------------------------------------------------------------------
# build_cost_report
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_records_by_arm(make_record):
    """A three-arm, two-round dataset where the cheap arm flips the winner."""
    high = {}
    low = {}
    allocated = {}
    for round_index in (0, 1):
        high[round_index] = [
            make_record(arm=ARM_ALL_HIGH, candidate="thorough-32b", task_id="t0",
                        passed=True, cost_usd=0.02, total_tokens=5000,
                        round_index=round_index),
            make_record(arm=ARM_ALL_HIGH, candidate="quickthink-7b", task_id="t0",
                        passed=False, cost_usd=0.02, total_tokens=5000,
                        round_index=round_index),
        ]
        low[round_index] = [
            make_record(arm=ARM_ALL_LOW, candidate="thorough-32b", task_id="t0",
                        passed=False, cost_usd=0.002, total_tokens=800,
                        tier="low", round_index=round_index),
            make_record(arm=ARM_ALL_LOW, candidate="quickthink-7b", task_id="t0",
                        passed=True, cost_usd=0.002, total_tokens=800,
                        tier="low", round_index=round_index),
        ]
        allocated[round_index] = [
            make_record(arm=ARM_ALLOCATED, candidate="thorough-32b", task_id="t0",
                        passed=True, cost_usd=0.01, total_tokens=2500,
                        tier="mid", round_index=round_index),
            make_record(arm=ARM_ALLOCATED, candidate="quickthink-7b", task_id="t0",
                        passed=False, cost_usd=0.01, total_tokens=2500,
                        tier="mid", round_index=round_index),
        ]
    return {
        ARM_ALL_HIGH: [r for rows in high.values() for r in rows],
        ARM_ALL_LOW: [r for rows in low.values() for r in rows],
        ARM_ALLOCATED: [r for rows in allocated.values() for r in rows],
    }


def test_build_cost_report_contains_every_required_key(synthetic_records_by_arm):
    report = build_cost_report(synthetic_records_by_arm)

    required = {
        "arms",
        "rounds",
        "totals_by_arm",
        "cost_by_category",
        "leaderboards",
        "pooled_leaderboards",
        "rank_stability",
        "rank_stability_by_round",
        "per_round",
        "cheap_arm_hallucination",
        "cost_rank_tradeoff",
        "headline",
        "methodology",
        "n_records_total",
        "n_records_by_arm",
    }
    assert required.issubset(report.keys())

    # Every per-arm view covers every arm.
    assert report["arms"] == [ARM_ALL_HIGH, ARM_ALL_LOW, ARM_ALLOCATED]
    for key in ("totals_by_arm", "cost_by_category", "leaderboards",
                "pooled_leaderboards", "n_records_by_arm"):
        assert set(report[key]) == set(report["arms"]), key

    # The three leaderboards exist per round (string keys, plus the "mean" alias).
    assert report["rounds"] == [0, 1]
    for arm in report["arms"]:
        assert set(report["leaderboards"][arm]) == {"0", "1", "mean"}
        assert report["leaderboards"][arm]["0"]

    # Stability, finding and trade-off are all populated.
    assert report["rank_stability"]["reference_arm"] == REFERENCE_ARM
    assert set(report["rank_stability"]["arms"]) == set(ARMS)
    assert report["rank_stability_by_round"][0]["reference_arm"] == REFERENCE_ARM
    assert report["cheap_arm_hallucination"]["wrongly_crowned"] == "quickthink-7b"
    # 4 high records * 0.02 = 0.08 USD; 4 allocated records * 0.01 = 0.04 USD.
    assert report["cost_rank_tradeoff"]["cost_reduction_pct"] == pytest.approx(50.0)
    assert "Cost down 50.00%" in report["headline"]
    assert isinstance(report["headline"], str) and report["headline"]


def test_build_cost_report_headline_states_the_xy_claim(synthetic_records_by_arm):
    report = build_cost_report(synthetic_records_by_arm)
    headline = report["headline"]

    assert "Cost down" in headline
    assert "rank stability changes" in headline
    assert "quickthink-7b" in headline


def test_build_cost_report_is_json_serialisable(synthetic_records_by_arm):
    report = build_cost_report(synthetic_records_by_arm)
    encoded = json.dumps(report, sort_keys=True)
    assert json.loads(encoded)["headline"] == report["headline"]


def test_build_cost_report_is_deterministic(synthetic_records_by_arm):
    forward = build_cost_report(synthetic_records_by_arm)
    reversed_records = {
        arm: list(reversed(rows)) for arm, rows in synthetic_records_by_arm.items()
    }
    backward = build_cost_report(reversed_records)

    assert json.dumps(forward, sort_keys=True) == json.dumps(backward, sort_keys=True)


def test_build_cost_report_empty_input():
    report = build_cost_report({})

    assert report["arms"] == []
    assert report["rounds"] == []
    assert report["n_records_total"] == 0
    assert report["totals_by_arm"] == {}
    assert report["rank_stability"]["arms"] == {}
    assert report["cheap_arm_hallucination"]["wrongly_crowned"] is None
    assert report["cost_rank_tradeoff"]["cost_reduction_pct"] is None
    assert report["partial"] is True
    assert isinstance(report["headline"], str)
    assert "not yet available" in report["headline"]
    assert report["methodology"]


def test_build_cost_report_tolerates_a_partial_run(make_record):
    """Only `allocated` has finished: no crash, no invented cost claim."""
    partial = {
        ARM_ALLOCATED: [
            make_record(arm=ARM_ALLOCATED, candidate="model-a", cost_usd=0.01),
            make_record(arm=ARM_ALLOCATED, candidate="model-b", cost_usd=0.01),
        ]
    }
    report = build_cost_report(partial)

    assert report["arms"] == [ARM_ALLOCATED]
    assert report["partial"] is True
    assert set(report["missing_arms"]) == {ARM_ALL_HIGH, ARM_ALL_LOW}
    for key in ("totals_by_arm", "cost_by_category", "leaderboards",
                "pooled_leaderboards", "n_records_by_arm", "arm_totals"):
        assert set(report[key]) == {ARM_ALLOCATED}, key
    # Comparison keys are None, not 0.0 — a partial run claims nothing.
    assert report["cost_rank_tradeoff"]["cost_reduction_pct"] is None
    assert report["cost_rank_tradeoff"]["baseline_available"] is False
    assert report["cheap_arm_hallucination"]["same"] is False
    assert report["cheap_arm_hallucination"]["wrongly_crowned"] is None
    assert "not yet available" in report["headline"]
    # Still serialisable and renderable.
    json.dumps(report)
    assert render_summary_markdown(report)


def test_build_cost_report_is_complete_with_all_arms(synthetic_records_by_arm):
    report = build_cost_report(synthetic_records_by_arm)
    assert report["partial"] is False
    assert report["missing_arms"] == []


def test_leaderboards_key_shape_matches_report_renderer(synthetic_records_by_arm):
    """report_render reads `leaderboards[arm]["mean"]` or a list of rows."""
    report = build_cost_report(synthetic_records_by_arm)
    for arm in report["arms"]:
        arm_lb = report["leaderboards"][arm]
        assert isinstance(arm_lb, dict)
        assert "mean" in arm_lb
        assert isinstance(arm_lb["mean"], list) and arm_lb["mean"]
        # Per-round keys coexist with the "mean" alias.
        assert {"0", "1"}.issubset(arm_lb.keys())
        assert report["arm_totals"] == report["totals_by_arm"]


def test_report_render_consumes_our_report(synthetic_records_by_arm):
    """The teammate renderer must be able to consume build_cost_report output."""
    report_render = pytest.importorskip("smoltrace_budget.report_render")
    markdown = report_render.render_cost_report_markdown(
        build_cost_report(synthetic_records_by_arm)
    )
    assert "Evaluation Cost Report" in markdown
    assert "thorough-32b" in markdown
    assert "Cost of running the evaluation" in markdown
    # The category table must render real numbers, not zeros from key mismatch.
    assert "Cost distribution by task category" in markdown


def test_build_cost_report_pooled_index_is_not_a_round(synthetic_records_by_arm):
    report = build_cost_report(synthetic_records_by_arm)
    for arm in report["arms"]:
        rounds = [row["round_index"] for row in report["pooled_leaderboards"][arm]]
        assert set(rounds) == {POOLED_ROUND_INDEX}


def test_headline_stability_uses_pooled_not_round_zero(make_record):
    """The headline must come from the pooled leaderboard, not a single round."""
    by_arm = {ARM_ALL_HIGH: [], ARM_ALLOCATED: []}
    for round_index in range(3):
        for model, n_passed in (("model-a", 4), ("model-b", 2)):
            for task_index in range(4):
                # Gold standard is stable in every round.
                by_arm[ARM_ALL_HIGH].append(
                    make_record(
                        arm=ARM_ALL_HIGH,
                        candidate=model,
                        task_id=f"t{task_index}",
                        passed=task_index < n_passed,
                        cost_usd=0.02,
                        total_tokens=4000,
                        round_index=round_index,
                        tier="high",
                    )
                )
                # The allocated arm is noisy: round 1 inverts this pair only,
                # so model-b wins that round while model-a wins overall.
                allocated_passed = (
                    task_index < (4 - n_passed) if round_index == 1 else task_index < n_passed
                )
                by_arm[ARM_ALLOCATED].append(
                    make_record(
                        arm=ARM_ALLOCATED,
                        candidate=model,
                        task_id=f"t{task_index}",
                        passed=allocated_passed,
                        cost_usd=0.01,
                        total_tokens=2000,
                        round_index=round_index,
                        tier="mid",
                    )
                )

    report = build_cost_report(by_arm)
    pooled = report["rank_stability"]["arms"][ARM_ALLOCATED]
    round1 = report["per_round"]["by_round"]["1"][ARM_ALLOCATED]

    # Pooled: model-a still finishes ahead overall -> gold-standard order holds.
    assert pooled["top3_ordering_identical"] is True
    assert pooled["stability_pct"] == 100.0
    # Round 1 alone is inverted, which is exactly why we do not headline it.
    assert round1["top3_ordering_identical"] is False
    assert round1["stability_pct"] < 100.0
    # And the pooled figure is genuinely different from round 0's-only claim.
    assert pooled["stability_pct"] != round1["stability_pct"]


def test_per_round_summary_reports_spread(synthetic_records_by_arm):
    report = build_cost_report(synthetic_records_by_arm)
    per_round = report["per_round"]

    assert per_round["rounds"] == [0, 1]
    assert set(per_round["by_round"]) == {"0", "1"}
    for arm in report["arms"]:
        entry = per_round["by_arm"][arm]
        assert entry["n_rounds"] == 2
        assert len(entry["stability_pct_per_round"]) == 2
        assert entry["min_stability_pct"] == min(entry["stability_pct_per_round"])
        assert entry["max_stability_pct"] == max(entry["stability_pct_per_round"])
        assert entry["spread_pct"] == pytest.approx(
            entry["max_stability_pct"] - entry["min_stability_pct"]
        )
        assert isinstance(entry["top3_ordering_identical_every_round"], bool)
        assert isinstance(entry["consistent_across_rounds"], bool)


def test_per_round_summary_empty_when_no_rounds():
    report = build_cost_report({})
    assert report["per_round"] == {"rounds": [], "by_round": {}, "by_arm": {}}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_summary_markdown_has_all_sections(synthetic_records_by_arm):
    markdown = render_summary_markdown(build_cost_report(synthetic_records_by_arm))

    for heading in (
        "# Cost-Aware Evaluation Scheduling",
        "## Headline",
        "## Leaderboards",
        "## Cost by arm",
        "## Cost by task category",
        "## Rank stability",
        "## Which model does the cheap arm wrongly crown?",
        "## The X% / Y% trade-off",
        "## Methodology",
        "## Appendix",
    ):
        assert heading in markdown, heading
    assert "| rank | model |" in markdown
    assert "thorough-32b" in markdown
    assert "50.00%" in markdown


def test_render_summary_markdown_is_deterministic(synthetic_records_by_arm):
    report = build_cost_report(synthetic_records_by_arm)
    assert render_summary_markdown(report) == render_summary_markdown(report)


def test_render_summary_markdown_on_empty_report():
    markdown = render_summary_markdown(build_cost_report({}))
    assert "# Cost-Aware Evaluation Scheduling" in markdown


# ---------------------------------------------------------------------------
# load_artifacts
# ---------------------------------------------------------------------------


def test_load_artifacts_reads_the_runner_layout(artifacts_tmpdir, make_record):
    tmp_path = artifacts_tmpdir
    records_dir = tmp_path / "records"
    records_dir.mkdir()

    payload = [
        make_record(arm=ARM_ALL_HIGH, candidate="model-a", round_index=0).to_dict(),
        make_record(arm=ARM_ALL_HIGH, candidate="model-b", round_index=0).to_dict(),
    ]
    (records_dir / "all_high_r0.json").write_text(json.dumps(payload), encoding="utf-8")
    (records_dir / "all_low_r1.json").write_text(
        json.dumps([make_record(arm=ARM_ALL_LOW, tier="low", round_index=1).to_dict()]),
        encoding="utf-8",
    )
    # Junk that must be ignored, not crash.
    (records_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    (records_dir / "broken.json").write_text("{not json", encoding="utf-8")

    artifacts = load_artifacts(tmp_path)

    assert sorted(artifacts) == [ARM_ALL_HIGH, ARM_ALL_LOW]
    assert artifacts[ARM_ALL_HIGH][0][0].candidate_model == "model-a"
    assert isinstance(artifacts[ARM_ALL_HIGH][0][0], TaskRecord)
    assert artifacts[ARM_ALL_LOW][1][0].allocated_tier == "low"
    assert artifacts[ARM_ALL_HIGH][0][0].arm == ARM_ALL_HIGH


def test_load_artifacts_accepts_the_records_dir_itself(artifacts_tmpdir, make_record):
    tmp_path = artifacts_tmpdir
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "allocated_r2.json").write_text(
        json.dumps([make_record(arm=ARM_ALLOCATED, round_index=2).to_dict()]),
        encoding="utf-8",
    )
    artifacts = load_artifacts(records_dir)
    assert artifacts[ARM_ALLOCATED][2]


def test_load_artifacts_missing_directory_is_empty(artifacts_tmpdir):
    assert load_artifacts(artifacts_tmpdir / "does-not-exist") == {}


def test_flatten_artifacts_pools_rounds(make_record):
    artifacts = {
        ARM_ALL_HIGH: {
            0: [make_record(round_index=0)],
            1: [make_record(round_index=1), make_record(round_index=1)],
        }
    }
    flattened = flatten_artifacts(artifacts)
    assert len(flattened[ARM_ALL_HIGH]) == 3


def test_leaderboards_by_round_splits_rounds(make_record):
    records = make_model_records(make_record, passed_by_model={"a": 2}, round_index=0) + (
        make_model_records(make_record, passed_by_model={"a": 1}, round_index=1)
    )
    by_round = leaderboards_by_round({ARM_ALL_HIGH: records})[ARM_ALL_HIGH]

    assert sorted(by_round) == [0, 1]
    assert by_round[0][0].pass_rate == 50.0
    assert by_round[1][0].pass_rate == 25.0


# ---------------------------------------------------------------------------
# 05_analyze.py CLI
# ---------------------------------------------------------------------------


def _run_cli(records_dir, out_dir):
    """Invoke scripts/05_analyze.py in-process and return (code, stdout, stderr)."""
    import importlib.util
    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    # The CLI does `from _common import ...`, which works when run as a script
    # because Python puts the script's directory on sys.path. Reproduce that.
    added = str(scripts_dir) not in sys.path
    if added:
        sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "_cli_05_analyze", scripts_dir / "05_analyze.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = module.main(["--records", str(records_dir), "--out", str(out_dir)])
        return code, out.getvalue(), err.getvalue()
    finally:
        if added:
            sys.path.remove(str(scripts_dir))


def test_cli_writes_report_from_synthetic_artifacts(artifacts_tmpdir, synthetic_records_by_arm):
    records_dir = artifacts_tmpdir / "records"
    records_dir.mkdir()
    by_arm_round = {}
    for arm, records in synthetic_records_by_arm.items():
        for record in records:
            by_arm_round.setdefault((arm, record.round_index), []).append(record.to_dict())
    for (arm, round_index), rows in by_arm_round.items():
        (records_dir / f"{arm}_r{round_index}.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )

    out_dir = artifacts_tmpdir / "report"
    code, stdout, stderr = _run_cli(records_dir, out_dir)

    assert code == 0, stderr
    report_path = out_dir / "cost_report.json"
    summary_path = out_dir / "summary.md"
    assert report_path.is_file() and summary_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["headline"]
    assert "Cost down" in stdout or "Headline" in stdout
    markdown = summary_path.read_text(encoding="utf-8")
    assert "## Leaderboards" in markdown


def test_cli_missing_artifacts_exits_nonzero_cleanly(artifacts_tmpdir):
    code, stdout, stderr = _run_cli(artifacts_tmpdir / "nope", artifacts_tmpdir / "report")

    assert code != 0
    assert "not found" in stderr
    # A clear message, never a traceback.
    assert "Traceback" not in stderr


def test_cli_empty_records_dir_exits_nonzero(artifacts_tmpdir):
    records_dir = artifacts_tmpdir / "records"
    records_dir.mkdir()

    code, _stdout, stderr = _run_cli(records_dir, artifacts_tmpdir / "report")

    assert code != 0
    assert "no record artifacts" in stderr
    assert "Traceback" not in stderr
