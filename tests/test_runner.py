"""Tests for the three-arm evaluation runner.

The suite runs a deliberately tiny matrix — 3 tasks (one per category), 2
candidates, 1 round, all 3 arms = 18 ``evaluate_single_test`` calls — but it
exercises the whole real pipeline: SMOLTRACE's OTEL provider, its agent builder,
its tool-calling loop, its trace extraction, its cost calculator and its
leaderboard row computation. Nothing about the integration is mocked, so a
break in any of it fails here.

The one thing these tests *do* care a lot about is the process-global OTEL
provider (``smoltrace.otel.setup_inmemory_otel`` installs global providers, and
Python honours only the first one). Running all three arms in a single process
is exactly what would break if the runner re-created the provider per arm-round,
so the multi-arm assertions double as a regression test for that design.
"""

import json
from pathlib import Path

import pytest

from smoltrace_budget.contracts import (
    ARMS,
    TIER_LABELS,
    TIER_ORDER,
    TaskRecord,
    validate_tier,
)
from smoltrace_budget.models import CANDIDATES
from smoltrace_budget.runner import run_arm, run_experiment
from smoltrace_budget.tasks import load_tasks

#: Two candidates is enough to prove the pool is honoured and that leaderboards
#: rank more than one model; the tier-favoured pair makes the comparison
#: meaningful even on a tiny subset.
TINY_CANDIDATES = list(CANDIDATES[:2])

#: One task per category, so ``by_category`` (gaia->high, math->mid,
#: simpleqa->low) must produce three *different* tiers in the allocated arm.
TINY_CATEGORIES = ["gaia", "math", "simpleqa"]


def _tiny_tasks() -> list:
    """One real task from each of the three benchmark categories."""
    tasks = []
    for category in TINY_CATEGORIES:
        selected = load_tasks(categories=[category], limit=1)
        assert selected, f"no task available for category {category!r}"
        tasks.append(selected[0])
    return tasks


@pytest.fixture(scope="module")
def tiny_tasks():
    return _tiny_tasks()


@pytest.fixture(scope="module")
def artifact_root(tmp_path_factory):
    return tmp_path_factory.mktemp("runner-artifacts")


@pytest.fixture(scope="module")
def experiment(tiny_tasks, artifact_root):
    """One tiny three-arm, one-round experiment shared by the read-only tests."""
    return run_experiment(
        tiny_tasks,
        TINY_CANDIDATES,
        arms=ARMS,
        n_rounds=1,
        policy="by_category",
        seed=7,
        out_dir=artifact_root,
        force=True,
        progress=False,
        quiet=True,
    )


# ---------------------------------------------------------------------------
# Shape of the result
# ---------------------------------------------------------------------------


def test_experiment_covers_every_arm(experiment):
    assert [arm["arm"] for arm in experiment["arms"]] == list(ARMS)
    assert experiment["config"]["arms"] == list(ARMS)
    assert experiment["config"]["n_rounds"] == 1


def test_every_arm_evaluates_the_same_task_candidate_pairs(experiment):
    """The whole comparison rests on this: identical row sets per arm."""
    pairs_by_arm = {}
    for arm_result in experiment["arms"]:
        pairs_by_arm[arm_result["arm"]] = {
            (record["task_id"], record["candidate_model"]) for record in arm_result["records"]
        }

    reference = pairs_by_arm["all_high"]
    expected_n = len(_tiny_tasks()) * len(TINY_CANDIDATES)
    assert len(reference) == expected_n, "each arm must produce one record per (task, candidate)"

    for arm in ARMS:
        assert pairs_by_arm[arm] == reference, f"arm {arm} evaluated a different row set"


def test_records_carry_every_contract_field(experiment):
    for arm_result in experiment["arms"]:
        for payload in arm_result["records"]:
            # Round-tripping through the dataclass proves the keys are exactly
            # the contract's, with nothing missing and nothing extra.
            record = TaskRecord.from_dict(payload)
            assert record.arm == arm_result["arm"]
            assert record.round_index == 0
            assert record.task_id
            assert record.candidate_model
            assert record.steps >= 0
            assert isinstance(record.passed, bool)


def test_every_record_has_a_valid_tier(experiment):
    for arm_result in experiment["arms"]:
        for payload in arm_result["records"]:
            tier = validate_tier(payload["allocated_tier"])
            assert tier in TIER_ORDER
            # The tier's model class is what the record was priced against.
            assert payload["allocated_model"] == TIER_LABELS[tier]


def test_fixed_arms_use_their_single_tier(experiment):
    by_arm = {arm["arm"]: arm for arm in experiment["arms"]}
    assert {r["allocated_tier"] for r in by_arm["all_high"]["records"]} == {"high"}
    assert {r["allocated_tier"] for r in by_arm["all_low"]["records"]} == {"low"}


def test_allocated_arm_tiers_are_not_all_identical(experiment):
    """``by_category`` on a mixed-category subset must spread the tiers."""
    allocated = next(arm for arm in experiment["arms"] if arm["arm"] == "allocated")
    tiers = {record["allocated_tier"] for record in allocated["records"]}
    assert len(tiers) > 1, f"allocated arm collapsed to a single tier: {tiers}"


def test_tokens_are_positive_for_every_record(experiment):
    for arm_result in experiment["arms"]:
        for record in arm_result["records"]:
            assert record["total_tokens"] > 0, (
                f"{record['task_id']}/{record['candidate_model']} in {record['arm']} "
                "produced no tokens; SMOLTRACE's span accounting is not reaching us"
            )
            assert record["prompt_tokens"] >= 0
            assert record["completion_tokens"] >= 0


def test_high_tier_costs_more_than_low_tier(experiment):
    """The fidelity gradient the whole project rests on must be visible."""
    by_arm = {arm["arm"]: arm for arm in experiment["arms"]}
    high = by_arm["all_high"]["total_tokens"]
    low = by_arm["all_low"]["total_tokens"]
    assert high > low, f"high tier ({high}) did not cost more tokens than low tier ({low})"


# ---------------------------------------------------------------------------
# Leaderboard rows
# ---------------------------------------------------------------------------


def test_leaderboard_rows_exist_per_candidate_per_arm(experiment):
    for arm_result in experiment["arms"]:
        rows = arm_result["leaderboard_rows"]
        assert len(rows) == len(TINY_CANDIDATES), (
            f"{arm_result['arm']} produced {len(rows)} SMOLTRACE leaderboard rows, "
            f"expected one per candidate ({len(TINY_CANDIDATES)})"
        )
        assert {row["candidate_model_id"] for row in rows} == {
            c.model_id for c in TINY_CANDIDATES
        }
        for row in rows:
            # These are compute_leaderboard_row's own fields, not a shape of ours.
            assert "total_tokens" in row
            assert "success_rate" in row
            assert row["run_id"] == arm_result["run_id"]


def test_smoltrace_leaderboard_rows_agree_with_our_records(experiment):
    """SMOLTRACE's own aggregation and our record join must not disagree."""
    for arm_result in experiment["arms"]:
        for row in arm_result["leaderboard_rows"]:
            ours = [
                r
                for r in arm_result["records"]
                if r["candidate_model"] == row["candidate_model_id"]
            ]
            assert row["total_tokens"] == sum(r["total_tokens"] for r in ours)
            assert row["total_tests"] == len(ours)


def test_every_arm_has_a_full_ranking(experiment):
    for arm_result in experiment["arms"]:
        ranking = arm_result["ranking"]
        assert len(ranking) == len(TINY_CANDIDATES)
        assert [row["rank"] for row in ranking] == list(range(1, len(TINY_CANDIDATES) + 1))
        pass_rates = [row["pass_rate"] for row in ranking]
        assert pass_rates == sorted(pass_rates, reverse=True)


# ---------------------------------------------------------------------------
# Artifacts on disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["records", "leaderboard", "allocations", "runs"])
def test_artifacts_are_written_and_re_readable(experiment, artifact_root, kind):
    directory = artifact_root / kind
    assert directory.is_dir()
    for arm in ARMS:
        path = directory / f"{arm}_r0.json"
        assert path.is_file(), f"missing artifact {path}"
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload is not None


def test_records_artifact_matches_in_memory_records(experiment, artifact_root):
    for arm_result in experiment["arms"]:
        path = artifact_root / "records" / f"{arm_result['arm']}_r0.json"
        with open(path, "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        assert on_disk == arm_result["records"]


def test_run_metadata_records_the_token_and_cost_totals(experiment, artifact_root):
    for arm_result in experiment["arms"]:
        path = artifact_root / "runs" / f"{arm_result['arm']}_r0.json"
        with open(path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        assert meta["arm"] == arm_result["arm"]
        assert meta["round_index"] == 0
        assert meta["n_tasks"] == len(_tiny_tasks())
        assert meta["n_candidates"] == len(TINY_CANDIDATES)
        assert meta["total_tokens"] == arm_result["total_tokens"]
        assert meta["total_tokens"] > 0
        assert meta["run_id"] == arm_result["run_id"]
        # otel_run_id must be present and distinct from the per-arm-round id:
        # one process-wide provider serves every round.
        assert meta["otel_run_id"], "run metadata must record the OTEL session id"
        assert meta["tiers"], "run metadata must record the tier histogram"


def test_allocations_artifact_has_one_decision_per_task(experiment, artifact_root):
    for arm_result in experiment["arms"]:
        path = artifact_root / "allocations" / f"{arm_result['arm']}_r0.json"
        with open(path, "r", encoding="utf-8") as handle:
            allocations = json.load(handle)
        assert len(allocations) == len(_tiny_tasks())
        assert {a["task_id"] for a in allocations} == {t.id for t in _tiny_tasks()}
        for allocation in allocations:
            assert validate_tier(allocation["tier"]) in TIER_ORDER
            assert allocation["expected_tokens"] > 0


def test_experiment_summary_is_written(experiment, artifact_root):
    path = artifact_root / "runs" / "experiment.json"
    assert path.is_file()
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["config"]["arms"] == list(ARMS)
    assert summary["totals"]["total_tokens"] == experiment["totals"]["total_tokens"]
    assert len(summary["rounds"]) == len(ARMS)


# ---------------------------------------------------------------------------
# Resume / skip
# ---------------------------------------------------------------------------


def test_second_run_skips_and_reloads_the_same_round(tiny_tasks, artifact_root):
    """``force=False`` must reload artifacts instead of re-evaluating."""
    before = run_experiment(
        tiny_tasks,
        TINY_CANDIDATES,
        arms=["all_low"],
        n_rounds=1,
        policy="by_category",
        seed=7,
        out_dir=artifact_root,
        force=True,
        progress=False,
        quiet=True,
    )
    after = run_experiment(
        tiny_tasks,
        TINY_CANDIDATES,
        arms=["all_low"],
        n_rounds=1,
        policy="by_category",
        seed=7,
        out_dir=artifact_root,
        force=False,
        progress=False,
        quiet=True,
    )
    assert all(arm["skipped"] is False for arm in before["arms"])
    assert all(arm["skipped"] is True for arm in after["arms"])
    # A reloaded round is byte-identical to the one it replaced.
    assert after["arms"][0]["records"] == before["arms"][0]["records"]
    assert after["arms"][0]["total_tokens"] == before["arms"][0]["total_tokens"]


# ---------------------------------------------------------------------------
# run_arm on its own
# ---------------------------------------------------------------------------


def test_run_arm_is_usable_standalone(tiny_tasks, tmp_path):
    result = run_arm(
        "allocated",
        tiny_tasks,
        TINY_CANDIDATES,
        round_index=2,
        policy="by_category",
        seed=11,
        out_dir=tmp_path,
        progress=False,
        quiet=True,
    )
    assert result.arm == "allocated"
    assert result.round_index == 2
    assert len(result.records) == len(tiny_tasks) * len(TINY_CANDIDATES)
    assert all(record["round_index"] == 2 for record in result.records)
    assert result.total_tokens > 0
    assert (tmp_path / "records" / "allocated_r2.json").is_file()


def test_round_index_rotates_the_backend_seed(tiny_tasks, tmp_path):
    """Two rounds of the same arm must be independent measurements."""
    first = run_arm(
        "all_high",
        tiny_tasks,
        TINY_CANDIDATES,
        round_index=0,
        seed=5,
        out_dir=tmp_path,
        progress=False,
        quiet=True,
    )
    second = run_arm(
        "all_high",
        tiny_tasks,
        TINY_CANDIDATES,
        round_index=1,
        seed=5,
        out_dir=tmp_path,
        progress=False,
        quiet=True,
    )
    # Rounds stay joinable: same tasks, same candidates.
    assert {(r["task_id"], r["candidate_model"]) for r in first.records} == {
        (r["task_id"], r["candidate_model"]) for r in second.records
    }
    assert all(r["round_index"] == 0 for r in first.records)
    assert all(r["round_index"] == 1 for r in second.records)


def test_arm_rejects_unknown_names(tiny_tasks, tmp_path):
    with pytest.raises(ValueError):
        run_arm("nonexistent_arm", tiny_tasks, TINY_CANDIDATES, out_dir=tmp_path)


def test_experiment_rejects_empty_inputs(tmp_path):
    with pytest.raises(ValueError):
        run_experiment([], TINY_CANDIDATES, n_rounds=1, out_dir=tmp_path)
    with pytest.raises(ValueError):
        run_experiment(_tiny_tasks(), [], n_rounds=1, out_dir=tmp_path)
