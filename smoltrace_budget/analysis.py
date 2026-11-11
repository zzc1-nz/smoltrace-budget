"""Comparison, cost-savings, rank-stability and cost-report layer.

This module turns the runner's per-task :class:`~smoltrace_budget.contracts.TaskRecord`
stream into the four analysis outputs DESIGN.md advertises:

1. three leaderboards (``all_high`` / ``all_low`` / ``allocated``) over an
   identical model pool, ranked by pass rate;
2. cost deltas — total tokens and total USD per arm, plus the percentage the
   allocated arm saves against the ``all_high`` gold standard;
3. ranking stability — top-3 membership + ordering agreement and Kendall's
   tau-b against ``all_high``, collapsed into a single ``stability_pct``;
4. a cost report — totals, cost distribution by task category, and the headline
   finding: **which model the cheap arm wrongly crowns first**.

Everything here is pure, deterministic and network-free. It imports nothing but
the standard library and :mod:`smoltrace_budget.contracts`, so it can run long
after the runner's environment has been torn down, and every function accepts
either :class:`TaskRecord` instances or their JSON dict form, so the same code
path serves unit tests and on-disk artifacts.

Formulas (quoted verbatim in the README)
----------------------------------------

``pass_rate`` (percent, rounded to 4 dp before ranking)::

    pass_rate = 100 * passed / evaluated

Leaderboard ordering (deterministic, no exceptions)::

    sort key = (-pass_rate, total_cost_usd, model)   # cheaper wins a tie

``kendall_tau`` — Kendall's tau-b over the models present in *both* arm and
reference leaderboards::

    tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))
    n0 = n(n-1)/2          all pairs
    n1 = pairs tied on the reference ranks (including pairs tied on both)
    n2 = pairs tied on the arm ranks        (including pairs tied on both)
    C  = concordant pairs, D = discordant pairs

``tau_b`` is ``None`` when fewer than two models are common, or when the
denominator is zero (all pairs tied on one side).

``top3_membership_overlap_ratio``::

    overlap_ratio = |{top3 of arm} ∩ {top3 of reference}|
                    / min(3, |top3 of reference|, |top3 of arm|)

With the canonical 5-model pool both top-3 lists are full, so this is simply
``overlap_count / 3``. The ratio is ``0.0`` when either top-3 is empty.

``stability_pct`` (0..100)::

    tau_norm      = (tau_b + 1) / 2                  # 0..1, None if tau_b is None
    stability_pct = 100 * (0.5 * overlap_ratio + 0.5 * tau_norm)
                  = 100 * overlap_ratio              # fallback when tau_b is None
    stability_delta_pct = stability_pct - 100        # reference arm => 0.0

Half the weight is top-of-leaderboard membership (the decision a practitioner
actually acts on), half is whole-pool rank correlation (whether the overall
ordering held). The reference arm scores exactly 100 and 0.0 by construction.

``cost_reduction_pct`` against ``ARM_ALL_HIGH``::

    cost_reduction_pct = 100 * (baseline_cost - allocated_cost) / baseline_cost

and ``0.0`` when the baseline cost is zero (nothing to save).

Determinism notes
-----------------
* Record order never matters: aggregates use :func:`math.fsum` (correctly
  rounded, order independent), and every collection that reaches the report is
  sorted explicitly (arms by :data:`~smoltrace_budget.contracts.ARMS` order then
  name, categories and models lexicographically, rounds numerically).
* No timestamps, no random ids, no wall-clock data ever enter a return value, so
  two runs over the same artifacts produce byte-identical JSON.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import (
    ARM_ALL_HIGH,
    ARM_ALL_LOW,
    ARM_ALLOCATED,
    ARM_DESCRIPTIONS,
    ARMS,
    REFERENCE_ARM,
    LeaderboardEntry,
    TaskRecord,
)

__all__ = [
    "STABILITY_TOP_K",
    "RANK_CHANGES_TOP_K",
    "MEMBERSHIP_WEIGHT",
    "TAU_WEIGHT",
    "POOLED_ROUND_INDEX",
    "build_leaderboard",
    "arm_cost",
    "cost_by_category",
    "kendall_tau",
    "pooled_leaderboards",
    "leaderboards_by_round",
    "rank_stability",
    "detect_cheap_arm_hallucination",    "cost_rank_tradeoff",
    "build_cost_report",
    "load_artifacts",
    "flatten_artifacts",
    "render_summary_markdown",
]

#: Size of the top-k that feeds ``stability_pct`` and the "did the podium move?"
#: question. The README claim is about the top-3, so this is fixed at 3.
STABILITY_TOP_K: int = 3

#: How deep ``rank_changes`` looks: models appearing in the top-5 of *either*
#: leaderboard are reported, which catches near-miss podium swaps.
RANK_CHANGES_TOP_K: int = 5

#: Weights of the two ``stability_pct`` components. They must sum to 1.0.
MEMBERSHIP_WEIGHT: float = 0.5
TAU_WEIGHT: float = 0.5

#: ``round_index`` stamped on a leaderboard that pools every round together.
POOLED_ROUND_INDEX: int = -1

_DIGITS_PCT: int = 4
_DIGITS_USD: int = 6
_INF: float = float("inf")

_LEADERBOARD_FIELDS = frozenset(LeaderboardEntry.__dataclass_fields__)  # type: ignore[attr-defined]

_RECORD_FILE_RE = re.compile(r"^(?P<arm>[A-Za-z0-9_]+)_r(?P<round>\d+)\.json$")


# ---------------------------------------------------------------------------
# Coercion helpers — every public function accepts dataclasses or raw dicts
# ---------------------------------------------------------------------------


def _as_task_records(records: Optional[Iterable[Any]]) -> List[TaskRecord]:
    """Normalise ``records`` into a list of :class:`TaskRecord`."""
    out: List[TaskRecord] = []
    for record in records or ():
        if isinstance(record, TaskRecord):
            out.append(record)
        elif isinstance(record, Mapping):
            out.append(TaskRecord.from_dict(dict(record)))
        else:  # pragma: no cover - defensive
            raise TypeError(
                f"expected TaskRecord or dict, got {type(record).__name__}"
            )
    return out


def _as_leaderboard_entries(entries: Optional[Iterable[Any]]) -> List[LeaderboardEntry]:
    """Normalise leaderboard rows into :class:`LeaderboardEntry` objects."""
    out: List[LeaderboardEntry] = []
    for entry in entries or ():
        if isinstance(entry, LeaderboardEntry):
            out.append(entry)
        elif isinstance(entry, Mapping):
            payload = {k: v for k, v in entry.items() if k in _LEADERBOARD_FIELDS}
            payload.setdefault("arm", "")
            payload.setdefault("round_index", 0)
            payload.setdefault("model", "")
            payload.setdefault("passed", 0)
            payload.setdefault("evaluated", 0)
            payload.setdefault("pass_rate", 0.0)
            payload.setdefault("total_tokens", 0)
            payload.setdefault("total_cost_usd", 0.0)
            payload.setdefault("avg_latency_ms", 0.0)
            out.append(LeaderboardEntry(**payload))
        else:  # pragma: no cover - defensive
            raise TypeError(
                f"expected LeaderboardEntry or dict, got {type(entry).__name__}"
            )
    return out


def _arm_sort_key(arm: str) -> tuple:
    """Canonical arm order: ``ARMS`` order first, then lexicographic."""
    if arm in ARMS:
        return (ARMS.index(arm), arm)
    return (len(ARMS), arm)


def _coerce_records_by_arm(
    records_by_arm: Optional[Mapping[str, Iterable[Any]]],
) -> Dict[str, List[TaskRecord]]:
    """Normalise ``{arm: records}``, returning arms in canonical order."""
    source = records_by_arm or {}
    return {
        arm: _as_task_records(source[arm])
        for arm in sorted(source.keys(), key=_arm_sort_key)
    }


def _order_entries(entries: List[LeaderboardEntry]) -> List[LeaderboardEntry]:
    """Sort leaderboard rows canonically and guarantee every row has a rank.

    Rows that already carry real ranks (``rank >= 1``) keep them, so a caller's
    explicit ranking is never second-guessed; otherwise the shared tie-break
    ``(-pass_rate, total_cost_usd, model)`` is applied and ranks 1..n are
    assigned. Without this, rows from a foreign producer (or hand-built rows with
    the default ``rank = 0``) would compare as "all ranked 0" and corrupt
    ``rank_changes`` and Kendall's tau.
    """
    if entries and all(entry.rank > 0 for entry in entries):
        return sorted(entries, key=lambda entry: (entry.rank, entry.model))

    ordered = sorted(
        entries, key=lambda entry: (-entry.pass_rate, entry.total_cost_usd, entry.model)
    )
    for position, entry in enumerate(ordered, start=1):
        entry.rank = position
    return ordered


def _top_models(entries: List[LeaderboardEntry], k: int) -> List[str]:
    """The first ``k`` model names of an ordered leaderboard."""
    return [entry.model for entry in entries[:k]]


def _share_pct(part: float, whole: float) -> float:
    """``100 * part / whole``, or ``0.0`` when ``whole`` is zero."""
    if not whole:
        return 0.0
    return round(100.0 * part / whole, _DIGITS_PCT)


# ---------------------------------------------------------------------------
# 1. Leaderboards
# ---------------------------------------------------------------------------


def build_leaderboard(
    records: Iterable[TaskRecord | Mapping[str, Any]],
    arm: str,
    round_index: int,
) -> List[LeaderboardEntry]:
    """Aggregate per-task records into one arm's leaderboard.

    Every record is one evaluated ``(task, candidate_model)`` pair. For each
    ``candidate_model`` this counts how many were passed and evaluated and sums
    their tokens / cost / latency::

        pass_rate     = 100 * passed / evaluated          # percent, 4 dp
        total_tokens  = sum(total_tokens)
        total_cost    = sum(cost_usd)
        avg_latency   = sum(latency_ms) / evaluated

    Rows are ordered by ``pass_rate`` descending, then ``total_cost_usd``
    ascending, then ``model`` ascending — so a tie on measured accuracy rewards
    the cheaper model and identical rows fall back to a stable lexicographic
    order. ``rank`` is then assigned 1..n.

    Sums use :func:`math.fsum`, so the result does not depend on input order.
    ``arm`` and ``round_index`` are metadata stamped onto each row; the caller is
    responsible for passing the slice of records that belongs to them.

    Args:
        records: the :class:`TaskRecord` objects (or their dict form) for this
            arm/round.
        arm: arm name to stamp on the emitted rows.
        round_index: round number to stamp on the emitted rows.

    Returns:
        ``[]`` when there are no records.
    """
    recs = _as_task_records(records)

    buckets: Dict[str, List[TaskRecord]] = {}
    for record in recs:
        buckets.setdefault(record.candidate_model, []).append(record)

    entries: List[LeaderboardEntry] = []
    for model in sorted(buckets):
        rows = buckets[model]
        passed = sum(1 for row in rows if bool(row.passed))
        evaluated = len(rows)
        pass_rate = round(100.0 * passed / evaluated, _DIGITS_PCT) if evaluated else 0.0
        total_tokens = int(math.fsum(row.total_tokens for row in rows))
        total_cost = round(math.fsum(row.cost_usd for row in rows), _DIGITS_USD)
        avg_latency = round(math.fsum(row.latency_ms for row in rows) / evaluated, 3)
        entries.append(
            LeaderboardEntry(
                arm=arm,
                round_index=round_index,
                model=model,
                passed=passed,
                evaluated=evaluated,
                pass_rate=pass_rate,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                avg_latency_ms=avg_latency,
            )
        )

    entries.sort(key=lambda entry: (-entry.pass_rate, entry.total_cost_usd, entry.model))
    for position, entry in enumerate(entries, start=1):
        entry.rank = position
    return entries


def pooled_leaderboards(
    records_by_arm: Mapping[str, Iterable[TaskRecord | Mapping[str, Any]]],
) -> Dict[str, List[LeaderboardEntry]]:
    """One leaderboard per arm with every round pooled together.

    Rows carry ``round_index = POOLED_ROUND_INDEX`` (-1). Pooling is the most
    statistically stable view of an arm (3x the records per model) and is what
    the headline stability number is computed on.
    """
    by_arm = _coerce_records_by_arm(records_by_arm)
    return {
        arm: build_leaderboard(recs, arm, POOLED_ROUND_INDEX)
        for arm, recs in by_arm.items()
    }


def leaderboards_by_round(
    records_by_arm: Mapping[str, Iterable[TaskRecord | Mapping[str, Any]]],
) -> Dict[str, Dict[int, List[LeaderboardEntry]]]:
    """``{arm: {round: [LeaderboardEntry]}}`` for the per-round view."""
    by_arm = _coerce_records_by_arm(records_by_arm)
    out: Dict[str, Dict[int, List[LeaderboardEntry]]] = {}
    for arm, recs in by_arm.items():
        rounds: Dict[int, List[TaskRecord]] = {}
        for record in recs:
            rounds.setdefault(int(record.round_index), []).append(record)
        out[arm] = {
            round_index: build_leaderboard(rows, arm, round_index)
            for round_index, rows in sorted(rounds.items())
        }
    return out


# ---------------------------------------------------------------------------
# 2. Cost
# ---------------------------------------------------------------------------


def arm_cost(
    records: Iterable[TaskRecord | Mapping[str, Any]],
) -> Dict[str, Any]:
    """Totals for one arm.

    Returns:
        ``{"total_tokens", "total_cost_usd", "n_records", "n_tasks", "n_models"}``
        where ``n_records`` counts evaluated ``(task, model)`` pairs, ``n_tasks``
        distinct ``task_id`` values and ``n_models`` distinct ``candidate_model``
        values. ``total_cost_usd`` is rounded to 6 dp; tokens are integers.
    """
    recs = _as_task_records(records)
    return {
        "total_tokens": int(math.fsum(record.total_tokens for record in recs)),
        "total_cost_usd": round(math.fsum(record.cost_usd for record in recs), _DIGITS_USD),
        "n_records": len(recs),
        "n_tasks": len({record.task_id for record in recs}),
        "n_models": len({record.candidate_model for record in recs}),
    }


def cost_by_category(
    records: Iterable[TaskRecord | Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Cost distribution over task categories.

    For every ``category`` (sorted lexicographically)::

        tokens          = sum(total_tokens)
        cost            = sum(cost_usd)
        n_records       = number of evaluated (task, model) pairs
        share_of_tokens = 100 * tokens / arm total tokens   # percent, 4 dp
        share_of_cost   = 100 * cost   / arm total cost     # percent, 4 dp

    Shares are ``0.0`` when the arm total is zero. Returns ``{}`` for no records.

    Each entry also carries ``total_tokens`` / ``total_cost_usd`` aliases under
    the same values, because :mod:`smoltrace_budget.report_render` reads those key
    names. ``tokens`` / ``cost`` remain the canonical keys.
    """
    recs = _as_task_records(records)

    buckets: Dict[str, List[TaskRecord]] = {}
    for record in recs:
        buckets.setdefault(record.category, []).append(record)

    total_tokens = int(math.fsum(record.total_tokens for record in recs))
    total_cost = math.fsum(record.cost_usd for record in recs)

    out: Dict[str, Dict[str, Any]] = {}
    for category in sorted(buckets):
        rows = buckets[category]
        tokens = int(math.fsum(row.total_tokens for row in rows))
        cost = round(math.fsum(row.cost_usd for row in rows), _DIGITS_USD)
        out[category] = {
            "tokens": tokens,
            "cost": cost,
            "n_records": len(rows),
            "share_of_tokens": _share_pct(tokens, total_tokens),
            "share_of_cost": _share_pct(cost, total_cost),
            # Aliases consumed by smoltrace_budget.report_render.
            "total_tokens": tokens,
            "total_cost_usd": cost,
        }
    return out


# ---------------------------------------------------------------------------
# 3. Rank stability
# ---------------------------------------------------------------------------


def kendall_tau(
    x: Sequence[float], y: Sequence[float]
) -> Optional[float]:
    """Kendall's tau-b between two equally long sequences.

    Implemented directly from the definition (no scipy), with tie correction::

        tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))

    where ``n0 = n(n-1)/2`` is the number of pairs, ``n1`` the number of pairs
    tied on ``x``, ``n2`` the number of pairs tied on ``y`` (pairs tied on both
    count in both ``n1`` and ``n2``, which is the standard tau-b correction), and
    ``C`` / ``D`` the concordant / discordant pairs.

    Cases that are mathematically undefined return ``None`` instead of raising or
    silently producing ``0.0``:

    * fewer than 2 observations;
    * a zero denominator (every pair tied on ``x`` or on ``y``), e.g. both
      sequences constant.

    Args:
        x: first sequence of values (ranks or scores).
        y: second sequence, same length.

    Returns:
        ``tau_b`` rounded to 6 dp, in ``[-1.0, 1.0]``, or ``None``.

    Raises:
        ValueError: if the two sequences have different lengths.
    """
    xs = list(x)
    ys = list(y)
    if len(xs) != len(ys):
        raise ValueError(
            f"kendall_tau needs equal-length sequences, got {len(xs)} and {len(ys)}"
        )

    n = len(xs)
    if n < 2:
        return None

    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0:
                tied_x += 1
            if dy == 0:
                tied_y += 1
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1

    n0 = n * (n - 1) // 2
    denominator = math.sqrt((n0 - tied_x) * (n0 - tied_y))
    if denominator == 0:
        return None
    return round((concordant - discordant) / denominator, 6)


def rank_stability(
    leaderboards: Mapping[str, Iterable[LeaderboardEntry | Mapping[str, Any]]],
) -> Dict[str, Any]:
    """How faithfully each arm reproduces the gold-standard ranking.

    Every arm in ``leaderboards`` is compared against
    :data:`~smoltrace_budget.contracts.REFERENCE_ARM` (``all_high``). For one
    arm, with ``ref`` = reference rows and ``arm`` = this arm's rows::

        overlap        = |set(top3(ref)) ∩ set(top3(arm))|            # 0..3
        overlap_ratio  = overlap / min(3, |top3(ref)|, |top3(arm)|)   # 0..1
        tau_b          = kendall_tau(ref_rank, arm_rank) over the models
                         present in BOTH leaderboards               # or None
        tau_norm       = (tau_b + 1) / 2                              # 0..1
        stability_pct  = 100 * (0.5 * overlap_ratio + 0.5 * tau_norm)
                       = 100 * overlap_ratio     # when tau_b is None
        stability_delta = stability_pct - 100

    ``top3_ordering_identical`` is ``True`` only when the same three models
    appear in the same order. ``rank_changes`` covers every model in the top-5
    of *either* leaderboard; ``delta = reference_rank - arm_rank``, so a positive
    delta means the arm ranked that model higher than the gold standard did, and
    ``None`` marks a model missing from one side. The reference arm trivially
    scores ``stability_pct = 100.0`` and ``stability_delta_pct = 0.0``.

    Args:
        leaderboards: ``{arm: [LeaderboardEntry] or [dict]}``.

    Returns:
        ``{"reference_arm", "reference_top3", "arms": {arm: {...}},
        "mean_stability_pct"}`` — see the per-arm keys in the docstring above
        plus ``top3_membership_overlap`` (int), ``top3_membership_overlap_ratio``,
        ``n_common_models``, ``top1_matches``.
        When the reference leaderboard is missing or empty, ``"arms"`` is empty
        and ``"mean_stability_pct"`` is ``None``.
    """
    normalised: Dict[str, List[LeaderboardEntry]] = {
        arm: _order_entries(_as_leaderboard_entries(entries))
        for arm, entries in (leaderboards or {}).items()
    }

    ref_entries = normalised.get(REFERENCE_ARM, [])
    ref_rank = {entry.model: entry.rank for entry in ref_entries}
    ref_top3 = _top_models(ref_entries, STABILITY_TOP_K)

    result: Dict[str, Any] = {
        "reference_arm": REFERENCE_ARM,
        "reference_top3": ref_top3,
        "arms": {},
        "mean_stability_pct": None,
    }
    if not ref_entries:
        return result

    arm_scores: List[float] = []
    for arm in sorted(normalised.keys(), key=_arm_sort_key):
        entries = normalised[arm]
        arm_rank = {entry.model: entry.rank for entry in entries}
        arm_top3 = _top_models(entries, STABILITY_TOP_K)

        reference_set = set(ref_top3)
        overlap_count = sum(1 for model in arm_top3 if model in reference_set)
        overlap_denominator = min(STABILITY_TOP_K, len(ref_top3), len(arm_top3))
        overlap_ratio = (
            round(overlap_count / overlap_denominator, _DIGITS_PCT)
            if overlap_denominator
            else 0.0
        )
        ordering_identical = arm_top3 == ref_top3 and len(ref_top3) == len(arm_top3)
        top1_matches = bool(ref_top3) and bool(arm_top3) and ref_top3[0] == arm_top3[0]

        # Kendall tau-b over the models present in both leaderboards.
        common_models = [model for model in ref_rank if model in arm_rank]
        tau = (
            kendall_tau(
                [float(ref_rank[model]) for model in common_models],
                [float(arm_rank[model]) for model in common_models],
            )
            if len(common_models) >= 2
            else None
        )

        if tau is None:
            stability_pct = round(100.0 * overlap_ratio, _DIGITS_PCT)
            tau_norm: Optional[float] = None
        else:
            tau_norm = (tau + 1.0) / 2.0
            stability_pct = round(
                100.0
                * (MEMBERSHIP_WEIGHT * overlap_ratio + TAU_WEIGHT * tau_norm),
                _DIGITS_PCT,
            )

        # Rank movements over the top-5 of either side.
        ref_top5 = _top_models(ref_entries, RANK_CHANGES_TOP_K)
        arm_top5 = _top_models(entries, RANK_CHANGES_TOP_K)
        watched = set(ref_top5) | set(arm_top5)
        rank_changes: Dict[str, Dict[str, Optional[int]]] = {}
        for model in sorted(
            watched,
            key=lambda name: (
                ref_rank.get(name, _INF),
                arm_rank.get(name, _INF),
                name,
            ),
        ):
            reference_position = ref_rank.get(model)
            arm_position = arm_rank.get(model)
            delta = (
                reference_position - arm_position
                if reference_position is not None and arm_position is not None
                else None
            )
            rank_changes[model] = {
                "reference_rank": reference_position,
                "arm_rank": arm_position,
                "delta": delta,
            }

        stability_delta = round(stability_pct - 100.0, _DIGITS_PCT)
        result["arms"][arm] = {
            "arm": arm,
            "reference_arm": REFERENCE_ARM,
            "stability_pct": stability_pct,
            "stability_delta_pct": stability_delta,
            "top3_membership_overlap": overlap_count,
            "top3_membership_overlap_denominator": overlap_denominator,
            "top3_membership_overlap_ratio": overlap_ratio,
            "top3_ordering_identical": ordering_identical,
            "top1_matches": top1_matches,
            "kendall_tau": tau,
            "kendall_tau_normalised": (
                round(tau_norm, _DIGITS_PCT) if tau_norm is not None else None
            ),
            "n_common_models": len(common_models),
            "rank_changes": rank_changes,
            "reference_top3": ref_top3,
            "arm_top3": arm_top3,
        }
        arm_scores.append(stability_pct)

    if arm_scores:
        result["mean_stability_pct"] = round(
            math.fsum(arm_scores) / len(arm_scores), _DIGITS_PCT
        )
    return result


# ---------------------------------------------------------------------------
# 4. The headline finding
# ---------------------------------------------------------------------------


def detect_cheap_arm_hallucination(
    leaderboards: Mapping[str, Iterable[LeaderboardEntry | Mapping[str, Any]]],
    *,
    cheap_arm: str = ARM_ALL_LOW,
    reference_arm: str = REFERENCE_ARM,
) -> Dict[str, Any]:
    """Did the cheap arm crown the wrong model?

    Compares the rank-1 model of ``cheap_arm`` (default ``all_low``) with the
    rank-1 model of ``reference_arm`` (default ``all_high``, the gold standard).

    A cheap evaluation is a *measurement* of the model pool, not a different
    pool, so when the two arms disagree the cheap arm has a measurement bias
    large enough to flip the top of the leaderboard — the failure mode this
    project exists to quantify.

    Args:
        leaderboards: ``{arm: [LeaderboardEntry] or [dict]}``.
        cheap_arm: the arm under suspicion.
        reference_arm: the gold-standard arm.

    Returns:
        ``{"all_high_winner", "all_low_winner", "same", "wrongly_crowned",
        "true_winner_pass_rate_all_high", "wrongly_crowned_pass_rate_all_high",
        "explanation"}`` plus these extras: ``wrongly_crowned_true_rank_all_high``
        (rank of the wrongly crowned model under the gold standard, or ``None``),
        ``true_winner_rank_cheap``, ``true_winner_pass_rate_cheap``,
        ``cheap_arm``, ``reference_arm``.

        ``same`` is ``True`` only when both arms produced a winner *and* the two
        winners are the same model; an empty leaderboard therefore reports
        ``same = False`` with an explanation saying the data is missing.
        ``wrongly_crowned`` is ``None`` when the arms agree.
    """
    normalised: Dict[str, List[LeaderboardEntry]] = {
        arm: _order_entries(_as_leaderboard_entries(entries))
        for arm, entries in (leaderboards or {}).items()
    }

    ref_entries = normalised.get(reference_arm, [])
    cheap_entries = normalised.get(cheap_arm, [])

    ref_winner = ref_entries[0].model if ref_entries else None
    cheap_winner = cheap_entries[0].model if cheap_entries else None
    same = bool(ref_winner) and bool(cheap_winner) and ref_winner == cheap_winner

    ref_rank = {entry.model: entry.rank for entry in ref_entries}
    cheap_rank = {entry.model: entry.rank for entry in cheap_entries}
    ref_pass_rate = {entry.model: entry.pass_rate for entry in ref_entries}
    cheap_pass_rate = {entry.model: entry.pass_rate for entry in cheap_entries}

    wrongly_crowned: Optional[str] = None
    if ref_winner and cheap_winner and not same:
        wrongly_crowned = cheap_winner

    if not ref_entries or not cheap_entries:
        explanation = (
            "No comparison possible: "
            + (
                f"the reference arm ({reference_arm}) produced no leaderboard."
                if not ref_entries
                else f"the cheap arm ({cheap_arm}) produced no leaderboard."
            )
        )
    elif same:
        explanation = (
            f"No distortion at the top: the cheap arm ({cheap_arm}) crowns "
            f"{ref_winner} first, matching the {reference_arm} gold standard "
            f"({ref_pass_rate.get(ref_winner)}% pass rate)."
        )
    else:
        true_rank = ref_rank.get(wrongly_crowned) if wrongly_crowned else None
        true_rate = ref_pass_rate.get(wrongly_crowned) if wrongly_crowned else None
        crowned_rate = cheap_pass_rate.get(wrongly_crowned) if wrongly_crowned else None
        position = f"#{true_rank} of {len(ref_entries)}" if true_rank else "unranked"
        under = f"at {true_rate}%" if true_rate is not None else "not present"
        explanation = (
            f"Cheap evaluation wrongly crowns {wrongly_crowned} first "
            f"({crowned_rate}% pass rate under {cheap_arm}), but under the "
            f"{reference_arm} gold standard {wrongly_crowned} is {position} "
            f"{under}; the true winner is {ref_winner} at "
            f"{ref_pass_rate.get(ref_winner)}%. Measuring every task cheaply "
            "flips the top of the leaderboard."
        )

    return {
        "cheap_arm": cheap_arm,
        "reference_arm": reference_arm,
        "all_high_winner": ref_winner,
        "all_low_winner": cheap_winner,
        "same": same,
        "wrongly_crowned": wrongly_crowned,
        "true_winner_pass_rate_all_high": (
            ref_pass_rate.get(ref_winner) if ref_winner else None
        ),
        "wrongly_crowned_pass_rate_all_high": (
            ref_pass_rate.get(wrongly_crowned) if wrongly_crowned else None
        ),
        "wrongly_crowned_true_rank_all_high": (
            ref_rank.get(wrongly_crowned) if wrongly_crowned else None
        ),
        "true_winner_rank_cheap": (
            cheap_rank.get(ref_winner) if ref_winner else None
        ),
        "true_winner_pass_rate_cheap": (
            cheap_pass_rate.get(ref_winner) if ref_winner else None
        ),
        "n_models_reference": len(ref_entries),
        "n_models_cheap": len(cheap_entries),
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# 5. The X% / Y% trade-off
# ---------------------------------------------------------------------------


def cost_rank_tradeoff(
    records_by_arm: Mapping[str, Iterable[TaskRecord | Mapping[str, Any]]],
    *,
    leaderboards: Optional[
        Mapping[str, Iterable[LeaderboardEntry | Mapping[str, Any]]]
    ] = None,
    baseline_arm: str = ARM_ALL_HIGH,
    allocated_arm: str = ARM_ALLOCATED,
) -> Dict[str, Any]:
    """The headline trade-off: how much money the allocated arm saves, and what
    that costs in ranking fidelity.

    Cost side (against ``baseline_arm``, default ``ARM_ALL_HIGH``)::

        cost_reduction_pct  = 100 * (baseline_cost  - allocated_cost)  / baseline_cost
        token_reduction_pct = 100 * (baseline_tokens - allocated_tokens) / baseline_tokens

    Ranking side::

        rank_stability_change_pct = stability_pct(allocated) - 100

    i.e. ``stability_delta_pct`` from :func:`rank_stability`, where the reference
    arm is 100 by definition. Both reduction percentages are ``0.0`` when the
    baseline total is zero.

    Args:
        records_by_arm: ``{arm: [TaskRecord] or [dict]}``.
        leaderboards: pre-computed leaderboards to read stability from. When
            omitted, they are rebuilt from ``records_by_arm`` with all rounds
            pooled, so the reported pair always comes from a single source.
        baseline_arm: gold-standard arm.
        allocated_arm: the arm whose savings are measured.

    Returns:
        ``{"cost_reduction_pct", "baseline_cost_usd", "allocated_cost_usd",
        "baseline_tokens", "allocated_tokens", "rank_stability_change_pct",
        "interpretation"}`` plus ``token_reduction_pct``, ``allocated_stability_pct``,
        ``baseline_arm``, ``allocated_arm``, ``cost_saving_usd``.

        **Missing arms.** ``cost_reduction_pct``, ``token_reduction_pct`` and
        ``cost_saving_usd`` are ``None`` — not ``0.0`` — when either arm is
        absent from ``records_by_arm``, because "no baseline was produced yet" is
        not the same claim as "no money was saved". ``0.0`` is still returned
        when both arms are present but their totals are zero. The
        ``baseline_available`` / ``allocated_available`` flags say which case
        applied.
    """
    by_arm = _coerce_records_by_arm(records_by_arm)
    baseline_available = baseline_arm in by_arm
    allocated_available = allocated_arm in by_arm
    baseline = arm_cost(by_arm.get(baseline_arm, []))
    allocated = arm_cost(by_arm.get(allocated_arm, []))

    baseline_cost = float(baseline["total_cost_usd"])
    allocated_cost = float(allocated["total_cost_usd"])
    baseline_tokens = int(baseline["total_tokens"])
    allocated_tokens = int(allocated["total_tokens"])
    comparable = baseline_available and allocated_available

    cost_reduction_pct: Optional[float] = None
    token_reduction_pct: Optional[float] = None
    cost_saving_usd: Optional[float] = None
    if comparable:
        cost_reduction_pct = round(
            100.0 * (baseline_cost - allocated_cost) / baseline_cost, _DIGITS_PCT
        ) if baseline_cost else 0.0
        token_reduction_pct = round(
            100.0 * (baseline_tokens - allocated_tokens) / baseline_tokens, _DIGITS_PCT
        ) if baseline_tokens else 0.0
        cost_saving_usd = round(baseline_cost - allocated_cost, _DIGITS_USD)

    boards = leaderboards
    if boards is None:
        boards = pooled_leaderboards(by_arm)
    stability = rank_stability(boards)
    allocated_stability = stability["arms"].get(allocated_arm)
    rank_stability_change_pct = (
        float(allocated_stability["stability_delta_pct"])
        if allocated_stability
        else 0.0
    )

    if not comparable:
        missing = [
            arm
            for arm, present in (
                (baseline_arm, baseline_available),
                (allocated_arm, allocated_available),
            )
            if not present
        ]
        interpretation = (
            f"Cost trade-off not computable yet: no records for "
            f"{', '.join(missing)}. Rank stability against {baseline_arm} is "
            f"reported once every arm has produced artifacts."
            if not allocated_stability
            else (
                f"Cost trade-off not computable yet: no records for "
                f"{', '.join(missing)}. No cost claim is made; the stability "
                f"figure above covers only the arms that did run."
            )
        )
    else:
        direction = "cuts" if cost_reduction_pct >= 0 else "raises"
        if rank_stability_change_pct == 0:
            stability_clause = "leaves top-3 rank stability unchanged"
        else:
            stability_clause = (
                f"moves top-3 rank stability by "
                f"{rank_stability_change_pct:+.2f}%"
            )
        interpretation = (
            f"The allocated arm {direction} evaluation cost by "
            f"{abs(cost_reduction_pct):.2f}% versus the {baseline_arm} gold standard "
            f"({baseline_cost:.4f} USD / {baseline_tokens:,} tokens -> "
            f"{allocated_cost:.4f} USD / {allocated_tokens:,} tokens) and "
            f"{stability_clause}."
        )

    return {
        "baseline_arm": baseline_arm,
        "allocated_arm": allocated_arm,
        "baseline_available": baseline_available,
        "allocated_available": allocated_available,
        "cost_reduction_pct": cost_reduction_pct,
        "token_reduction_pct": token_reduction_pct,
        "baseline_cost_usd": round(baseline_cost, _DIGITS_USD),
        "allocated_cost_usd": round(allocated_cost, _DIGITS_USD),
        "cost_saving_usd": cost_saving_usd,
        "baseline_tokens": baseline_tokens,
        "allocated_tokens": allocated_tokens,
        "rank_stability_change_pct": rank_stability_change_pct,
        "allocated_stability_pct": (
            float(allocated_stability["stability_pct"]) if allocated_stability else 0.0
        ),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 6. The whole report
# ---------------------------------------------------------------------------

_METHODOLOGY: Dict[str, str] = {
    "pass_rate": "100 * passed / evaluated, rounded to 4 dp before ranking",
    "leaderboard_order": (
        "pass_rate descending, then total_cost_usd ascending, then model name "
        "ascending (deterministic, no exceptions)"
    ),
    "arm_cost": (
        "total_tokens and total_cost_usd are math.fsum over the arm's records, "
        "so they are independent of record order"
    ),
    "kendall_tau": (
        "Kendall tau-b = (C - D) / sqrt((n0 - n1) * (n0 - n2)), n0 = n(n-1)/2, "
        "n1/n2 = pairs tied on each side; None for <2 common models or a zero "
        "denominator"
    ),
    "top3_membership_overlap_ratio": (
        "|top3(arm) ∩ top3(reference)| / min(3, len(top3(reference)), "
        "len(top3(arm)))"
    ),
    "stability_pct": (
        "100 * (0.5 * top3_membership_overlap_ratio + 0.5 * (kendall_tau + 1)/2); "
        "falls back to 100 * overlap_ratio when kendall_tau is undefined"
    ),
    "stability_delta_pct": "stability_pct - 100 (reference arm is 0.0 by definition)",
    "cost_reduction_pct": (
        "100 * (baseline_cost_usd - allocated_cost_usd) / baseline_cost_usd "
        "against ARM_ALL_HIGH"
    ),
}


def _summarise_per_round(
    stability_by_round: Mapping[int, Mapping[str, Any]],
    arm_order: Sequence[str],
) -> Dict[str, Any]:
    """Compact per-round roll-up of the rank-stability analysis.

    The headline stability figure is computed on the **pooled** 3-round
    leaderboard (see :func:`build_cost_report`), which is far less noisy than any
    single round. This helper answers the follow-up question — *is that pooled
    finding a real property of the arm, or one lucky draw?*

    For each arm it reports every round's ``stability_pct``, the spread across
    rounds, and whether the top-3 ordering was identical in every round. A
    ``spread_pct`` of exactly ``0.0`` with
    ``top3_ordering_identical_every_round = True`` means the finding does not
    depend on which round you look at.

    Returns ``{"rounds": [int], "by_round": {"<round>": {arm: {...}}},
    "by_arm": {arm: {...}}}``; ``rounds`` is empty when no round had records.
    """
    if not stability_by_round:
        return {"rounds": [], "by_round": {}, "by_arm": {}}

    rounds = sorted(int(round_index) for round_index in stability_by_round)

    def _entry(round_index: int, arm: str) -> Dict[str, Any]:
        payload = stability_by_round.get(round_index) or {}
        return (payload.get("arms") or {}).get(arm) or {}

    by_arm: Dict[str, Any] = {}
    for arm in arm_order:
        values: List[float] = []
        ordering_flags: List[bool] = []
        overlaps: List[int] = []
        for round_index in rounds:
            entry = _entry(round_index, arm)
            if not entry:
                continue
            values.append(float(entry.get("stability_pct") or 0.0))
            ordering_flags.append(bool(entry.get("top3_ordering_identical")))
            overlaps.append(int(entry.get("top3_membership_overlap") or 0))
        if not values:
            continue
        by_arm[arm] = {
            "n_rounds": len(values),
            "stability_pct_per_round": [round(value, _DIGITS_PCT) for value in values],
            "min_stability_pct": round(min(values), _DIGITS_PCT),
            "max_stability_pct": round(max(values), _DIGITS_PCT),
            "spread_pct": round(max(values) - min(values), _DIGITS_PCT),
            "mean_stability_pct": round(math.fsum(values) / len(values), _DIGITS_PCT),
            "top3_membership_overlap_per_round": overlaps,
            "top3_ordering_identical_every_round": all(ordering_flags),
            "consistent_across_rounds": (max(values) - min(values)) == 0.0,
        }

    by_round: Dict[str, Dict[str, Any]] = {}
    for round_index in rounds:
        by_round[str(round_index)] = {
            arm: {
                "stability_pct": _entry(round_index, arm).get("stability_pct"),
                "kendall_tau": _entry(round_index, arm).get("kendall_tau"),
                "top3_membership_overlap": _entry(round_index, arm).get(
                    "top3_membership_overlap"
                ),
                "top3_ordering_identical": _entry(round_index, arm).get(
                    "top3_ordering_identical"
                ),
            }
            for arm in arm_order
            if _entry(round_index, arm)
        }

    return {"rounds": rounds, "by_round": by_round, "by_arm": by_arm}


def build_cost_report(
    records_by_arm: Mapping[str, Iterable[TaskRecord | Mapping[str, Any]]],
    *,
    arms: Optional[Sequence[str]] = None,
    reference_arm: str = REFERENCE_ARM,
    cheap_arm: str = ARM_ALL_LOW,
    allocated_arm: str = ARM_ALLOCATED,
) -> Dict[str, Any]:
    """Assemble the JSON-ready analysis document.

    Everything the README quotes is derived here from one consistent pass over
    the records, so the headline numbers can never disagree with the tables.

    Returns:
        A dict with these keys (all JSON-serialisable, no timestamps):

        ``arms``
            arm names present, in canonical order.
        ``rounds``
            round indices present, ascending.
        ``n_records_total`` / ``n_records_by_arm``
            record counts.
        ``totals_by_arm``
            :func:`arm_cost` per arm.
        ``cost_by_category``
            :func:`cost_by_category` per arm.
        ``leaderboards``
            ``{arm: {str(round): [row dict], "mean": [row dict]}}`` — the three
            leaderboards per round, plus a pooled ``"mean"`` entry that
            :mod:`smoltrace_budget.report_render` reads. Round keys are strings
            so the document is uniformly JSON-keyed.
        ``pooled_leaderboards``
            ``{arm: [row dict]}`` with all rounds pooled.
        ``rank_stability``
            :func:`rank_stability` over the **pooled** leaderboard (all rounds
            combined), never a single round — this is the headline figure and it
            is the least noisy estimate of the arm's true ranking fidelity.
        ``rank_stability_by_round``
            ``{round: rank_stability result}``, the raw per-round analyses.
        ``per_round``
            compact per-round roll-up (:func:`_summarise_per_round`): each arm's
            per-round ``stability_pct``, the spread across rounds, and whether
            the top-3 ordering was identical every round. Use it to show the
            pooled finding is not an artifact of one draw.
        ``cheap_arm_hallucination``
            :func:`detect_cheap_arm_hallucination` over the pooled leaderboards.
        ``cost_rank_tradeoff``
            :func:`cost_rank_tradeoff`, measured against ``reference_arm``.
        ``headline``
            one sentence assembling the X% / Y% claim.
        ``methodology``
            the exact formulas, as strings, for the README.

        **Partial arms.** Callers may pass a subset of :data:`ARMS` (for example
        while a run is still in progress). Every per-arm mapping then contains
        only the arms that have records — nothing is invented for a missing arm
        and no comparison raises. Keys whose baseline is missing are ``None``
        rather than ``0.0``: ``cost_rank_tradeoff["cost_reduction_pct"]``,
        ``["token_reduction_pct"]`` and ``["cost_saving_usd"]``; the
        ``baseline_available`` / ``allocated_available`` flags say why. The
        ``headline`` states that the figure is not yet available instead of
        claiming a saving.

        **Renderer compatibility.** For :mod:`smoltrace_budget.report_render`
        the following aliases are also present: ``arm_totals`` (== ``totals_by_arm``)
        and, per arm, ``leaderboards[arm]["mean"]`` (the pooled rows) alongside
        the per-round keys, so ``leaderboards[arm]`` is both
        ``{round: rows} | {"mean": rows}`` and usable as the renderer expects.
    """
    by_arm = _coerce_records_by_arm(records_by_arm)
    if arms is None:
        arm_order = sorted(by_arm.keys(), key=_arm_sort_key)
    else:
        arm_order = [arm for arm in arms if arm in by_arm]

    rounds = sorted({int(record.round_index) for recs in by_arm.values() for record in recs})

    per_round: Dict[str, Dict[int, List[LeaderboardEntry]]] = {
        arm: leaderboards_by_round({arm: by_arm.get(arm, [])}).get(arm, {})
        for arm in arm_order
    }
    pooled = {
        arm: build_leaderboard(by_arm.get(arm, []), arm, POOLED_ROUND_INDEX)
        for arm in arm_order
    }

    stability = rank_stability(pooled)
    stability_by_round: Dict[int, Dict[str, Any]] = {}
    for round_index in rounds:
        stability_by_round[round_index] = rank_stability(
            {arm: per_round[arm].get(round_index, []) for arm in arm_order}
        )
    tradeoff = cost_rank_tradeoff(
        by_arm,
        leaderboards=pooled,
        baseline_arm=reference_arm,
        allocated_arm=allocated_arm,
    )
    hallucination = detect_cheap_arm_hallucination(
        pooled, cheap_arm=cheap_arm, reference_arm=reference_arm
    )

    missing_arms = [arm for arm in (reference_arm, allocated_arm, cheap_arm) if arm not in by_arm]
    if tradeoff["cost_reduction_pct"] is None:
        # A partial run must never publish an X% claim it has not measured.
        headline = (
            f"Cost reduction vs the {reference_arm} gold standard is not yet "
            f"available ({', '.join(missing_arms)} produced no records)."
        )
    else:
        reduction = tradeoff["cost_reduction_pct"]
        # Phrase the direction rather than printing a signed number after "down":
        # a small positive value is still a saving, and a negative one is an
        # increase, so "Cost down -0.01%" would misread in both directions.
        if reduction > 0.005:
            cost_phrase = f"Cost down {reduction:.2f}%"
        elif reduction < -0.005:
            cost_phrase = f"Cost up {abs(reduction):.2f}%"
        else:
            cost_phrase = "Cost unchanged"
        headline = (
            f"{cost_phrase}, top-3 rank stability "
            f"changes {tradeoff['rank_stability_change_pct']:+.2f}% versus the "
            f"{reference_arm} gold standard."
        )
        if hallucination["wrongly_crowned"]:
            headline += (
                f" Measured cheaply, the leaderboard wrongly crowns "
                f"{hallucination['wrongly_crowned']} first."
            )
        elif hallucination["same"]:
            headline += " The cheap and gold-standard arms agree on the winner."
        else:
            headline += (
                f" The {cheap_arm} arm produced no leaderboard, so the "
                "cheap-vs-gold finding is pending."
            )

    leaderboards: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for arm in arm_order:
        per_arm: Dict[str, List[Dict[str, Any]]] = {
            str(round_index): [entry.to_dict() for entry in per_round[arm][round_index]]
            for round_index in sorted(per_round[arm])
        }
        # Renderer alias: `mean` is the pooled (all-rounds) leaderboard. Round
        # keys are strings so the dict is uniformly JSON-keyed and sortable.
        per_arm["mean"] = [entry.to_dict() for entry in pooled[arm]]
        leaderboards[arm] = per_arm

    return {
        "arms": arm_order,
        "rounds": rounds,
        "reference_arm": reference_arm,
        "cheap_arm": cheap_arm,
        "allocated_arm": allocated_arm,
        "partial": bool(missing_arms),
        "missing_arms": missing_arms,
        "n_records_total": sum(len(recs) for recs in by_arm.values()),
        "n_records_by_arm": {arm: len(by_arm.get(arm, [])) for arm in arm_order},
        "totals_by_arm": {arm: arm_cost(by_arm.get(arm, [])) for arm in arm_order},
        "cost_by_category": {
            arm: cost_by_category(by_arm.get(arm, [])) for arm in arm_order
        },
        "leaderboards": leaderboards,
        "pooled_leaderboards": {
            arm: [entry.to_dict() for entry in pooled[arm]] for arm in arm_order
        },
        "rank_stability": stability,
        "rank_stability_by_round": stability_by_round,
        "per_round": _summarise_per_round(stability_by_round, arm_order),
        "cheap_arm_hallucination": hallucination,
        "cost_rank_tradeoff": tradeoff,
        "headline": headline,
        # Renderer alias for `totals_by_arm` (see report_render.render_cost_table).
        "arm_totals": {arm: arm_cost(by_arm.get(arm, [])) for arm in arm_order},
        "methodology": dict(_METHODOLOGY),
    }


# ---------------------------------------------------------------------------
# 7. Artifact loading
# ---------------------------------------------------------------------------

_RECORD_LIST_KEYS = ("records", "task_records", "rows", "data", "items")


def _records_from_payload(payload: Any) -> List[Any]:
    """Extract a list of record payloads from a JSON document.

    The runner writes a bare list, but tolerating a wrapper object keeps the
    analysis layer working if that ever changes.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in _RECORD_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # A single record as a bare object.
        if "candidate_model" in payload and "task_id" in payload:
            return [payload]
    return []


def load_artifacts(
    artifacts_dir: Path | str,
) -> Dict[str, Dict[int, List[TaskRecord]]]:
    """Read the runner's ``artifacts/records/<arm>_r<round>.json`` layout.

    Each file holds a list of :class:`TaskRecord` dicts for one arm and one
    round. Files that do not match ``<arm>_r<round>.json`` are ignored.

    Args:
        artifacts_dir: either the artifacts root (which contains a ``records/``
            directory) or the ``records/`` directory itself.

    Returns:
        ``{arm: {round_index: [TaskRecord]}}`` with arms in canonical order and
        rounds ascending. ``{}`` when no record files exist (or when
        ``artifacts_dir`` does not exist) — never raises for a missing directory.
    """
    root = Path(artifacts_dir)
    records_dir = root / "records" if (root / "records").is_dir() else root
    if not records_dir.is_dir():
        return {}

    found: Dict[str, Dict[int, List[TaskRecord]]] = {}
    for path in sorted(records_dir.glob("*.json")):
        match = _RECORD_FILE_RE.match(path.name)
        if not match:
            continue
        arm = match.group("arm")
        round_index = int(match.group("round"))
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        records = _as_task_records(_records_from_payload(payload))
        found.setdefault(arm, {})[round_index] = records

    return {
        arm: {round_index: found[arm][round_index] for round_index in sorted(found[arm])}
        for arm in sorted(found.keys(), key=_arm_sort_key)
    }


def flatten_artifacts(
    artifacts: Mapping[str, Mapping[int, Iterable[TaskRecord | Mapping[str, Any]]]],
) -> Dict[str, List[TaskRecord]]:
    """``{arm: {round: [TaskRecord]}}`` -> ``{arm: [TaskRecord]}`` (all rounds)."""
    out: Dict[str, List[TaskRecord]] = {}
    for arm in sorted(artifacts.keys(), key=_arm_sort_key):
        rows: List[TaskRecord] = []
        for round_index in sorted(artifacts[arm]):
            rows.extend(_as_task_records(artifacts[arm][round_index]))
        out[arm] = rows
    return out


# ---------------------------------------------------------------------------
# 8. Human-readable rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int) -> str:
    """Format a number for markdown, or an em dash for ``None``."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Render a markdown table as a list of lines."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def render_summary_markdown(report: Mapping[str, Any]) -> str:
    """Render a :func:`build_cost_report` document as a human-readable README.

    Sections: headline, per-arm leaderboards (pooled across rounds), cost table,
    cost-by-category breakdown, rank-stability table, the cheap-arm finding, the
    X%/Y% trade-off paragraph, the exact formulas, and an appendix of per-round
    leaderboards.

    The rendering is deterministic: identical reports produce identical
    markdown, with no timestamps.
    """
    lines: List[str] = []
    arms: List[str] = list(report.get("arms") or [])
    rounds: List[int] = list(report.get("rounds") or [])
    reference_arm = report.get("reference_arm", REFERENCE_ARM)
    totals = report.get("totals_by_arm", {})
    tradeoff = report.get("cost_rank_tradeoff", {})
    hallucination = report.get("cheap_arm_hallucination", {})
    stability = report.get("rank_stability", {})

    lines.append("# Cost-Aware Evaluation Scheduling — Analysis Report")
    lines.append("")
    lines.append(
        f"Arms: {', '.join(f'`{arm}`' for arm in arms) or '—'} · "
        f"Rounds: {len(rounds)} · "
        f"Records: {_fmt_int(report.get('n_records_total'))} · "
        f"Gold standard: `{reference_arm}`"
    )
    lines.append("")

    # --- headline ---------------------------------------------------------
    lines.append("## Headline")
    lines.append("")
    lines.append(f"**{report.get('headline', '')}**")
    lines.append("")
    if hallucination.get("explanation"):
        lines.append(hallucination["explanation"])
        lines.append("")

    # --- leaderboards -----------------------------------------------------
    lines.append("## Leaderboards")
    lines.append("")
    lines.append(
        "Ranked by pass rate over an identical model pool. Ordering is "
        "`pass_rate` descending, then `total_cost_usd` ascending, then model "
        "name; rounds are pooled here, per-round tables are in the appendix."
    )
    lines.append("")
    pooled = report.get("pooled_leaderboards", {})
    for arm in arms:
        description = ARM_DESCRIPTIONS.get(arm, "")
        title = f"### `{arm}`" + (f" — {description}" if description else "")
        lines.append(title)
        lines.append("")
        rows = pooled.get(arm) or []
        if not rows:
            lines.append("_No records for this arm._")
            lines.append("")
            continue
        lines.extend(
            _md_table(
                [
                    "rank",
                    "model",
                    "pass rate %",
                    "passed / evaluated",
                    "tokens",
                    "cost USD",
                    "avg latency ms",
                ],
                [
                    [
                        str(row.get("rank", "")),
                        f"`{row.get('model', '')}`",
                        _fmt(row.get("pass_rate"), 2),
                        f"{row.get('passed', 0)} / {row.get('evaluated', 0)}",
                        _fmt_int(row.get("total_tokens")),
                        _fmt(row.get("total_cost_usd"), 4),
                        _fmt(row.get("avg_latency_ms"), 1),
                    ]
                    for row in rows
                ],
            )
        )
        lines.append("")

    # --- cost table -------------------------------------------------------
    lines.append("## Cost by arm")
    lines.append("")
    baseline_cost = float(tradeoff.get("baseline_cost_usd") or 0.0)
    baseline_tokens = float(tradeoff.get("baseline_tokens") or 0)
    cost_rows: List[List[str]] = []
    for arm in arms:
        arm_totals = totals.get(arm, {})
        cost = float(arm_totals.get("total_cost_usd") or 0.0)
        tokens = int(arm_totals.get("total_tokens") or 0)
        cost_rows.append(
            [
                f"`{arm}`",
                _fmt_int(arm_totals.get("n_records")),
                _fmt_int(arm_totals.get("n_tasks")),
                _fmt_int(arm_totals.get("n_models")),
                _fmt_int(tokens),
                _fmt(cost, 4),
                _fmt(_share_pct(cost, baseline_cost), 2) + " %",
                _fmt(_share_pct(tokens, baseline_tokens), 2) + " %",
            ]
        )
    lines.extend(
        _md_table(
            [
                "arm",
                "records",
                "tasks",
                "models",
                "total tokens",
                "total cost USD",
                "cost vs all_high",
                "tokens vs all_high",
            ],
            cost_rows,
        )
    )
    lines.append("")

    # --- category breakdown ----------------------------------------------
    lines.append("## Cost by task category")
    lines.append("")
    category_rows: List[List[str]] = []
    by_category = report.get("cost_by_category", {})
    for arm in arms:
        for category in sorted((by_category.get(arm) or {}).keys()):
            entry = by_category[arm][category]
            category_rows.append(
                [
                    f"`{arm}`",
                    category,
                    _fmt_int(entry.get("n_records")),
                    _fmt_int(entry.get("tokens")),
                    _fmt(entry.get("share_of_tokens"), 2) + " %",
                    _fmt(entry.get("cost"), 4),
                    _fmt(entry.get("share_of_cost"), 2) + " %",
                ]
            )
    if category_rows:
        lines.extend(
            _md_table(
                [
                    "arm",
                    "category",
                    "records",
                    "tokens",
                    "token share",
                    "cost USD",
                    "cost share",
                ],
                category_rows,
            )
        )
    else:
        lines.append("_No category data._")
    lines.append("")

    # --- rank stability ---------------------------------------------------
    lines.append("## Rank stability vs the gold standard")
    lines.append("")
    stability_rows: List[List[str]] = []
    stability_arms = (stability or {}).get("arms", {})
    for arm in arms:
        entry = stability_arms.get(arm)
        if not entry:
            continue
        stability_rows.append(
            [
                f"`{arm}`",
                _fmt(entry.get("stability_pct"), 2),
                f"{float(entry.get('stability_delta_pct') or 0.0):+.2f}",
                f"{entry.get('top3_membership_overlap', 0)}"
                f"/{entry.get('top3_membership_overlap_denominator', STABILITY_TOP_K)}",
                _fmt(entry.get("top3_membership_overlap_ratio"), 3),
                _fmt(entry.get("kendall_tau"), 3),
                _fmt(entry.get("top3_ordering_identical"), 0),
                ", ".join(f"`{m}`" for m in entry.get("arm_top3") or []) or "—",
            ]
        )
    if stability_rows:
        lines.append(
            "Stability is computed on the **pooled** leaderboard (all rounds "
            "combined), which is far less noisy than any single round."
        )
        lines.append("")
        lines.extend(
            _md_table(
                [
                    "arm",
                    "stability %",
                    "delta vs 100",
                    "top-3 overlap",
                    "overlap ratio",
                    "Kendall tau-b",
                    "same order",
                    "top-3",
                ],
                stability_rows,
            )
        )

        # Is the pooled finding a real property, or one lucky draw?
        per_round = report.get("per_round") or {}
        per_arm = per_round.get("by_arm") or {}
        if per_arm:
            lines.append("")
            lines.append("### Is the finding stable across rounds?")
            lines.append("")
            lines.append(
                "Each round produces its own leaderboard and its own stability "
                "score; a pooled number is only trustworthy if the rounds agree."
            )
            lines.append("")
            lines.extend(
                _md_table(
                    [
                        "arm",
                        "rounds",
                        "stability % per round",
                        "min",
                        "max",
                        "spread",
                        "same top-3 order every round",
                    ],
                    [
                        [
                            f"`{arm}`",
                            str(entry.get("n_rounds", 0)),
                            ", ".join(
                                f"{value:.2f}"
                                for value in entry.get("stability_pct_per_round") or []
                            )
                            or "—",
                            _fmt(entry.get("min_stability_pct"), 2),
                            _fmt(entry.get("max_stability_pct"), 2),
                            _fmt(entry.get("spread_pct"), 2),
                            _fmt(entry.get("top3_ordering_identical_every_round"), 0),
                        ]
                        for arm, entry in sorted(per_arm.items())
                    ],
                )
            )
    else:
        lines.append("_No stability data._")
    lines.append("")

    # --- cheap arm finding -----------------------------------------------
    lines.append("## Which model does the cheap arm wrongly crown?")
    lines.append("")
    if hallucination.get("all_high_winner") or hallucination.get("all_low_winner"):
        lines.append(
            f"- gold standard (`{reference_arm}`) winner: "
            f"**{hallucination.get('all_high_winner')}** at "
            f"{_fmt(hallucination.get('true_winner_pass_rate_all_high'), 2)}% "
            "pass rate"
        )
        lines.append(
            f"- cheap arm (`{hallucination.get('cheap_arm')}`) winner: "
            f"**{hallucination.get('all_low_winner')}**"
        )
        if hallucination.get("wrongly_crowned"):
            lines.append(
                f"- wrongly crowned: **{hallucination.get('wrongly_crowned')}** — "
                f"true rank under the gold standard: "
                f"{hallucination.get('wrongly_crowned_true_rank_all_high')} "
                f"({_fmt(hallucination.get('wrongly_crowned_pass_rate_all_high'), 2)}% "
                "pass rate)"
            )
        lines.append("")
        lines.append(hallucination.get("explanation", ""))
    else:
        lines.append("_No leaderboard data._")
    lines.append("")

    # --- trade-off --------------------------------------------------------
    lines.append("## The X% / Y% trade-off")
    lines.append("")
    lines.append(tradeoff.get("interpretation", ""))
    lines.append("")
    cost_reduction = tradeoff.get("cost_reduction_pct")
    token_reduction = tradeoff.get("token_reduction_pct")
    lines.append(
        f"- cost reduction vs `{reference_arm}`: "
        + (
            "**not yet available** (arm missing from this run)"
            if cost_reduction is None
            else f"**{_fmt(cost_reduction, 2)}%** "
            f"({_fmt(tradeoff.get('baseline_cost_usd'), 4)} USD -> "
            f"{_fmt(tradeoff.get('allocated_cost_usd'), 4)} USD)"
        )
    )
    lines.append(
        f"- token reduction vs `{reference_arm}`: "
        + (
            "**not yet available** (arm missing from this run)"
            if token_reduction is None
            else f"**{_fmt(token_reduction, 2)}%** "
            f"({_fmt_int(tradeoff.get('baseline_tokens'))} -> "
            f"{_fmt_int(tradeoff.get('allocated_tokens'))} tokens)"
        )
    )
    lines.append(
        f"- top-3 rank stability change: "
        f"**{float(tradeoff.get('rank_stability_change_pct') or 0.0):+.2f}%** "
        f"(allocated arm scores "
        f"{_fmt(tradeoff.get('allocated_stability_pct'), 2)}/100)"
    )
    lines.append("")

    # --- methodology ------------------------------------------------------
    lines.append("## Methodology")
    lines.append("")
    methodology = report.get("methodology") or {}
    for key in sorted(methodology):
        lines.append(f"- **{key}**: {methodology[key]}")
    lines.append("")

    # --- appendix ---------------------------------------------------------
    lines.append("## Appendix — per-round leaderboards")
    lines.append("")
    per_round = report.get("leaderboards", {})
    for arm in arms:
        arm_rounds = per_round.get(arm) or {}
        # `leaderboards[arm]` also carries the renderer's "mean" alias, which is
        # not a round; only integer-ish keys are per-round tables.
        round_keys = sorted(
            (key for key in arm_rounds if str(key).isdigit()), key=lambda key: int(key)
        )
        for round_index in round_keys:
            rows = arm_rounds[round_index] or arm_rounds.get(str(round_index)) or []
            lines.append(f"### `{arm}` — round {round_index}")
            lines.append("")
            if not rows:
                lines.append("_No records._")
                lines.append("")
                continue
            lines.extend(
                _md_table(
                    ["rank", "model", "pass rate %", "passed / evaluated", "tokens", "cost USD"],
                    [
                        [
                            str(row.get("rank", "")),
                            f"`{row.get('model', '')}`",
                            _fmt(row.get("pass_rate"), 2),
                            f"{row.get('passed', 0)} / {row.get('evaluated', 0)}",
                            _fmt_int(row.get("total_tokens")),
                            _fmt(row.get("total_cost_usd"), 4),
                        ]
                        for row in rows
                    ],
                )
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
