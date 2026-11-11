"""Evaluation runner: drives SMOLTRACE's real pipeline for 3 arms x N rounds.

This module is the orchestrator of the whole experiment. It does **not**
reimplement any part of SMOLTRACE: agents are built with
``smoltrace.core.initialize_agent``, tasks are executed with
``smoltrace.core.evaluate_single_test``, tokens/costs come from
``smoltrace.core.extract_traces`` (i.e. from SMOLTRACE's own OTEL span
accounting, priced by its cost calculator), metrics from
``smoltrace.core.extract_metrics``, and leaderboard rows from
``smoltrace.utils.compute_leaderboard_row``.

What the runner adds on top of SMOLTRACE is the capability SMOLTRACE lacks: a
**per-task, per-candidate model injection**. SMOLTRACE evaluates one model with
one fidelity configuration for a whole run; here every ``(task, candidate)``
pair is measured at the fidelity tier the allocator chose for that task, which
is what makes the three arms comparable on an identical row set.


The three arms
--------------

``all_high``
    Every task at the ``high`` tier. The gold standard.
``all_low``
    Every task at the ``low`` tier. The cheapest possible measurement.
``allocated``
    ``smoltrace_budget.allocator`` assigns a tier per task under the chosen
    policy (``by_category`` / ``by_pass_rate`` / ``greedy_budget``).

All three arms evaluate the *same* ``(task, candidate)`` pairs, so their
leaderboards rank the same models over the same tasks and can be compared
directly.


One OTEL provider per process, one span window per arm-round
------------------------------------------------------------

``smoltrace.otel.setup_inmemory_otel`` installs **global** providers
(``trace.set_tracer_provider`` / ``metrics.set_meter_provider``). Python honours
only the *first* global tracer provider set in a process, so calling it again
returns a fresh ``span_exporter`` that will never receive another span:

    >>> tracer1, _, exp1, *_ = setup_inmemory_otel(enable_otel=True, run_id="A")
    >>> tracer2, _, exp2, *_ = setup_inmemory_otel(enable_otel=True, run_id="B")
    >>> assert trace.get_tracer_provider() is trace.get_tracer_provider()  # same
    >>> # spans emitted through tracer2 land in exp1, never in exp2

This was verified empirically on this machine (SMOLTRACE 0.2.12 /
opentelemetry-sdk 1.45.2) before writing this module, and it is the reason for
the design below:

* the OTEL session is created **once per process** (``_ensure_otel``);
* each arm-round gets a monotonically advancing **span window** over that one
  exporter (``_SpanWindow``) so ``extract_traces`` only ever sees the spans of
  the round being closed;
* each arm-round still passes its **own** ``run_id`` to ``extract_traces``, so
  trace documents carry the round's identity and stay separable downstream.

Traces are grouped by ``trace_id``, not by ``run_id``, so a single provider is
sufficient: one ``evaluate_single_test`` call emits exactly one root
``test_evaluation`` span and therefore exactly one trace.

The exporter is drained after every round so a 5940-call run does not retain
~90k span dictionaries.


Performance notes
-----------------

The full matrix is 132 tasks x 5 candidates x 3 arms x 3 rounds = 5940 calls,
each a real smolagents ``ToolCallingAgent`` loop. To keep that in the minutes
range:

* agents are cached per ``(candidate, agent_type, tier)`` — the per-task model is
  swapped into the cached agent (``agent.model = ...``) instead of rebuilding
  the agent and its tool set every call;
* the tool set is therefore built at most ``n_candidates * n_tiers`` times per
  process rather than once per task;
* smolagents' console logging is disabled at agent construction
  (``verbosity_level=-1``) instead of being filtered per call;
* the OTEL provider, its processors and its cost calculator are created once.

"""

from __future__ import annotations

import io
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import ARTIFACTS_DIR
from .contracts import (
    ARM_ALL_HIGH,
    ARM_ALL_LOW,
    ARM_ALLOCATED,
    ARMS,
    TIER_LABELS,
    TIER_ORDER,
    TIER_SPEC,
    Allocation,
    LeaderboardEntry,
    ModelProfile,
    TaskRecord,
    TaskSpec,
    validate_tier,
)
from .models import CANDIDATES

__all__ = [
    "DEFAULT_AGENT_TYPE",
    "ArmResult",
    "run_arm",
    "run_experiment",
    "allocate_tiers",
    "load_round_artifacts",
    "experiment_totals",
]

#: The agent loop the simulated backend speaks. It emits
#: ``ToolCallingAgent``-compatible ``ChatMessage`` tool calls, so every task —
#: including the handful declared ``agent_type: "code"`` — is measured through
#: the same tool-calling loop. Using one agent type for the whole matrix keeps
#: the three arms measuring an identical estimator; SMOLTRACE's grader reads the
#: same ``expected_tool`` / ``expected_keywords`` fields either way.
DEFAULT_AGENT_TYPE = "tool"

#: SMOLTRACE dataset names recorded on the leaderboard rows.
DATASET_USED = "kshitijthakkar/smoltrace-benchmark-v1"
PROVIDER = "litellm"
#: Pinned submitter. ``compute_leaderboard_row`` otherwise calls the HF API to
#: resolve ``$HF_TOKEN`` into a username — a network round-trip per row.
SUBMITTED_BY = "smoltrace-budget"

#: Progress lines are printed every this many evaluated ``(task, candidate)``
#: pairs (there are ``n_candidates`` pairs per task).
PROGRESS_EVERY = 20

#: artifact sub-directories written under ``artifacts/``
RECORDS_DIR = "records"
LEADERBOARD_DIR = "leaderboard"
ALLOCATIONS_DIR = "allocations"
RUNS_DIR = "runs"


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
# Importing smoltrace pulls in smolagents, opentelemetry and the cost tables
# (~7 s on this machine). Keeping those imports lazy means ``import
# smoltrace_budget.runner`` stays instant, which keeps test collection fast and
# lets modules that only need the contracts import it safely.


def _smoltrace():
    """Return the SMOLTRACE modules the runner drives."""
    import smoltrace.core as core
    import smoltrace.otel as otel
    import smoltrace.utils as utils

    return core, otel, utils


def _allocator_mod():
    from . import allocator

    return allocator


def _backend_mod():
    from . import sim_backend

    return sim_backend


# ---------------------------------------------------------------------------
# Console noise control
# ---------------------------------------------------------------------------
# SMOLTRACE's trace extraction prints one line per priced span
# (``[POST-CALC] ...``) and its metric exporter prints on every export. Both are
# stdout-only and would drown a 5940-call run, so they are silenced unless the
# caller asked for verbose output.


class _Quiet:
    """Redirect stdout to a sink for the duration of the block."""

    __slots__ = ("enabled", "_sink", "_saved")

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._sink: Optional[io.StringIO] = None
        self._saved = None

    def __enter__(self) -> "_Quiet":
        if self.enabled:
            self._saved = sys.stdout
            self._sink = io.StringIO()
            sys.stdout = self._sink
        return self

    def __exit__(self, *exc_info) -> bool:
        if self.enabled and self._saved is not None:
            sys.stdout = self._saved
        return False


# ---------------------------------------------------------------------------
# OTEL session: one provider per process, one span window per arm-round
# ---------------------------------------------------------------------------


class _SpanWindow:
    """A snapshot of the spans one arm-round emitted.

    SMOLTRACE's ``extract_traces(span_exporter, run_id)`` only calls
    ``span_exporter.get_finished_spans()``, so this adapter is a drop-in
    stand-in that exposes exactly one arm-round's spans out of the process-wide
    exporter. It does not mutate anything.

    The snapshot is taken eagerly: a live slice would silently start returning
    ``[]`` if a later round advanced the cursor, and an empty trace list is
    exactly the failure this module is built to make impossible.
    """

    __slots__ = ("_spans",)

    def __init__(self, exporter: Any, start: int = 0, end: Optional[int] = None):
        spans = list(exporter.get_finished_spans())
        self._spans = spans[max(0, int(start)) : None if end is None else int(end)]

    def get_finished_spans(self) -> List[Dict[str, Any]]:
        return self._spans

    def __len__(self) -> int:
        return len(self._spans)


@dataclass
class _OtelSession:
    """The single process-wide in-memory OTEL session."""

    tracer: Any
    meter: Any
    span_exporter: Any
    metric_exporter: Any
    aggregator: Any
    service_name: str
    #: the ``run_id`` SMOLTRACE stamped on the provider's resource attributes.
    #: One per process, because only the first provider set is honoured.
    otel_run_id: str
    enabled: bool = True
    #: index of the first span belonging to the next arm-round
    cursor: int = 0

    def begin_round(self) -> int:
        """Start a new span window and return its starting index."""
        _drain_spans(self.span_exporter)
        self.cursor = 0
        return self.cursor

    def close_round(self) -> None:
        """Advance the cursor past everything the round emitted."""
        self.cursor = len(self.span_exporter.get_finished_spans())

    def window(self) -> _SpanWindow:
        return _SpanWindow(self.span_exporter, self.cursor)


_OTEL_SESSION: Optional[_OtelSession] = None


def _drain_spans(exporter: Any) -> None:
    """Drop the spans the exporter holds.

    ``InMemorySpanExporter`` exposes no ``clear()``; it keeps finished spans in a
    plain list. Clearing it in place (rather than replacing it) keeps the
    exporter's own reference valid, and keeps a 5940-call run from retaining
    ~90k span dicts. Purely a memory bound: correctness never depends on it,
    because ``_SpanWindow`` already slices by index.
    """
    spans = getattr(exporter, "_spans", None)
    if isinstance(spans, list):
        spans.clear()


def _ensure_otel(service_name: str = "smoltrace-budget", *, reset: bool = False) -> _OtelSession:
    """Return the process-wide in-memory OTEL session, creating it once.

    ``smoltrace.otel.setup_inmemory_otel`` does two things that come apart when
    something else in the process has already installed a global provider:

    * ``trace.set_tracer_provider(new_provider)`` — a **no-op** after the first
      call in a process;
    * ``tracer = trace.get_tracer(...)`` — resolved against the **live global**
      provider, which may therefore *not* be ``new_provider``;
    * the returned ``span_exporter`` is attached to ``new_provider`` only.

    So when another component (a test, a host application, a previous library)
    initialised OTel first, the returned tracer emits into the *global*
    provider's exporters while the returned exporter stays permanently empty —
    and ``extract_traces`` then yields no traces and zero tokens, silently.

    The fix is to attach our own sink to whichever provider actually records,
    rather than trusting the exporter handed back by ``setup_inmemory_otel``.
    This keeps working whether or not this process was the first to set up OTel.
    """
    global _OTEL_SESSION
    if _OTEL_SESSION is not None and not reset:
        return _OTEL_SESSION

    _, otel, _ = _smoltrace()
    tracer, meter, _returned_exporter, metric_exporter, aggregator, otel_run_id = (
        otel.setup_inmemory_otel(
            enable_otel=True,
            service_name=service_name,
            run_id=f"smoltrace-budget-{uuid.uuid4().hex[:12]}",
            enable_gpu_metrics=False,
        )
    )

    span_exporter = _attach_exporter_to_live_provider(otel)

    session = _OtelSession(
        tracer=tracer,
        meter=meter,
        span_exporter=span_exporter,
        metric_exporter=metric_exporter,
        aggregator=aggregator,
        service_name=service_name,
        otel_run_id=str(otel_run_id or ""),
        enabled=all(x is not None for x in (tracer, span_exporter)),
    )
    _OTEL_SESSION = session
    return session


def _attach_exporter_to_live_provider(otel: Any) -> Any:
    """Attach a fresh in-memory span sink to the provider that actually records.

    ``SimpleSpanProcessor`` is synchronous, so spans are in the exporter by the
    time ``evaluate_single_test`` returns — which is what the span-window
    accounting depends on.
    """
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = otel.InMemorySpanExporter()
    provider = otel_trace.get_tracer_provider()
    add_processor = getattr(provider, "add_span_processor", None)
    if callable(add_processor):
        add_processor(SimpleSpanProcessor(exporter))
    return exporter


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def _round_seed(seed: int, round_index: int) -> int:
    """A deterministic per-round seed.

    Kept for callers that want one scalar to identify a round. The backend
    itself takes ``round_index`` as a first-class input (it salts its outcome
    draw with it), so :func:`_evaluate_round` passes the round explicitly rather
    than relying on this rotation alone.
    """
    return (int(seed) * 1_000_003 + int(round_index) * 7919) & 0x7FFFFFFF


def _stage_for_arm(arm: str) -> str:
    """The backend ``stage`` label for an arm.

    ``sim_backend`` salts its outcome draw with the stage precisely so that each
    arm is an *independent measurement* of the same model pool. Sharing one
    stage across arms would give the allocated arm an artificially perfect rank
    correlation with the gold standard and hide the stability loss we report.
    """
    return f"arm:{arm}"


def allocate_tiers(
    arm: str,
    tasks: Sequence[TaskSpec],
    *,
    policy: str = "by_category",
    token_budget: Optional[int] = None,
    history: Optional[Dict[str, Any]] = None,
) -> List[Allocation]:
    """Resolve the tier every task is measured at in ``arm``.

    For the two fixed-fidelity arms the decision is trivial, but an
    :class:`~smoltrace_budget.contracts.Allocation` is still produced for every
    task: it carries the expected token outlay and an explanation, and it gives
    the analysis layer one uniform input shape for all three arms.

    For ``allocated`` the decision comes from
    ``smoltrace_budget.allocator.make_allocator(...).allocate(tasks)``.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    allocator = _allocator_mod()
    estimate = allocator.estimate_task_tokens

    if arm == ARM_ALL_HIGH:
        return [
            Allocation(
                task_id=t.id,
                tier="high",
                policy="fixed_all_high",
                expected_tokens=float(estimate(t, "high")),
                reason="all_high arm: every task measured at full fidelity",
            )
            for t in tasks
        ]

    if arm == ARM_ALL_LOW:
        return [
            Allocation(
                task_id=t.id,
                tier="low",
                policy="fixed_all_low",
                expected_tokens=float(estimate(t, "low")),
                reason="all_low arm: every task measured at the cheapest fidelity",
            )
            for t in tasks
        ]

    kwargs: Dict[str, Any] = {}
    if token_budget is not None:
        kwargs["token_budget"] = token_budget
    if history is not None:
        kwargs["history"] = history
    budget_allocator = allocator.make_allocator(policy, **kwargs)
    allocations = list(budget_allocator.allocate(tasks))

    by_id = {a.task_id: a for a in allocations}
    missing = [t.id for t in tasks if t.id not in by_id]
    if missing:
        raise RuntimeError(
            f"allocator returned no decision for {len(missing)} task(s), e.g. {missing[:3]}"
        )
    for task in tasks:
        validate_tier(by_id[task.id].tier)
    return [by_id[t.id] for t in tasks]


# ---------------------------------------------------------------------------
# Agent pool
# ---------------------------------------------------------------------------


@dataclass
class _AgentSlot:
    """A cached agent for one ``(candidate, agent_type, tier)`` combination."""

    agent: Any
    agent_type: str
    tier: str
    builds: int = 0


class _AgentPool:
    """Caches one agent per ``(candidate, agent_type, tier)``.

    The tool set is built by ``initialize_agent`` and never changes, so an agent
    is reusable for every task measured at its tier; only the injected model
    differs per (task, candidate). Swapping ``agent.model`` is what keeps the
    per-call overhead near the cost of the smolagents step loop itself.

    The model is injected by the caller: the simulated backend's behaviour
    depends on the task (its correctness draw and its token profile both do), so
    a fresh model per ``(task, candidate)`` is required, but the *agent* around
    it is not.
    """

    def __init__(self) -> None:
        self._slots: Dict[Tuple[str, str, str], _AgentSlot] = {}
        self.builds = 0

    @property
    def size(self) -> int:
        return len(self._slots)

    def get(self, candidate: ModelProfile, agent_type: str, tier: str, model: Any):
        """Return the slot for this key, constructing the agent on first use."""
        key = (candidate.model_id, agent_type, tier)
        slot = self._slots.get(key)
        if slot is not None:
            self.bind(slot, model)
            return slot
        core, _, _ = _smoltrace()
        prompt_config = {
            "max_steps": int(TIER_SPEC[tier]["max_steps"]),
            # Silence smolagents' rich console at construction: a 5940-call run
            # would otherwise print a banner per task.
            "verbosity_level": -1,
        }
        agent = core.initialize_agent(
            model_name=candidate.api_model_id,
            agent_type=agent_type,
            provider=PROVIDER,
            model_instance=model,
            prompt_config=prompt_config,
        )
        slot = _AgentSlot(agent=agent, agent_type=agent_type, tier=tier, builds=1)
        self._slots[key] = slot
        self.builds += 1
        return slot

    @staticmethod
    def bind(slot: _AgentSlot, model: Any) -> None:
        """Inject this task's model into the cached agent."""
        slot.agent.model = model
        # Keep smolagents' token monitor pointed at the live model.
        monitor = getattr(slot.agent, "monitor", None)
        if monitor is not None:
            monitor.tracked_model = model


# ---------------------------------------------------------------------------
# Per-round result
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """Everything one ``(arm, round)`` produced."""

    arm: str
    round_index: int
    run_id: str
    otel_run_id: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    leaderboard_rows: List[Dict[str, Any]] = field(default_factory=list)
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    allocations: List[Dict[str, Any]] = field(default_factory=list)
    paths: Dict[str, str] = field(default_factory=dict)
    wall_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    n_tasks: int = 0
    n_candidates: int = 0
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _round_run_id(arm: str, round_index: int) -> str:
    """A unique, human-readable id per arm-round (also the OTEL ``run_id``)."""
    return f"{arm}-r{round_index}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------


def _artifacts_root(out_dir: Optional[Path | str] = None) -> Path:
    return Path(out_dir) if out_dir is not None else Path(ARTIFACTS_DIR)


def _round_paths(out_dir: Optional[Path | str], arm: str, round_index: int) -> Dict[str, Path]:
    root = _artifacts_root(out_dir)
    stem = f"{arm}_r{round_index}"
    return {
        "records": root / RECORDS_DIR / f"{stem}.json",
        "leaderboard": root / LEADERBOARD_DIR / f"{stem}.json",
        "allocations": root / ALLOCATIONS_DIR / f"{stem}.json",
        "run": root / RUNS_DIR / f"{stem}.json",
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _rank_candidates(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank the candidate pool over one arm-round's records.

    Ordering matches ``smoltrace_budget.analysis.build_leaderboard`` so the two
    layers cannot disagree: pass rate descending, then total cost ascending,
    then model name for a deterministic tie-break.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for record in records:
        model = record["candidate_model"]
        entry = stats.setdefault(
            model,
            {
                "model": model,
                "passed": 0,
                "evaluated": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "latency_sum": 0.0,
            },
        )
        entry["evaluated"] += 1
        entry["passed"] += 1 if record["passed"] else 0
        entry["total_tokens"] += int(record["total_tokens"])
        entry["total_cost_usd"] += float(record["cost_usd"])
        entry["latency_sum"] += float(record["latency_ms"])

    rows = []
    for model, entry in stats.items():
        evaluated = entry["evaluated"]
        rows.append(
            {
                "model": model,
                "passed": entry["passed"],
                "evaluated": evaluated,
                "pass_rate": round(100.0 * entry["passed"] / evaluated, 4) if evaluated else 0.0,
                "total_tokens": entry["total_tokens"],
                "total_cost_usd": round(entry["total_cost_usd"], 6),
                "avg_latency_ms": round(entry["latency_sum"] / evaluated, 4) if evaluated else 0.0,
            }
        )
    rows.sort(key=lambda r: (-r["pass_rate"], r["total_cost_usd"], r["model"]))
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows


def _leaderboard_entry(
    arm: str, round_index: int, ranked: Dict[str, Any]
) -> LeaderboardEntry:
    return LeaderboardEntry(
        arm=arm,
        round_index=round_index,
        model=ranked["model"],
        passed=ranked["passed"],
        evaluated=ranked["evaluated"],
        pass_rate=ranked["pass_rate"],
        total_tokens=ranked["total_tokens"],
        total_cost_usd=ranked["total_cost_usd"],
        avg_latency_ms=ranked["avg_latency_ms"],
        rank=ranked["rank"],
    )


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def _index_traces(traces: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Index extracted traces by ``trace_id`` and by ``test.case_uid``."""
    by_trace: Dict[str, Dict[str, Any]] = {}
    by_uid: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        trace_id = trace.get("trace_id")
        if trace_id:
            by_trace[trace_id] = trace
        for uid in trace.get("test_case_uids") or []:
            by_uid.setdefault(uid, trace)
    return by_trace, by_uid


def _build_record(
    *,
    arm: str,
    round_index: int,
    task: TaskSpec,
    candidate: ModelProfile,
    tier: str,
    agent_type: str,
    result: Dict[str, Any],
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Join one SMOLTRACE result with its trace into a :class:`TaskRecord`."""
    trace = trace or {}
    total_tokens = int(trace.get("total_tokens") or 0)
    record = TaskRecord(
        round_index=round_index,
        arm=arm,
        task_id=task.id,
        category=task.category,
        difficulty=task.difficulty,
        agent_type=agent_type,
        candidate_model=candidate.model_id,
        allocated_tier=tier,
        allocated_model=TIER_LABELS[tier],
        prompt_tokens=int(trace.get("total_prompt_tokens") or 0),
        completion_tokens=int(trace.get("total_completion_tokens") or 0),
        total_tokens=total_tokens,
        latency_ms=float(trace.get("total_duration_ms") or 0.0),
        steps=int(result.get("steps") or 0),
        passed=bool(result.get("success")),
        cost_usd=float(trace.get("total_cost_usd") or 0.0),
        stop_reason=result.get("stop_reason"),
        error=result.get("error"),
        trace_id=result.get("trace_id"),
    )
    return record.to_dict()


# ---------------------------------------------------------------------------
# The round itself
# ---------------------------------------------------------------------------


def _evaluate_round(
    *,
    arm: str,
    round_index: int,
    tasks: Sequence[TaskSpec],
    candidates: Sequence[ModelProfile],
    tier_by_task: Dict[str, str],
    agent_type: str,
    run_id: str,
    seed: int,
    pool: _AgentPool,
    session: _OtelSession,
    progress: bool = True,
    quiet: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate every ``(task, candidate)`` pair of one arm-round.

    Returns ``(records, accounting)``: the :class:`TaskRecord` dicts for the
    round, plus the raw trace/result material the leaderboard step needs (the
    raw SMOLTRACE results grouped by candidate, and the traces indexed by
    ``trace_id`` so each candidate can be handed only its own).
    """
    core, _, _ = _smoltrace()
    backend = _backend_mod()

    results_by_candidate: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    #: (candidate, task, tier, result) in evaluation order, used to build records
    ordered: List[Tuple[ModelProfile, TaskSpec, str, Dict[str, Any]]] = []

    total_pairs = len(tasks) * len(candidates)
    done = 0
    started = time.perf_counter()

    for task in tasks:
        tier = tier_by_task[task.id]
        payload = task.to_dict()
        for candidate in candidates:
            model = backend.build_model(
                candidate,
                tier,
                _stage_for_arm(arm),
                seed=seed,
                task=task,
                round_index=round_index,
            )
            slot = pool.get(candidate, agent_type, tier, model)

            with _Quiet(enabled=quiet):
                result = core.evaluate_single_test(
                    slot.agent,
                    dict(payload),
                    agent_type,
                    session.tracer,
                    session.meter,
                    verbose=False,
                )

            ordered.append((candidate, task, tier, result))
            grouped = results_by_candidate.setdefault(candidate.model_id, {"tool": [], "code": []})
            grouped.setdefault(agent_type, []).append(result)

            done += 1
            if progress and (done % PROGRESS_EVERY == 0 or done == total_pairs):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [{arm} r{round_index}] {done}/{total_pairs} pairs "
                    f"({elapsed:.1f}s, {rate:.0f} pairs/s)",
                    flush=True,
                )

    # --- close the span window and let SMOLTRACE account for the round -------
    with _Quiet(enabled=quiet):
        trace_data = core.extract_traces(session.window(), run_id)
    by_trace, by_uid = _index_traces(trace_data)

    records: List[Dict[str, Any]] = []
    trace_ids_by_candidate: Dict[str, List[str]] = {}
    for candidate, task, tier, result in ordered:
        trace = by_trace.get(result.get("trace_id") or "")
        if trace is None:
            # Defensive: the exporter kept no span for this task (e.g. the agent
            # raised before the root span ended). Fall back to the per-execution
            # uid SMOLTRACE promotes onto every trace document.
            trace = by_uid.get(f"{agent_type}:{task.id}")
        record = _build_record(
            arm=arm,
            round_index=round_index,
            task=task,
            candidate=candidate,
            tier=tier,
            agent_type=agent_type,
            result=result,
            trace=trace,
        )
        records.append(record)
        if record["trace_id"]:
            trace_ids_by_candidate.setdefault(candidate.model_id, []).append(record["trace_id"])

    session.close_round()
    return records, {
        "trace_data": trace_data,
        "by_trace": by_trace,
        "results_by_candidate": results_by_candidate,
        "trace_ids_by_candidate": trace_ids_by_candidate,
    }


def _leaderboard_rows(
    *,
    arm: str,
    round_index: int,
    candidates: Sequence[ModelProfile],
    run_id: str,
    accounting: Dict[str, Any],
    quiet: bool = True,
) -> List[Dict[str, Any]]:
    """Ask SMOLTRACE for one leaderboard row per candidate.

    ``compute_leaderboard_row`` aggregates whatever traces and results it is
    given, so each candidate is handed only its own: the per-candidate rows then
    carry that candidate's real token, cost and latency totals.
    """
    _, _, utils = _smoltrace()
    trace_data = accounting["trace_data"]
    by_trace = accounting["by_trace"]
    results_by_candidate = accounting["results_by_candidate"]
    trace_ids_by_candidate = accounting["trace_ids_by_candidate"]

    rows: List[Dict[str, Any]] = []
    # Metric aggregates are computed over the whole round, so they are only
    # meaningful for a single-candidate run; they are passed through for
    # completeness and are identical for every candidate of a round.
    with _Quiet(enabled=quiet):
        metric_data = _extract_round_metrics(
            trace_data=trace_data,
            results_by_candidate=results_by_candidate,
            run_id=run_id,
        )

    for candidate in candidates:
        candidate_traces = [
            by_trace[tid] for tid in trace_ids_by_candidate.get(candidate.model_id, []) if tid in by_trace
        ]
        candidate_results = results_by_candidate.get(candidate.model_id, {"tool": [], "code": []})
        n_results = sum(len(v) for v in candidate_results.values())
        if not n_results:
            continue

        with _Quiet(enabled=quiet):
            row = utils.compute_leaderboard_row(
                model_name=candidate.api_model_id,
                all_results=candidate_results,
                trace_data=candidate_traces,
                metric_data=metric_data,
                dataset_used=DATASET_USED,
                results_dataset=f"smoltrace-budget/{arm}/results",
                traces_dataset=f"smoltrace-budget/{arm}/traces",
                metrics_dataset=f"smoltrace-budget/{arm}/metrics",
                agent_type=DEFAULT_AGENT_TYPE,
                run_id=run_id,
                provider=PROVIDER,
                hardware=_hardware(),
                # Pinned, so compute_leaderboard_row never falls back to an
                # HF network lookup for the submitter name.
                submitted_by=SUBMITTED_BY,
            )
        row["candidate_model_id"] = candidate.model_id
        row["arm"] = arm
        row["round_index"] = round_index
        rows.append(row)
    return rows


_ROUND_METRICS_CACHE: Dict[str, Dict[str, Any]] = {}


def _extract_round_metrics(
    *, trace_data: List[Dict[str, Any]], results_by_candidate: Dict[str, Any], run_id: str
) -> Dict[str, Any]:
    """``extract_metrics`` once per round, memoised by ``run_id``."""
    cached = _ROUND_METRICS_CACHE.get(run_id)
    if cached is not None:
        return cached
    core, _, _ = _smoltrace()
    session = _ensure_otel()
    merged: Dict[str, List[Dict[str, Any]]] = {"tool": [], "code": []}
    for grouped in results_by_candidate.values():
        for agent_type, results in grouped.items():
            merged.setdefault(agent_type, []).extend(results)
    metrics = core.extract_metrics(
        session.metric_exporter, session.aggregator, trace_data, merged, run_id
    )
    _ROUND_METRICS_CACHE[run_id] = metrics
    return metrics


_HARDWARE: Optional[Dict[str, Any]] = None


def _hardware() -> Dict[str, Any]:
    """Detect this machine once and reuse it on every leaderboard row."""
    global _HARDWARE
    if _HARDWARE is None:
        from smoltrace.hardware import detect_hardware

        _HARDWARE = detect_hardware(PROVIDER, "smoltrace-budget")
    return _HARDWARE


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_arm(
    arm: str,
    tasks: Sequence[TaskSpec],
    candidates: Sequence[ModelProfile],
    *,
    round_index: int = 0,
    policy: str = "by_category",
    token_budget: Optional[int] = None,
    seed: int = 0,
    n_rounds: int = 1,
    history: Optional[Dict[str, Any]] = None,
    agent_type: str = DEFAULT_AGENT_TYPE,
    out_dir: Optional[Path | str] = None,
    force: bool = False,
    progress: bool = True,
    quiet: bool = True,
) -> ArmResult:
    """Run one arm for one round and persist its artifacts.

    Args:
        arm: one of ``"all_high"``, ``"all_low"``, ``"allocated"``.
        tasks: the task manifest slice to evaluate.
        candidates: the leaderboard pool. Every arm must be given the same pool
            so the leaderboards rank an identical row set.
        round_index: which repetition this is; it is recorded on every record and
            rotates the backend seed so rounds are independent measurements.
        policy: allocator policy, used only by the ``allocated`` arm.
        token_budget: optional explicit budget for ``greedy_budget``.
        seed: base RNG seed.
        n_rounds: total rounds in the experiment. Accepted for signature
            compatibility; it does not change the result of a single round.
        history: optional historical pass-rate table for ``by_pass_rate`` /
            ``greedy_budget``.
        agent_type: smolagents agent loop; the whole matrix uses one.
        out_dir: artifacts root (defaults to ``<project>/artifacts``).
        force: re-run even if this round's artifacts already exist.
        progress: print a progress line every ``PROGRESS_EVERY`` pairs.
        quiet: silence SMOLTRACE's own stdout chatter.

    Returns:
        An :class:`ArmResult` holding the records, the SMOLTRACE leaderboard rows
        and our ranking, and the paths written.
    """
    del n_rounds  # accepted for signature stability; a round is self-contained
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if not tasks:
        raise ValueError("run_arm needs at least one task")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("run_arm needs at least one candidate")

    tasks = list(tasks)
    paths = _round_paths(out_dir, arm, round_index)

    if not force and paths["run"].is_file() and paths["records"].is_file():
        return _load_round(arm, round_index, paths)

    started = time.perf_counter()
    session = _ensure_otel()
    session.begin_round()

    allocations = allocate_tiers(
        arm, tasks, policy=policy, token_budget=token_budget, history=history
    )
    tier_by_task = {a.task_id: a.tier for a in allocations}
    #: this arm-round's identity: passed to ``extract_traces`` so its trace
    #: documents carry the round, and stored on every record.
    run_id = _round_run_id(arm, round_index)

    pool = _AgentPool()
    records, accounting = _evaluate_round(
        arm=arm,
        round_index=round_index,
        tasks=tasks,
        candidates=candidates,
        tier_by_task=tier_by_task,
        agent_type=agent_type,
        run_id=run_id,
        seed=seed,
        pool=pool,
        session=session,
        progress=progress,
        quiet=quiet,
    )

    leaderboard_rows = _leaderboard_rows(
        arm=arm,
        round_index=round_index,
        candidates=candidates,
        run_id=run_id,
        accounting=accounting,
        quiet=quiet,
    )
    ranking = [
        _leaderboard_entry(arm, round_index, row).to_dict() for row in _rank_candidates(records)
    ]

    wall_ms = (time.perf_counter() - started) * 1000.0
    total_tokens = sum(int(r["total_tokens"]) for r in records)
    total_cost = sum(float(r["cost_usd"]) for r in records)

    result = ArmResult(
        arm=arm,
        round_index=round_index,
        run_id=run_id,
        # The OTEL session is process-wide, so this is not per-round; it is
        # recorded so a reader can tell which provider produced the traces.
        otel_run_id=session.otel_run_id,
        records=records,
        leaderboard_rows=leaderboard_rows,
        ranking=ranking,
        allocations=[a.to_dict() for a in allocations],
        paths={key: str(path) for key, path in paths.items()},
        wall_ms=round(wall_ms, 2),
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 8),
        n_tasks=len(tasks),
        n_candidates=len(candidates),
    )
    _persist_round(result, paths)
    return result


def _persist_round(result: ArmResult, paths: Dict[str, Path]) -> None:
    _write_json(paths["records"], result.records)
    _write_json(
        paths["leaderboard"],
        {
            "arm": result.arm,
            "round_index": result.round_index,
            "run_id": result.run_id,
            "n_tasks": result.n_tasks,
            "n_candidates": result.n_candidates,
            # SMOLTRACE's own leaderboard rows (one per candidate)
            "rows": result.leaderboard_rows,
            # our ranking over the same records, for cross-arm comparison
            "ranking": result.ranking,
        },
    )
    _write_json(paths["allocations"], result.allocations)
    _write_json(
        paths["run"],
        {
            "arm": result.arm,
            "round_index": result.round_index,
            "run_id": result.run_id,
            "otel_run_id": result.otel_run_id,
            "n_tasks": result.n_tasks,
            "n_candidates": result.n_candidates,
            "n_records": len(result.records),
            "total_tokens": result.total_tokens,
            "total_cost_usd": result.total_cost_usd,
            "wall_ms": result.wall_ms,
            "tiers": _tier_histogram(result.records),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


def _tier_histogram(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    histogram: Dict[str, int] = {tier: 0 for tier in TIER_ORDER}
    for record in records:
        histogram[record["allocated_tier"]] = histogram.get(record["allocated_tier"], 0) + 1
    return histogram


def _load_round(arm: str, round_index: int, paths: Dict[str, Path]) -> ArmResult:
    """Rebuild an :class:`ArmResult` from artifacts already on disk (resume)."""
    records = list(_read_json(paths["records"]))
    run_meta = _read_json(paths["run"]) if paths["run"].is_file() else {}
    leaderboard = _read_json(paths["leaderboard"]) if paths["leaderboard"].is_file() else {}
    allocations = _read_json(paths["allocations"]) if paths["allocations"].is_file() else []
    return ArmResult(
        arm=arm,
        round_index=round_index,
        run_id=str(run_meta.get("run_id") or _round_run_id(arm, round_index)),
        otel_run_id=str(run_meta.get("otel_run_id") or ""),
        records=records,
        leaderboard_rows=list(leaderboard.get("rows") or []),
        ranking=list(leaderboard.get("ranking") or []),
        allocations=list(allocations),
        paths={key: str(path) for key, path in paths.items()},
        wall_ms=float(run_meta.get("wall_ms") or 0.0),
        total_tokens=int(run_meta.get("total_tokens") or sum(int(r["total_tokens"]) for r in records)),
        total_cost_usd=float(
            run_meta.get("total_cost_usd") or sum(float(r["cost_usd"]) for r in records)
        ),
        n_tasks=int(run_meta.get("n_tasks") or len({r["task_id"] for r in records})),
        n_candidates=int(
            run_meta.get("n_candidates") or len({r["candidate_model"] for r in records})
        ),
        skipped=True,
    )


def load_round_artifacts(
    out_dir: Optional[Path | str] = None,
    *,
    arm: Optional[str] = None,
    round_index: Optional[int] = None,
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Read ``artifacts/records/*.json`` back into ``{arm: {round: records}}``."""
    root = _artifacts_root(out_dir) / RECORDS_DIR
    collected: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    if not root.is_dir():
        return collected
    for path in sorted(root.glob("*_r*.json")):
        stem = path.stem
        try:
            arm_name, round_part = stem.rsplit("_r", 1)
            round_number = int(round_part)
        except ValueError:
            continue
        if arm is not None and arm_name != arm:
            continue
        if round_index is not None and round_number != round_index:
            continue
        collected.setdefault(arm_name, {})[round_number] = list(_read_json(path))
    return collected


def experiment_totals(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate token and cost totals out of a :func:`run_experiment` result."""
    per_arm: Dict[str, Dict[str, Any]] = {}
    grand_tokens = 0
    grand_cost = 0.0
    for arm_result in experiment.get("arms", []):
        arm = arm_result["arm"]
        bucket = per_arm.setdefault(arm, {"total_tokens": 0, "total_cost_usd": 0.0, "n_records": 0})
        bucket["total_tokens"] += int(arm_result["total_tokens"])
        bucket["total_cost_usd"] = round(
            bucket["total_cost_usd"] + float(arm_result["total_cost_usd"]), 8
        )
        bucket["n_records"] += len(arm_result["records"])
        grand_tokens += int(arm_result["total_tokens"])
        grand_cost += float(arm_result["total_cost_usd"])
    return {
        "total_tokens": grand_tokens,
        "total_cost_usd": round(grand_cost, 8),
        "n_records": sum(b["n_records"] for b in per_arm.values()),
        "per_arm": per_arm,
    }


def run_experiment(
    tasks: Sequence[TaskSpec],
    candidates: Optional[Sequence[ModelProfile]] = None,
    *,
    arms: Sequence[str] = ARMS,
    n_rounds: int = 3,
    policy: str = "by_category",
    token_budget: Optional[int] = None,
    seed: int = 0,
    history: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path | str] = None,
    force: bool = False,
    progress: bool = True,
    quiet: bool = True,
    agent_type: str = DEFAULT_AGENT_TYPE,
) -> Dict[str, Any]:
    """Run the full three-arm x N-round matrix and persist every artifact.

    Artifact layout under ``out_dir`` (default ``<project>/artifacts``)::

        records/<arm>_r<round>.json      list[TaskRecord]
        leaderboard/<arm>_r<round>.json  {rows: SMOLTRACE rows, ranking: ours}
        allocations/<arm>_r<round>.json  list[Allocation]
        runs/<arm>_r<round>.json         run metadata
        runs/experiment.json             whole-matrix summary

    ``force=False`` makes the run resumable: a round whose artifacts already
    exist is loaded instead of re-evaluated.

    Returns:
        ``{"arms": [ArmResult dicts], "records": {arm: {round: records}},
        "totals": {...}, "config": {...}, "paths": {...}}``
    """
    tasks = list(tasks)
    candidates = list(candidates) if candidates is not None else list(CANDIDATES)
    arms = list(arms)
    if not tasks:
        raise ValueError("run_experiment needs at least one task")
    if not candidates:
        raise ValueError("run_experiment needs at least one candidate")

    out_root = _artifacts_root(out_dir)
    started = time.perf_counter()

    arm_results: List[ArmResult] = []
    for arm in arms:
        for round_index in range(n_rounds):
            print(f"[run_experiment] {arm} round {round_index}", flush=True)
            result = run_arm(
                arm,
                tasks,
                candidates,
                round_index=round_index,
                policy=policy,
                token_budget=token_budget,
                seed=seed,
                n_rounds=n_rounds,
                history=history,
                agent_type=agent_type,
                out_dir=out_root,
                force=force,
                progress=progress,
                quiet=quiet,
            )
            status = "skipped (cached)" if result.skipped else f"{result.wall_ms:.0f} ms"
            print(
                f"[run_experiment] {arm} r{round_index}: {status}, "
                f"{result.total_tokens} tokens, ${result.total_cost_usd:.6f}",
                flush=True,
            )
            arm_results.append(result)

    #: {arm: {round: records}} — in-memory mirror of artifacts/records/
    records_by_arm_round: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for result in arm_results:
        records_by_arm_round.setdefault(result.arm, {})[result.round_index] = result.records

    experiment = {
        "config": {
            "arms": arms,
            "n_rounds": n_rounds,
            "policy": policy,
            "token_budget": token_budget,
            "seed": seed,
            "n_tasks": len(tasks),
            "n_candidates": len(candidates),
            "candidates": [c.model_id for c in candidates],
            "agent_type": agent_type,
        },
        "arms": [result.to_dict() for result in arm_results],
        "records": records_by_arm_round,
        "totals": {},
        "paths": {"experiment": str(out_root / RUNS_DIR / "experiment.json")},
        "wall_ms": 0.0,
    }
    experiment["totals"] = experiment_totals(experiment)
    experiment["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 2)

    _write_json(out_root / RUNS_DIR / "experiment.json", _experiment_summary(experiment))
    print(
        f"[run_experiment] done in {experiment['wall_ms'] / 1000.0:.1f}s: "
        f"{experiment['totals']['total_tokens']} tokens, "
        f"${experiment['totals']['total_cost_usd']:.4f}, "
        f"{experiment['totals']['n_records']} records",
        flush=True,
    )
    return experiment


def _experiment_summary(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON-safe view of an experiment (drops the full record payloads)."""
    return {
        "config": experiment["config"],
        "totals": experiment["totals"],
        "wall_ms": experiment["wall_ms"],
        "rounds": [
            {
                "arm": arm_result["arm"],
                "round_index": arm_result["round_index"],
                "run_id": arm_result["run_id"],
                "otel_run_id": arm_result["otel_run_id"],
                "n_records": len(arm_result["records"]),
                "total_tokens": arm_result["total_tokens"],
                "total_cost_usd": arm_result["total_cost_usd"],
                "wall_ms": arm_result["wall_ms"],
                "skipped": arm_result["skipped"],
                "paths": arm_result["paths"],
            }
            for arm_result in experiment["arms"]
        ],
    }
