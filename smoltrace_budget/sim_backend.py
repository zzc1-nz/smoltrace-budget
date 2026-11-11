"""Calibrated simulated LM backend that emits *real* OpenTelemetry LLM spans.

This module is the scientific core of the experiment. It provides a genuine
``smolagents.Model`` subclass (:class:`SimulatedLiteLLMModel`) that plugs into
SMOLTRACE's own ``ToolCallingAgent`` loop exactly like ``LiteLLMModel`` would,
but answers from a calibrated stochastic model instead of a network call.
Nothing is faked downstream: the spans this model emits are ordinary OTel spans,
and SMOLTRACE's own ``extract_traces`` / ``extract_metrics`` /
``compute_leaderboard_row`` do all token, cost and pass-rate accounting.

Why a simulation at all
-----------------------
The experiment is a 132-task x 3-arm x 3-round x 5-model matrix. Running it on
live APIs would cost money, be non-reproducible and drift as vendors re-train.
The DESIGN (see ``DESIGN.md``, "Model backend") therefore specifies a calibrated
backend: determinism and a known ground truth are *features*, not compromises.
Every millisecond of span emission, token accounting, cost calculation and
leaderboard aggregation downstream is the real SMOLTRACE code path.

The phenomenon this backend must reproduce
------------------------------------------
The project studies a specific distortion: a **cheap measurement tier is
systematically unfair to models that need tools and multiple steps of
deliberation**. Concretely, ``ModelProfile`` carries two latent traits:

* ``tool_reliance`` -- how much the model needs tools to reach its true ability;
* ``deliberation``  -- how much reasoning it does before answering.

A tier supplies a limited *process budget* (``TIER_SPEC[tier]["max_steps"]``).
A model whose needs exceed that budget cannot express itself, so its **measured**
pass rate is pulled below its **true** pass rate. The pull is
*magnitude-proportional*: models with low ``tool_reliance``/``deliberation``
(QuickThink-7B) lose almost nothing from a cheap tier, while models that need
tools and depth (Thorough-32B) are under-measured badly. That asymmetry is what
flips the leaderboard, and it is stated explicitly here rather than hidden behind
a magic constant.

The bias formula
----------------
For a candidate model ``m`` on a task of category ``c``:

1. **Process need** -- how many tool calls the model wants before answering::

       impulse(m, c) = 3.0 * m.tool_reliance + 2.0 * m.deliberation        # in [0, 5]
       want(m, c)    = clamp(round(1 + impulse(m, c) * need(c)) - 1, 1, 5)

   ``need(c)`` (``CATEGORY_PROCESS_NEED``) is how much process a category
   rewards: ``gaia`` 1.00 (multi-hop, tool-heavy), ``math`` 0.62, ``simpleqa``
   0.15 (a lookup). ``3.0`` and ``2.0`` are the relative weights of tool use vs.
   deliberation. The ``+1`` is the one model call that even a no-tool answer
   needs; subtracting it converts "steps" into "tool calls", and the clamp keeps
   the value inside what the most generous tier can offer.

2. **Tier capacity** -- tool calls the tier's step budget still allows after
   reserving one step for ``final_answer``::

       capacity(tier) = TIER_SPEC[tier]["max_steps"] - 1      # low 1, mid 3, high 5

3. **Shortfall** -- the unmet fraction of the model's process need::

       actual(m, tier, c) = min(want(m, c), capacity(tier))
       shortfall(m, tier, c) = 1 - actual / want              # in [0, 1]

4. **Measured pass rate** -- a multiplicative, tier-relative penalty on the
   model's *true* ability::

       p_measured(m, tier, c) = m.true_pass_rate[c] * (1 - BIAS_STRENGTH * shortfall)

   with ``BIAS_STRENGTH = 0.75``: at maximum shortfall the model keeps only a
   quarter of its true ability.

Properties worth stating, because they are load-bearing:

* **The high tier is the gold standard by construction.** ``capacity("high") = 5``
  is the clamp ceiling of ``want``, so ``shortfall == 0`` at the high tier for
  every model and every category, hence ``p_measured == true_pass_rate``. That is
  deliberate: the "gold standard" arm must be an unbiased measurement, otherwise
  there is no truth to compare the cheap arms against.
* **The low tier is where the unfairness lives.** ``capacity("low") = 1``, so any
  model that wants more than one tool call is truncated. The size of the loss is
  exactly proportional to how far its need exceeds the budget -- i.e. to
  ``tool_reliance`` and ``deliberation``.
* The expectation is monotone in tier: ``low <= mid <= high`` per (model, task).

A note on how truncation is modelled. ``smolagents`` reacts to a step-budget
overrun by hard-failing the run (``AgentMaxStepsError``: no answer at all, so
``response_correct=False``). Emitting that literally would make *every* model
that wants more than one tool call score exactly 0% at the low tier, collapsing
the arm into a five-way tie at zero -- no ranking, no distortion, nothing to
measure. Real agents instead *notice* the budget and answer early, which is
strictly better and is what ``shortfall`` prices: the model answers within
budget, having done only ``actual`` of the ``want`` tool calls it wanted, and is
correspondingly less likely to be right. The model therefore never overruns
``max_steps``; every ``(tier, candidate, task)`` triple terminates on its own
``final_answer``.

Grading stays in SMOLTRACE's hands
----------------------------------
This backend does **not** decide success. It only guarantees that the text it
emits is *grader-consistent*: when its internal draw says "correct", the
``final_answer`` argument contains the task's expected keyword; when it says
"incorrect", the argument provably contains **none** of them. SMOLTRACE's own
keyword grader (``core.evaluate_single_test``) then produces ``success``. The
backend never imports or reimplements the grader.

Two facts about SMOLTRACE 0.2.12 that constrain the emitted tool calls
---------------------------------------------------------------------
1. ``evaluate_single_test`` computes::

       success = tool_called and correct_tool and final_answer_called and response_correct

   where ``tool_called`` counts only **non-``final_answer``** tool calls
   (``core.extract_tools_from_action_step`` skips it), and all 132 benchmark
   tasks declare ``expected_tool="multiple"`` / ``expected_tool_calls=1``. So a
   correct answer with *zero* tool calls scores ``success=False``. This is why
   even the low tier makes exactly ``LOW_TIER_TOOL_CALLS = 1`` tool call: with
   zero, the entire ``all_low`` arm would be a structural 0% for every model and
   the experiment would measure nothing. The low tier still does no tool
   *exploration* (1 call vs. up to 3 at mid and 5 at high), which is where the
   asymmetry comes from.
2. A tool call is only counted if smolagents can validate its arguments against
   the tool's input schema; a rejected call raises ``AgentToolCallError`` and the
   step records no tool calls at all. Every emitted call therefore uses arguments
   taken from :data:`TOOL_ARGUMENTS`, keyed by tool name, and falls back to
   ``calculator`` for unknown tools rather than guessing a schema.

Cost follows the tier, not the candidate
----------------------------------------
The span's ``llm.model_name`` / ``gen_ai.request.model`` is the **tier's** real
API model id (``TIER_API_MODEL``), never the candidate's. This is deliberate and
is the point of the design: a tier is named after the model class you would have
to pay for to obtain that measurement fidelity, so evaluation cost must follow
the tier. SMOLTRACE prices each span by model id via genai_otel's
``CostCalculator``, which means the cost that lands in the trace, the
leaderboard row and the report is exactly the cost of the tier that was bought.

``TIER_API_MODEL["high"]`` is ``anthropic/claude-sonnet-4-5`` rather than
``anthropic/claude-sonnet-4`` on purpose: genai_otel has no pricing key for
``claude-sonnet-4`` and silently prices it at **$0.000000**, which would make the
gold-standard arm free and every "cost saved" figure meaningless.

Determinism
-----------
All randomness comes from :func:`_unit_hash`, a SHA-256 based uniform in [0, 1)
over an explicit tuple of inputs. No global random state, no ``random`` module,
no wall-clock dependence. The same ``(candidate, tier, stage, seed, task,
round_index)`` reproduces the same decision, the same tool-call count, the same
text and the same token counts on a fresh interpreter.

The tier, the stage and the round are part of the hash on purpose: each arm and
each round is a *separate measurement run*, so its sampling noise must be
independent of the other arms'. Re-using one draw across arms would give the
allocated arm an artificially perfect rank correlation with ``all_high`` and
would hide the very stability loss the project reports.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from smolagents import Model
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    TokenUsage,
)

from .contracts import (
    TIER_SPEC,
    ModelProfile,
    TaskSpec,
    validate_tier,
)

__all__ = [
    "SimulatedLiteLLMModel",
    "TierModelFactory",
    "build_model",
    "measured_pass_rate",
    "shortfall",
    "needed_tool_calls",
    "tier_tool_capacity",
    "tier_api_model",
    "TIER_API_MODEL",
    "CATEGORY_PROCESS_NEED",
    "BIAS_STRENGTH",
    "LOW_TIER_TOOL_CALLS",
    "SPAN_NAME",
]

# ---------------------------------------------------------------------------
# Constants of the simulation
# ---------------------------------------------------------------------------

#: Span name for one model call. Required by SMOLTRACE's trace extractor, which
#: groups spans per trace and aggregates ``llm.token_count.*`` off them.
SPAN_NAME = "llm.chat"

#: The real API model id behind each measurement tier. Cost follows the tier,
#: because a tier *is* "the model class you'd have to pay for to get this
#: fidelity". ``low``/``mid`` match ``contracts.TIER_LABELS`` exactly; ``high``
#: uses ``claude-sonnet-4-5`` because genai_otel prices ``claude-sonnet-4`` at
#: zero (see the module docstring).
TIER_API_MODEL: Dict[str, str] = {
    "low": "openai/gpt-4o-mini",
    "mid": "openai/gpt-4o",
    "high": "anthropic/claude-sonnet-4-5",
}

#: How strongly an unmet process need depresses measured ability. At
#: ``shortfall == 1`` a model keeps ``1 - BIAS_STRENGTH`` of its true pass rate.
BIAS_STRENGTH: float = 0.75

#: Relative weight of tool use vs. deliberation in the process-need impulse.
TOOL_RELIANCE_WEIGHT: float = 3.0
DELIBERATION_WEIGHT: float = 2.0

#: How much of the model's [0, 1] need a category actually rewards. GAIA is
#: multi-hop and tool-heavy; SimpleQA is a lookup; MATH sits in between.
CATEGORY_PROCESS_NEED: Dict[str, float] = {
    "gaia": 1.00,
    "math": 0.62,
    "simpleqa": 0.15,
    "default": 0.50,
}

#: Tool calls the low tier makes. One, not zero: SMOLTRACE's grader requires a
#: non-final tool call for ``success`` (see the module docstring). Raising this
#: would erode the tier's unfairness; lowering it to 0 turns ``all_low`` into a
#: structural 0% for every model.
LOW_TIER_TOOL_CALLS: int = 1

#: Schema-valid arguments per tool name. A call whose arguments fail smolagents'
#: validation is dropped, so unknown tools fall back to ``calculator`` instead of
#: being called with a guessed schema. These are the default SMOLTRACE sandbox
#: tools (``smoltrace.core.get_all_tools``).
TOOL_ARGUMENTS: Dict[str, Dict[str, Any]] = {
    "calculator": {"expression": "1 + 1"},
    "get_weather": {"location": "Paris, France"},
    "get_current_time": {"timezone": "UTC"},
}

#: Preference order when several schema-safe tools are available. ``calculator``
#: is deterministic and cheap (its observation is short), which keeps token
#: accounting stable; ``get_current_time`` leaks a wall-clock string into the
#: prompt and is only used if nothing better exists.
TOOL_PREFERENCE: Sequence[str] = ("calculator", "get_weather", "get_current_time")

#: Tokens a model spends per call on hidden reasoning, scaled by deliberation.
DELIBERATION_COMPLETION_TOKENS: int = 80

#: A model always emits at least this many completion tokens per call, so a very
#: terse model still costs something realistic.
MIN_COMPLETION_TOKENS: int = 8

#: ``len(text) // CHARS_PER_TOKEN`` is the prompt/completion token estimator used
#: throughout. Four characters per token is the usual English rule of thumb and is
#: what the working prototype uses, which keeps this backend's accounting
#: consistent with SMOLTRACE's own numbers.
CHARS_PER_TOKEN: int = 4

#: Plausible per-call latency envelope (milliseconds), scaled by tier speed.
#: Recorded as a span attribute for the report; it is *modelled* latency, not
#: wall-clock, because the whole point of the backend is to cost nothing to run.
LATENCY_MS_BY_TIER: Dict[str, float] = {"low": 320.0, "mid": 900.0, "high": 2100.0}


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------


def _unit_hash(*parts: Any) -> float:
    """Stable uniform draw in ``[0, 1)`` from an explicit tuple of inputs.

    Uses SHA-256 over a JSON-ish rendering of ``parts`` so the value is identical
    across processes, platforms and interpreter versions -- unlike ``hash()``,
    which is salted per process. Never touches global random state.
    """
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


# ---------------------------------------------------------------------------
# The bias model (pure functions, reused by tests and analysis)
# ---------------------------------------------------------------------------


def tier_tool_capacity(tier: str) -> int:
    """Tool calls a tier's step budget still allows after reserving one step.

    ``TIER_SPEC`` budgets ``max_steps`` model calls per task; the last call must
    be ``final_answer``, so at most ``max_steps - 1`` calls can be tool use.
    """
    validate_tier(tier)
    return max(0, int(TIER_SPEC[tier]["max_steps"]) - 1)


def category_process_need(category: str) -> float:
    """How much tool/multi-step process a category rewards, in ``[0, 1]``."""
    return float(CATEGORY_PROCESS_NEED.get(category, CATEGORY_PROCESS_NEED["default"]))


def process_impulse(candidate: ModelProfile) -> float:
    """How much process the model *wants*, before the category scales it."""
    return (
        TOOL_RELIANCE_WEIGHT * float(candidate.tool_reliance)
        + DELIBERATION_WEIGHT * float(candidate.deliberation)
    )


def needed_tool_calls(candidate: ModelProfile, category: str) -> int:
    """Tool calls the model would like before answering, clamped to ``[1, 5]``.

    The upper clamp is ``tier_tool_capacity("high")`` so the most generous tier is
    *always* sufficient; that is what makes the high tier an unbiased reference.
    """
    raw = 1.0 + process_impulse(candidate) * category_process_need(category)
    # Round half up, not Python's banker's rounding: the threshold must not depend
    # on whether the fractional part lands on an even integer.
    steps = int(math.floor(raw + 0.5))
    return max(1, min(steps - 1, tier_tool_capacity("high")))


def shortfall(candidate: ModelProfile, tier: str, category: str) -> float:
    """Unmet fraction of the model's process need on this tier, in ``[0, 1]``.

    ``0.0`` means the tier gave the model everything it wanted (so it is measured
    at its true ability); ``1.0`` means it got nothing it asked for.
    """
    want = needed_tool_calls(candidate, category)
    actual = min(want, tier_tool_capacity(tier))
    return max(0.0, min(1.0, 1.0 - (actual / want)))


def measured_pass_rate(candidate: ModelProfile, tier: str, category: str) -> float:
    """The model's pass rate **as this tier measures it**.

    This is the analytic expectation the simulated draws are built on::

        p = true_pass_rate[category] * (1 - BIAS_STRENGTH * shortfall(candidate, tier, category))

    At the high tier ``shortfall == 0``, so ``p == true_pass_rate`` and the tier is
    an unbiased measurement. Lower tiers are biased downward, and the size of the
    bias grows with ``tool_reliance`` and ``deliberation``.
    """
    validate_tier(tier)
    base = candidate.pass_rate_for(category)
    loss = BIAS_STRENGTH * shortfall(candidate, tier, category)
    return max(0.0, min(1.0, base * (1.0 - loss)))


def tier_api_model(tier: str) -> str:
    """The real API model id a tier is priced as."""
    validate_tier(tier)
    return TIER_API_MODEL[tier]


# ---------------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------------


def _serialize_messages(messages: Iterable[Any]) -> str:
    """Render the prompt the way a chat API would serialize it.

    Token counts are derived from this string, so it has to reflect the *whole*
    growing conversation: that is why a tier that takes more steps costs more.
    """
    rendered: List[Dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role is not None and hasattr(role, "value"):
            role = role.value
        entry: Dict[str, Any] = {
            "role": str(role),
            "content": str(getattr(message, "content", "") or ""),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "function": {
                        "name": getattr(getattr(call, "function", None), "name", None),
                        "arguments": getattr(getattr(call, "function", None), "arguments", None),
                    }
                }
                for call in tool_calls
            ]
        rendered.append(entry)
    return json.dumps(rendered, default=str, ensure_ascii=False)


def _positive_answer(task: Optional[TaskSpec]) -> str:
    """An answer that is guaranteed to contain the task's first expected keyword."""
    if task is None or not task.expected_keywords:
        return "The answer is 42."
    keyword = task.expected_keywords[0]
    return f"The answer is {keyword}."


#: Candidate phrasings for an *incorrect* answer. The first one that contains
#: none of the task's expected keywords is used; the empty string is the
#: guaranteed-safe terminal fallback (no keyword can be a substring of "").
_NEGATIVE_ANSWER_CANDIDATES: Sequence[str] = (
    "I do not know.",
    "unable to determine",
    "???",
    "",
)


def _negative_answer(task: Optional[TaskSpec]) -> str:
    """An answer that provably contains **none** of the task's expected keywords.

    A wrong answer must not accidentally satisfy SMOLTRACE's keyword grader, so
    the choice is verified rather than assumed -- keywords can be as short as a
    single digit and must not appear anywhere in the text.
    """
    keywords = [k.lower() for k in (task.expected_keywords if task else []) if k]
    for candidate in _NEGATIVE_ANSWER_CANDIDATES:
        if not any(keyword in candidate.lower() for keyword in keywords):
            return candidate
    return ""


def _tool_call_payload(step_index: int, available: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Pick a schema-valid tool call for this step, or ``None`` if impossible.

    Only tools whose argument schema is known (:data:`TOOL_ARGUMENTS`) are called;
    an unrecognised tool name would produce an invalid call that smolagents drops,
    costing the whole task its ``tool_called`` flag, so ``calculator`` (always
    present in SMOLTRACE's default sandbox) is the fallback.
    """
    known = [name for name in TOOL_PREFERENCE if name in available]
    if known:
        name = known[step_index % len(known)]
    elif "calculator" in available:
        name = "calculator"
    else:
        return None
    arguments = dict(TOOL_ARGUMENTS[name])
    # Vary the arithmetic so successive steps do not send byte-identical turns;
    # that keeps the growing-prompt token growth realistic across steps.
    if name == "calculator":
        left = step_index * 7 + 3
        right = step_index * 5 + 2
        arguments = {"expression": f"{left} + {right} * 2"}
    return {"name": name, "arguments": arguments}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class SimulatedLiteLLMModel(Model):
    """A calibrated, deterministic, span-emitting stand-in for ``LiteLLMModel``.

    One instance measures **one** (candidate model, tier, task) triple, which is
    exactly the granularity SMOLTRACE cannot express on its own: its pipeline
    holds one model for a whole run, while this project needs a different
    fidelity tier (and therefore a different priced model id) *per task*.

    Args:
        candidate: the model under test, carrying its latent ``true_pass_rate``,
            ``tool_reliance``, ``deliberation`` and ``verbosity``.
        tier: measurement fidelity, one of ``contracts.TIER_ORDER``.
        stage: opaque label for the pipeline stage or experimental arm
            (e.g. ``"arm:all_high"``). It salts the RNG, so different stages draw
            independent measurement noise, and is recorded on every span.
        seed: experiment seed.
        task: the task being measured; supplies ``category`` (which drives the
            bias) and ``expected_keywords`` (which drives grader-consistency).
        round_index: experimental round; part of the RNG salt, because each round
            must be an independent measurement.

    Attributes:
        record_calls: number of ``generate()`` calls made so far. The runner uses
            this to sanity-check that a high-tier task really did cost more model
            calls than a low-tier one.
        is_correct: the up-front correctness draw for this (task, candidate, tier,
            stage, seed, round). Fixed before the first call so every step of one
            run is consistent with a single outcome.

    The class emits one ``"llm.chat"`` span per ``generate()`` call, carrying the
    GenAI semantic-convention attributes SMOLTRACE reads, plus this project's
    ``allocation_tier``. Token counts are derived from the actual prompt and the
    actual generated text -- never from a constant -- so a tier that takes more
    steps and re-sends a longer history genuinely costs more.
    """

    def __init__(
        self,
        candidate: ModelProfile,
        tier: str,
        stage: str = "eval",
        *,
        seed: int = 0,
        task: Optional[TaskSpec] = None,
        round_index: int = 0,
        **kwargs: Any,
    ) -> None:
        self.candidate = candidate
        self.tier = validate_tier(tier)
        self.stage = str(stage)
        self.seed = int(seed)
        self.task = task
        self.round_index = int(round_index)
        self.category = (task.category if task is not None else "default") or "default"
        self.tier_model_id = TIER_API_MODEL[self.tier]

        # A custom model instance is not an API client: never let smolagents
        # forward provider kwargs to it.
        kwargs.pop("api_key", None)
        kwargs.pop("api_base", None)
        super().__init__(model_id=self.tier_model_id, **kwargs)

        # --- the single up-front outcome draw -----------------------------
        # Drawn once per (task, candidate, tier, stage, seed, round) so that every
        # step of the run agrees on the outcome. The tier/stage/round are in the
        # salt so the three arms of one round are independent measurements.
        draw = _unit_hash(
            "outcome",
            self.seed,
            self.round_index,
            self.stage,
            self.tier,
            candidate.api_model_id,
            self.category,
            task.id if task is not None else "",
        )
        self.is_correct: bool = draw < measured_pass_rate(candidate, self.tier, self.category)

        # --- process budget ------------------------------------------------
        self.wanted_tool_calls = needed_tool_calls(candidate, self.category)
        self.capacity = tier_tool_capacity(self.tier)
        self.tool_calls_to_make = min(self.wanted_tool_calls, self.capacity)
        if self.tier == "low":
            # Structural floor: SMOLTRACE cannot grade a task without one
            # non-final tool call (see module docstring).
            self.tool_calls_to_make = max(self.tool_calls_to_make, LOW_TIER_TOOL_CALLS)
        self.shortfall = shortfall(candidate, self.tier, self.category)
        self.expected_pass_rate = measured_pass_rate(candidate, self.tier, self.category)

        # --- bookkeeping ---------------------------------------------------
        self.record_calls: int = 0
        self.call_records: List[Dict[str, Any]] = []
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self._tracer = None

    # -- description ------------------------------------------------------

    @property
    def calls(self) -> int:
        """Alias for :attr:`record_calls`."""
        return self.record_calls

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def describe(self) -> Dict[str, Any]:
        """Everything the runner/analysis needs to explain this measurement."""
        return {
            "candidate": self.candidate.model_id,
            "tier": self.tier,
            "tier_model_id": self.tier_model_id,
            "stage": self.stage,
            "category": self.category,
            "task_id": self.task.id if self.task is not None else None,
            "wanted_tool_calls": self.wanted_tool_calls,
            "tier_capacity": self.capacity,
            "tool_calls_to_make": self.tool_calls_to_make,
            "shortfall": round(self.shortfall, 6),
            "expected_pass_rate": round(self.expected_pass_rate, 6),
            "is_correct": self.is_correct,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SimulatedLiteLLMModel(candidate={self.candidate.model_id!r}, tier={self.tier!r}, "
            f"category={self.category!r}, shortfall={self.shortfall:.3f}, "
            f"p={self.expected_pass_rate:.3f}, correct={self.is_correct})"
        )

    # -- token accounting -------------------------------------------------

    def _prompt_tokens(self, messages: Sequence[Any]) -> int:
        """Prompt tokens from the *actual* serialized conversation, x verbosity.

        Scaling by ``ModelProfile.verbosity`` is causally right, not cosmetic: a
        verbose model's own previous assistant turns are part of the next prompt,
        so the history it re-sends really is longer.
        """
        chars = len(_serialize_messages(messages))
        return max(1, int((chars / CHARS_PER_TOKEN) * float(self.candidate.verbosity)))

    def _completion_tokens(self, text: str) -> int:
        """Completion tokens from the generated text plus hidden reasoning.

        ``deliberation`` buys extra internal reasoning tokens per call, which is
        why a high-deliberation model is measurably more expensive to evaluate
        than a terse one even at equal step counts.
        """
        visible = len(text) / CHARS_PER_TOKEN
        reasoning = DELIBERATION_COMPLETION_TOKENS * float(self.candidate.deliberation)
        return max(MIN_COMPLETION_TOKENS, int((visible + reasoning) * float(self.candidate.verbosity)))

    # -- span emission ----------------------------------------------------

    def _emit_span(self, prompt_tokens: int, completion_tokens: int, step_index: int, tool_name: Optional[str]) -> None:
        """Emit the one ``llm.chat`` span SMOLTRACE prices this call from.

        The attribute names are SMOLTRACE's contract, not a style choice:
        ``core.extract_traces`` sums ``llm.token_count.total`` per trace and prices
        each span from ``llm.model_name`` via genai_otel's ``CostCalculator``.
        ``allocation_tier`` is this project's custom attribute -- the allocation
        decision travels *inside* the trace, so allocation-aware analysis needs no
        side-file join.
        """
        if self._tracer is None:
            from opentelemetry import trace as otel_trace

            self._tracer = otel_trace.get_tracer("smoltrace_budget.sim")

        with self._tracer.start_as_current_span(SPAN_NAME) as span:
            span.set_attribute("openinference.span.kind", "LLM")
            # Cost follows the TIER, never the candidate: the tier is what was
            # bought, and the price list is keyed by model id.
            span.set_attribute("llm.model_name", self.tier_model_id)
            span.set_attribute("gen_ai.request.model", self.tier_model_id)
            span.set_attribute("llm.token_count.prompt", prompt_tokens)
            span.set_attribute("llm.token_count.completion", completion_tokens)
            span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)
            span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
            span.set_attribute("allocation_tier", self.tier)
            # Extra, non-contract attributes: purely additive, SMOLTRACE ignores
            # them, but they let the report explain why a measurement looked the
            # way it did without re-deriving the simulation.
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("sim.candidate_model", self.candidate.api_model_id)
            span.set_attribute("sim.stage", self.stage)
            span.set_attribute("sim.step_index", int(step_index))
            span.set_attribute("sim.shortfall", float(round(self.shortfall, 6)))
            span.set_attribute("sim.expected_pass_rate", float(round(self.expected_pass_rate, 6)))
            span.set_attribute("sim.is_correct", bool(self.is_correct))
            span.set_attribute("sim.tool_name", tool_name or "")
            span.set_attribute("sim.latency_ms", float(LATENCY_MS_BY_TIER[self.tier]))

    def _modeled_latency_ms(self, step_index: int) -> float:
        """Plausible per-call latency, jittered deterministically around the tier mean."""
        base = LATENCY_MS_BY_TIER[self.tier]
        jitter = 0.75 + 0.5 * _unit_hash(
            "latency",
            self.seed,
            self.round_index,
            self.stage,
            self.tier,
            self.candidate.api_model_id,
            self.task.id if self.task is not None else "",
            step_index,
        )
        return round(base * jitter, 3)

    # -- generation -------------------------------------------------------

    def _message(self, text: str, tool_call: Optional[Dict[str, Any]], prompt_tokens: int, completion_tokens: int) -> ChatMessage:
        """Assemble the ``ChatMessage`` smolagents' agent loop expects."""
        calls = None
        if tool_call is not None:
            calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name=tool_call["name"],
                        arguments=dict(tool_call["arguments"]),
                    ),
                    id=f"call_{self.record_calls}_{tool_call['name']}",
                    type="function",
                )
            ]
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            tool_calls=calls,
            token_usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

    def generate(  # type: ignore[override]
        self,
        messages: List[Any],
        stop_sequences: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Produce one agent step and emit one priced ``llm.chat`` span.

        The call sequence per task is deliberately plain: emit
        ``tool_calls_to_make`` schema-valid tool calls, then finish with
        ``final_answer``. Because ``tool_calls_to_make <= max_steps - 1``, the run
        always terminates on its own final answer and never trips
        ``AgentMaxStepsError`` -- the tier's unfairness is priced by
        :func:`shortfall` instead (see the module docstring for why a hard
        step-limit failure would destroy the measurement).

        ``tools_to_call_from`` is ``None`` under ``CodeAgent``; in that mode no
        tool call is emitted and the step content is a Python code block, which is
        what ``CodeAgent`` parses. (Note that SMOLTRACE's ``extract_tools_from_code``
        cannot attribute code-mode tool use, so ``agent_type="code"`` tasks cannot
        be graded as successes in this SMOLTRACE build; tool mode is the
        measurement path.)
        """
        step_index = self.record_calls
        self.record_calls += 1

        available: List[str] = []
        for tool in tools_to_call_from or []:
            name = getattr(tool, "name", None)
            if isinstance(name, str):
                available.append(name)

        code_mode = tools_to_call_from is None
        # Reserve every remaining step for the answer once the tool budget is spent.
        emit_tool_call = (not code_mode) and step_index < self.tool_calls_to_make
        tool_call = _tool_call_payload(step_index, available) if emit_tool_call else None
        if emit_tool_call and tool_call is None:
            # No schema-safe tool available; answer rather than emit an invalid call
            # (an invalid call would also strip the task of its `tool_called` flag).
            emit_tool_call = False

        answer_text = _positive_answer(self.task) if self.is_correct else _negative_answer(self.task)

        if tool_call is not None:
            content = ""
            rendered_text = json.dumps({"tool": tool_call["name"], "arguments": tool_call["arguments"]})
        elif code_mode:
            # CodeAgent requires a parsable code blob; `final_answer(...)` is
            # rewritten by smolagents into the agent's final answer.
            content = f"```python\nfinal_answer({json.dumps(answer_text)})\n```"
            rendered_text = content
        else:
            content = ""
            tool_call = {
                "name": "final_answer",
                "arguments": {"answer": answer_text},
            }
            rendered_text = json.dumps({"tool": "final_answer", "arguments": tool_call["arguments"]})

        prompt_tokens = self._prompt_tokens(messages)
        completion_tokens = self._completion_tokens(rendered_text)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        self._emit_span(
            prompt_tokens,
            completion_tokens,
            step_index,
            tool_call["name"] if tool_call else None,
        )

        self.call_records.append(
            {
                "step_index": step_index,
                "tool": tool_call["name"] if tool_call else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_ms": self._modeled_latency_ms(step_index),
            }
        )

        return self._message(content, tool_call, prompt_tokens, completion_tokens)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TierModelFactory:
    """Convenience factory so callers need not remember the constructor details.

    The runner needs a fresh model per ``(candidate, tier, stage, task)``; keeping
    that in one place means the tier-validation and salt-ordering rules live in a
    single spot::

        factory = TierModelFactory(seed=7, stage="arm:allocated", round_index=1)
        model = factory.build(candidate, "high", task)
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        stage: str = "eval",
        round_index: int = 0,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.seed = int(seed)
        self.stage = str(stage)
        self.round_index = int(round_index)
        self.model_kwargs: Dict[str, Any] = dict(model_kwargs or {})

    def build(
        self,
        candidate: ModelProfile,
        tier: str,
        task: Optional[TaskSpec] = None,
        *,
        stage: Optional[str] = None,
        round_index: Optional[int] = None,
    ) -> SimulatedLiteLLMModel:
        """Return a ready-to-inject model for one measurement."""
        return SimulatedLiteLLMModel(
            candidate,
            tier,
            stage if stage is not None else self.stage,
            seed=self.seed,
            task=task,
            round_index=self.round_index if round_index is None else round_index,
            **self.model_kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TierModelFactory(seed={self.seed}, stage={self.stage!r}, "
            f"round_index={self.round_index})"
        )


def build_model(
    candidate: ModelProfile,
    tier: str,
    stage: str = "eval",
    seed: int = 0,
    task: Optional[TaskSpec] = None,
    *,
    round_index: int = 0,
    **kwargs: Any,
) -> SimulatedLiteLLMModel:
    """Build one :class:`SimulatedLiteLLMModel` -- the entry point the runner uses.

    Equivalent to ``TierModelFactory(...).build(...)`` and kept as a plain
    function because that is the signature the evaluation runner codes against.
    """
    return SimulatedLiteLLMModel(
        candidate,
        tier,
        stage,
        seed=seed,
        task=task,
        round_index=round_index,
        **kwargs,
    )
