"""Tests for the calibrated simulated LM backend.

The backend is the measurement instrument of this experiment: it produces both
the pass/fail outcome SMOLTRACE grades and the token counts SMOLTRACE accounts
for. These tests pin the properties the reported numbers depend on:

* determinism (a rerun must reproduce a published figure exactly),
* tier fidelity ordering (cheaper tiers must under-measure tool-dependent
  models — that bias is the phenomenon being studied),
* statutory validity of tool calls (an invalid call makes a task
  structurally unpassable, silently zeroing an entire arm),
* the ``allocation_tier`` span attribute and tier-based pricing.
"""

from __future__ import annotations

import pytest

from smoltrace_budget.contracts import ModelProfile, TaskSpec
from smoltrace_budget.models import CANDIDATES, get_candidate
from smoltrace_budget import sim_backend


def _task(category: str = "gaia", task_id: str | None = None) -> TaskSpec:
    return TaskSpec(
        id=task_id or f"{category}_0",
        prompt=f"A {category} question that needs research.",
        difficulty="hard",
        agent_type="both",
        category=category,
        source=category.upper(),
        expected_tool="multiple",
        expected_tool_calls=1,
        expected_keywords=["17"],
    )


def _messages():
    from smolagents.models import ChatMessage, MessageRole

    return [ChatMessage(role=MessageRole.USER, content="solve this")]


def _tools():
    """Stand-ins with the only attribute the backend reads: ``.name``."""

    class _Tool:
        def __init__(self, name: str):
            self.name = name

    return [_Tool("calculator"), _Tool("get_weather"), _Tool("final_answer")]


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------


def test_every_tier_maps_to_a_priceable_api_model():
    """An unpriced tier id would silently cost $0 and void the cost comparison."""
    for tier in ("low", "mid", "high"):
        model_id = sim_backend.tier_api_model(tier)
        assert model_id
        assert model_id != "anthropic/claude-sonnet-4", (
            "claude-sonnet-4 has no genai_otel pricing key; use claude-sonnet-4-5"
        )


def test_tier_tool_capacity_increases_with_tier():
    capacities = [sim_backend.tier_tool_capacity(t) for t in ("low", "mid", "high")]
    assert capacities == sorted(capacities)
    assert capacities[0] < capacities[-1], "a cheaper tier must offer fewer process steps"


def test_low_tier_still_makes_one_tool_call():
    """Regression guard: zero tool calls makes every task in the arm unpassable."""
    assert sim_backend.LOW_TIER_TOOL_CALLS >= 1


# ---------------------------------------------------------------------------
# The bias model (the thing the experiment measures)
# ---------------------------------------------------------------------------


def test_high_tier_measures_a_model_at_its_true_ability():
    for candidate in CANDIDATES:
        assert sim_backend.shortfall(candidate, "high", "gaia") == pytest.approx(0.0)
        assert sim_backend.measured_pass_rate(candidate, "high", "gaia") == pytest.approx(
            candidate.pass_rate_for("gaia")
        )


def test_cheaper_tiers_under_measure_a_tool_dependent_model():
    """Thorough-32B needs tools and deliberation; the cheap tier hides that."""
    thorough = get_candidate("thorough-32b")
    high = sim_backend.measured_pass_rate(thorough, "high", "gaia")
    mid = sim_backend.measured_pass_rate(thorough, "mid", "gaia")
    low = sim_backend.measured_pass_rate(thorough, "low", "gaia")

    assert high > mid >= low, f"expected a downward bias, got high={high} mid={mid} low={low}"


def test_bias_is_smaller_for_a_model_that_needs_no_tools():
    """The cheap tier's advantage to a self-sufficient model is the whole story."""
    quick = get_candidate("quickthink-7b")
    thorough = get_candidate("thorough-32b")

    quick_loss = quick.pass_rate_for("gaia") - sim_backend.measured_pass_rate(quick, "low", "gaia")
    thorough_loss = (
        thorough.pass_rate_for("gaia") - sim_backend.measured_pass_rate(thorough, "low", "gaia")
    )

    assert quick_loss < thorough_loss, (
        "the low tier must penalise the tool-dependent model more than the "
        f"self-sufficient one (quick lost {quick_loss:.3f}, thorough lost {thorough_loss:.3f})"
    )


def test_measured_pass_rate_stays_a_probability():
    for candidate in CANDIDATES:
        for tier in ("low", "mid", "high"):
            for category in ("gaia", "math", "simpleqa"):
                rate = sim_backend.measured_pass_rate(candidate, tier, category)
                assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_inputs_produce_the_same_answer_and_tokens():
    task = _task("math")
    candidate = get_candidate("balanced-13b")

    def run_once():
        model = sim_backend.build_model(
            candidate, "high", "arm:all_high", seed=7, task=task
        )
        results = []
        for _ in range(3):
            message = model.generate(_messages(), tools_to_call_from=_tools())
            calls = message.tool_calls or []
            results.append(
                (
                    str(message.content),
                    tuple(
                        (getattr(c.tool.function_name, "__str__", lambda: c)() if False else c.function.name)
                        for c in calls
                    ),
                    message.token_usage.input_tokens if message.token_usage else None,
                    message.token_usage.output_tokens if message.token_usage else None,
                )
            )
        return results

    assert run_once() == run_once(), "the backend must be deterministic for a fixed seed"


def test_different_seeds_can_give_different_outcomes():
    """Sanity check that the seed actually drives the correctness draw.

    The draw decides whether the emitted final answer contains the task's
    expected keyword, so the observable is the *answer text* carried in the
    ``final_answer`` tool call (``content`` stays a short narration).
    """
    task = _task("gaia")
    candidate = get_candidate("balanced-13b")

    def final_answer_text(message) -> str:
        for call in message.tool_calls or []:
            if call.function.name == "final_answer":
                return str(call.function.arguments.get("answer", ""))
        return ""

    outcomes = set()
    for seed in range(32):
        model = sim_backend.build_model(candidate, "low", "arm:all_low", seed=seed, task=task)
        for _ in range(6):
            message = model.generate(_messages(), tools_to_call_from=_tools())
            text = final_answer_text(message)
            if text:
                outcomes.add(text)
                break

    assert len(outcomes) > 1, (
        f"a fixed outcome across seeds suggests the seed is ignored: {outcomes}"
    )


# ---------------------------------------------------------------------------
# Tool-call validity: an invalid call silently makes a task unpassable
# ---------------------------------------------------------------------------


def test_tool_calls_use_schema_valid_arguments():
    """smolagents rejects a call whose arguments violate the tool's schema.

    A rejected call records ``tool_calls=None`` on the ActionStep, so SMOLTRACE
    never sees the tool as used and the task can never pass.
    """
    task = _task("gaia")
    candidate = get_candidate("thorough-32b")
    model = sim_backend.build_model(candidate, "high", "arm:all_high", seed=1, task=task)

    for _ in range(sim_backend.LOW_TIER_TOOL_CALLS + 1):
        message = model.generate(_messages(), tools_to_call_from=_tools())
        for call in message.tool_calls or []:
            name = call.function.name
            arguments = call.function.arguments
            assert name in {"calculator", "get_weather", "get_current_time", "final_answer"}
            assert isinstance(arguments, dict), f"{name} arguments must be a mapping"
            if name == "calculator":
                assert "expression" in arguments
            elif name == "get_weather":
                assert "location" in arguments
            elif name == "get_current_time":
                assert "timezone" in arguments
            elif name == "final_answer":
                assert "answer" in arguments


def test_low_tier_makes_at_least_one_non_final_tool_call():
    """Zero non-final calls would make the entire all_low arm score 0%."""
    task = _task("math")
    candidate = get_candidate("quickthink-7b")
    model = sim_backend.build_model(candidate, "low", "arm:all_low", seed=2, task=task)

    message = model.generate(_messages(), tools_to_call_from=_tools())
    names = [c.function.name for c in (message.tool_calls or [])]
    assert any(n != "final_answer" for n in names), (
        f"low tier produced only {names}; SMOLTRACE would grade this as no tool used"
    )


def test_final_answer_is_its_own_step():
    """smolagents raises if a final answer shares a step with other tool calls."""
    task = _task("simpleqa", "simpleqa_0")
    candidate = get_candidate("balanced-13b")
    model = sim_backend.build_model(candidate, "high", "arm:all_high", seed=3, task=task)

    for _ in range(8):
        message = model.generate(_messages(), tools_to_call_from=_tools())
        names = [c.function.name for c in (message.tool_calls or [])]
        if "final_answer" in names:
            assert names == ["final_answer"], f"final_answer was batched with {names}"
            return
    pytest.fail("the backend never produced a final answer")


# ---------------------------------------------------------------------------
# Span emission: the allocation_tier attribute and tier pricing
# ---------------------------------------------------------------------------


def test_generate_emits_a_span_carrying_allocation_tier():
    """Pin the custom attribute and tier-based pricing.

    OTel honours only the first global tracer provider set in a process, so this
    test attaches its sink to whichever provider is live rather than trying to
    replace it — replacing it silently yields an exporter that never records.
    """
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from smoltrace.otel import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = otel_trace.get_tracer_provider()
    if not callable(getattr(provider, "add_span_processor", None)):
        pytest.skip("no SDK tracer provider available in this process")
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    task = _task("gaia")
    candidate = get_candidate("thorough-32b")
    model = sim_backend.build_model(candidate, "mid", "arm:allocated", seed=4, task=task)
    model.generate(_messages(), tools_to_call_from=_tools())

    spans = [s for s in exporter.get_finished_spans() if s.get("name") == "llm.chat"]
    assert spans, "generate() emitted no llm.chat span"
    # SMOLTRACE's exporter returns plain dicts, not span objects.
    attributes = spans[0].get("attributes", {})
    assert attributes.get("allocation_tier") == "mid"
    assert attributes.get("llm.model_name") == sim_backend.tier_api_model("mid")
    assert attributes.get("llm.token_count.total", 0) > 0


def test_prompt_tokens_grow_as_the_conversation_grows():
    """Higher tiers cost more because the re-sent history grows, not by fiat."""
    task = _task("gaia")
    candidate = get_candidate("thorough-32b")
    model = sim_backend.build_model(candidate, "high", "arm:all_high", seed=5, task=task)

    from smolagents.models import ChatMessage, MessageRole

    base = [ChatMessage(role=MessageRole.USER, content="solve this")]
    first = model.generate(base, tools_to_call_from=_tools())

    long_history = base + [
        ChatMessage(role=MessageRole.ASSISTANT, content="x" * 4000),
        ChatMessage(role=MessageRole.USER, content="y" * 4000),
    ]
    second = model.generate(long_history, tools_to_call_from=_tools())

    assert second.token_usage.input_tokens > first.token_usage.input_tokens
