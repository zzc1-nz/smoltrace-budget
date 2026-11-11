"""Integration tests for the SMOLTRACE OTEL contract the runner depends on.

These guard the single most dangerous failure mode in this project: an arm-round
that measures nothing and reports zero tokens. It is dangerous because nothing
raises — ``extract_traces`` on an empty exporter simply returns ``[]``, so a
broken run would produce artifacts that look structurally valid and a cost
report computed against a zero baseline.

The root cause is that ``smoltrace.otel.setup_inmemory_otel`` calls
``opentelemetry.trace.set_tracer_provider``, which OTel specifies as a no-op
after the first call in a process. Calling it once per arm-round therefore
leaves every subsequent exporter permanently empty.
"""

from __future__ import annotations

import pytest

from smoltrace.core import extract_traces
from smoltrace.otel import InMemorySpanExporter, setup_inmemory_otel

PROMPT_TOKENS = 800
COMPLETION_TOKENS = 200


def _emit_llm_spans(tracer, run_id: str, count: int, tier: str = "high") -> None:
    for index in range(count):
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.model_name", "openai/gpt-4o-mini")
            span.set_attribute("gen_ai.request.model", "openai/gpt-4o-mini")
            span.set_attribute("llm.token_count.prompt", PROMPT_TOKENS)
            span.set_attribute("llm.token_count.completion", COMPLETION_TOKENS)
            span.set_attribute("llm.token_count.total", PROMPT_TOKENS + COMPLETION_TOKENS)
            span.set_attribute("allocation_tier", tier)
            span.set_attribute("test.id", f"{run_id}-task-{index}")


def _new_exporter_on_live_provider():
    """Attach a fresh span sink to whichever provider OTel actually records to."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    setup_inmemory_otel(enable_otel=True, service_name="pytest-smoltrace-budget", run_id="bootstrap")
    provider = otel_trace.get_tracer_provider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


# ---------------------------------------------------------------------------
# The bug this project must never ship
# ---------------------------------------------------------------------------


def test_repeated_setup_inmemory_otel_only_records_the_first_time():
    """Documents the OTel footgun: later sessions silently receive no spans.

    If this test ever starts FAILING (i.e. later exporters do get spans), the
    upstream constraint has changed and the runner's workaround may be revisited
    — but it will keep working either way.
    """
    results = []
    for round_index in range(3):
        run_id = f"repeated-{round_index}"
        tracer, _, exporter, _, _, _ = setup_inmemory_otel(
            enable_otel=True, service_name="pytest-repeated", run_id=run_id
        )
        _emit_llm_spans(tracer, run_id, 2)
        results.append(len(exporter.get_finished_spans()))

    # The first setup wins the global provider; every later one is a no-op.
    assert results[0] > 0, "the first setup_inmemory_otel call must record spans"
    assert results[1:] == [0, 0], (
        "expected the OTel 'set tracer provider only once' behaviour; "
        f"got span counts {results}"
    )


def test_fresh_exporter_per_round_records_every_round():
    """The supported workaround: one setup, a new exporter per arm-round."""
    total = 0
    seen_tiers = []
    for round_index, tier in enumerate(("low", "mid", "high")):
        run_id = f"fresh-{round_index}"
        exporter = _new_exporter_on_live_provider()
        tracer = setup_inmemory_otel.__globals__["trace"].get_tracer("pytest-fresh")
        _emit_llm_spans(tracer, run_id, 3, tier=tier)

        traces = extract_traces(exporter, run_id)
        tokens = sum(trace.get("total_tokens", 0) for trace in traces)
        assert traces, f"round {round_index} produced no traces"
        assert tokens > 0, f"round {round_index} produced zero tokens"
        total += tokens
        seen_tiers.append(
            {
                span["attributes"].get("allocation_tier")
                for trace in traces
                for span in trace["spans"]
                if span["attributes"].get("allocation_tier")
            }
        )

    assert total > 0
    assert seen_tiers == [{"low"}, {"mid"}, {"high"}], (
        "each round's traces must carry that round's allocation_tier"
    )


# ---------------------------------------------------------------------------
# The custom attribute the whole project is built on
# ---------------------------------------------------------------------------


def test_allocation_tier_survives_extract_traces():
    """`allocation_tier` must round-trip through SMOLTRACE's own trace extractor."""
    exporter = _new_exporter_on_live_provider()
    run_id = "tier-roundtrip"
    tracer = setup_inmemory_otel.__globals__["trace"].get_tracer("pytest-tier")
    _emit_llm_spans(tracer, run_id, 1, tier="mid")

    traces = extract_traces(exporter, run_id)
    assert traces, "expected a trace"

    attrs = [span["attributes"] for trace in traces for span in trace["spans"]]
    tiers = [a.get("allocation_tier") for a in attrs if a.get("allocation_tier")]
    assert tiers == ["mid"], f"allocation_tier did not survive extraction: {attrs}"


def test_extracted_trace_accounts_tokens_and_cost():
    """Token totals and a non-zero cost come from SMOLTRACE, not our bookkeeping."""
    exporter = _new_exporter_on_live_provider()
    run_id = "token-accounting"
    tracer = setup_inmemory_otel.__globals__["trace"].get_tracer("pytest-tokens")
    _emit_llm_spans(tracer, run_id, 4, tier="high")

    traces = extract_traces(exporter, run_id)
    assert len(traces) == 4

    total_tokens = sum(trace["total_tokens"] for trace in traces)
    assert total_tokens == 4 * (PROMPT_TOKENS + COMPLETION_TOKENS)

    total_cost = sum(trace["total_cost_usd"] for trace in traces)
    assert total_cost > 0, (
        "cost must be priced from llm.model_name; a zero cost means the tier "
        "model id is not in genai_otel's price table"
    )


@pytest.mark.parametrize(
    ("model_id", "should_price"),
    [
        ("openai/gpt-4o-mini", True),
        ("openai/gpt-4o", True),
        ("anthropic/claude-sonnet-4-5", True),
        # Unpriced ids silently produce $0, which would make the all_high arm free.
        ("anthropic/claude-sonnet-4", False),
        ("anthropic/claude-3-5-sonnet", False),
    ],
)
def test_tier_model_ids_are_priced_by_the_cost_calculator(model_id, should_price):
    """Pin which model ids the tier system may use.

    A zero-cost tier would silently invalidate every cost comparison, so the
    priceable ids are asserted rather than assumed.
    """
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = otel_trace.get_tracer_provider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = otel_trace.get_tracer("pytest-pricing")

    run_id = f"price-{model_id.replace('/', '-')}"
    with tracer.start_as_current_span("llm.chat") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", model_id)
        span.set_attribute("gen_ai.request.model", model_id)
        span.set_attribute("llm.token_count.prompt", 1000)
        span.set_attribute("llm.token_count.completion", 500)
        span.set_attribute("llm.token_count.total", 1500)

    traces = extract_traces(exporter, run_id)
    assert traces, "expected a trace"
    cost = sum(trace["total_cost_usd"] for trace in traces)

    if should_price:
        assert cost > 0, f"{model_id} must be priceable but produced $0"
    else:
        assert cost == 0, (
            f"{model_id} is expected to be UNPRICED by genai_otel; if it now has a "
            "price this test is stale but the tier config is still valid"
        )
