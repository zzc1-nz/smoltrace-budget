"""Verify the OTEL "setup only works once" claim, and validate the fix.

`smoltrace.otel.setup_inmemory_otel` calls `trace.set_tracer_provider()`, which
OpenTelemetry defines as a no-op after the first call in a process. If the runner
built a new session per arm-round, only the first would ever see spans and every
later artifact would silently record zero tokens.

This script proves the failure and confirms the per-round fresh-exporter fix.
"""

from _common import PROJECT_ROOT  # noqa: F401  (bootstrap first)

import json

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from smoltrace.core import extract_traces
from smoltrace.otel import InMemorySpanExporter, setup_inmemory_otel


def emit(tracer, run_id: str, n_spans: int, tier: str = "high") -> None:
    for index in range(n_spans):
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.model_name", "openai/gpt-4o-mini")
            span.set_attribute("gen_ai.request.model", "openai/gpt-4o-mini")
            span.set_attribute("llm.token_count.prompt", 800)
            span.set_attribute("llm.token_count.completion", 200)
            span.set_attribute("llm.token_count.total", 1000)
            span.set_attribute("allocation_tier", tier)
            span.set_attribute("test.id", f"{run_id}-task-{index}")


def main() -> None:
    print("=== NAIVE: one setup_inmemory_otel() call per round (the bug) ===")
    naive = []
    for round_index in range(3):
        run_id = f"naive-{round_index}"
        tracer, _, exporter, _, _, _ = setup_inmemory_otel(
            enable_otel=True, service_name="probe-naive", run_id=run_id
        )
        emit(tracer, run_id, 4)
        traces = extract_traces(exporter, run_id)
        tokens = sum(t.get("total_tokens", 0) for t in traces)
        naive.append((round_index, len(exporter.get_finished_spans()), len(traces), tokens))
        print(f"  round {round_index}: exporter spans={len(exporter.get_finished_spans()):>2} "
              f"n_traces={len(traces):>2} tokens={tokens:>6}")

    broken = sum(1 for _, spans, _, _ in naive[1:] if spans == 0)
    print(f"  -> {broken} of {len(naive) - 1} later rounds are EMPTY (bug reproduced: {broken > 0})")

    print()
    print("=== FIXED: setup once, fresh exporter per round ===")
    tracer, _, _, _, _, _ = setup_inmemory_otel(
        enable_otel=True, service_name="probe-fixed", run_id="bootstrap"
    )
    provider = otel_trace.get_tracer_provider()

    total_tokens = 0
    ok = True
    for round_index in range(3):
        run_id = f"fixed-{round_index}"
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        emit(tracer, run_id, 4, tier=["low", "mid", "high"][round_index])
        traces = extract_traces(exporter, run_id)
        tokens = sum(t.get("total_tokens", 0) for t in traces)
        total_tokens += tokens
        spans = len(exporter.get_finished_spans())
        tiers = {
            span["attributes"].get("allocation_tier")
            for trace in traces
            for span in trace["spans"]
            if span["attributes"].get("allocation_tier")
        }
        print(f"  round {round_index}: spans={spans:>2} n_traces={len(traces):>2} "
              f"tokens={tokens:>6} tiers={sorted(tiers)}")
        if spans == 0 or tokens == 0:
            ok = False

    print(f"  -> all rounds non-empty: {ok}, cumulative tokens={total_tokens}")
    if not ok:
        raise SystemExit("FIX VERIFICATION FAILED")

    print()
    print("VERDICT: the fix works — each arm-round can own its traces.")


if __name__ == "__main__":
    main()
