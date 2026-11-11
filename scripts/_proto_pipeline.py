"""Prototype: drive SMOLTRACE's real pipeline with a custom smolagents Model.

Verifies the three things the whole project depends on:
  1. a custom `smolagents.Model` subclass can run inside SMOLTRACE's agent loop;
  2. the LLM spans our backend emits are picked up by `smoltrace.core.extract_traces`
     (so token accounting comes from SMOLTRACE, not from our own bookkeeping);
  3. `smoltrace.utils.compute_leaderboard_row` turns those traces into a real row.
"""

from _common import PROJECT_ROOT  # noqa: F401  (bootstrap + sys.path first)

import json

from opentelemetry import trace as otel_trace
from smolagents import Model
from smolagents.models import ChatMessage, MessageRole

from smoltrace.core import extract_metrics, extract_traces, setup_inmemory_otel, initialize_agent, evaluate_single_test
from smoltrace.utils import compute_leaderboard_row

TASKS = PROJECT_ROOT / "data" / "tasks" / "benchmark_v1.jsonl"


class InstrumentedEchoModel(Model):
    """Emits a GenAI-shaped LLM span per call and answers with a final_answer call."""

    def __init__(self, model_id: str = "openai/gpt-4o-mini", allocation_tier: str = "low", **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.allocation_tier = allocation_tier
        self.calls = 0

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self.calls += 1
        tracer = otel_trace.get_tracer("prototype")
        prompt_text = json.dumps(
            [{"role": m.role.value if hasattr(m.role, "value") else str(m.role), "content": str(m.content)} for m in messages],
            default=str,
        )
        prompt_tokens = max(1, len(prompt_text) // 4)
        completion_tokens = 24

        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.model_name", self.model_id)
            span.set_attribute("gen_ai.request.model", self.model_id)
            span.set_attribute("llm.token_count.prompt", prompt_tokens)
            span.set_attribute("llm.token_count.completion", completion_tokens)
            span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)
            span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
            span.set_attribute("allocation_tier", self.allocation_tier)

        answer = "final_answer(\"3\")"
        return ChatMessage(role=MessageRole.ASSISTANT, content=answer)


def main() -> None:
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"tasks loaded: {len(tasks)}")
    task = next(t for t in tasks if t["category"] == "simpleqa")

    tracer, meter, span_exporter, metric_exporter, aggregator, run_id = setup_inmemory_otel(
        enable_otel=True, service_name="prototype", run_id="proto-run-1"
    )
    print("otel run_id:", run_id)

    model = InstrumentedEchoModel(model_id="openai/gpt-4o-mini", allocation_tier="low")
    agent = initialize_agent(
        model_name="unused-by-custom-model",
        agent_type="tool",
        provider="litellm",
        model_instance=model,
    )
    print("agent:", type(agent).__name__, "tools:", len(agent.tools))

    result = evaluate_single_test(agent, task.copy(), "tool", tracer, None, verbose=False)
    print("\n--- result ---")
    print(json.dumps({k: result[k] for k in ("test_id", "success", "steps", "trace_id", "stop_reason")}, indent=2))
    print("response:", result["response"][:120])

    trace_data = extract_traces(span_exporter, run_id)
    print("\n--- traces ---")
    print("n_traces:", len(trace_data))
    for tr in trace_data:
        print(
            f"  trace {tr['trace_id'][:14]} spans={len(tr['spans'])} "
            f"tokens={tr['total_tokens']} cost=${tr['total_cost_usd']:.6f} "
            f"prompt={tr['total_prompt_tokens']} completion={tr['total_completion_tokens']}"
        )
        for sp in tr["spans"]:
            attrs = sp.get("attributes", {})
            print(f"     span {sp['name']!r} tier={attrs.get('allocation_tier')} model={attrs.get('llm.model_name')}")

    metric_data = extract_metrics(metric_exporter, aggregator, trace_data, {"tool": [result], "code": []}, run_id)

    row = compute_leaderboard_row(
        model_name="openai/gpt-4o-mini",
        all_results={"tool": [result], "code": []},
        trace_data=trace_data,
        metric_data=metric_data,
        dataset_used="kshitijthakkar/smoltrace-benchmark-v1",
        results_dataset="proto/results",
        traces_dataset="proto/traces",
        metrics_dataset="proto/metrics",
        agent_type="tool",
        run_id=run_id,
        provider="litellm",
    )
    print("\n--- leaderboard row (SMOLTRACE-computed) ---")
    for key in (
        "model",
        "pass_rate",
        "success_rate",
        "num_tests",
        "total_tokens",
        "total_cost_usd",
        "avg_tokens_per_test",
        "avg_cost_per_test_usd",
        "avg_duration_ms",
    ):
        if key in row:
            print(f"  {key}: {row[key]}")


if __name__ == "__main__":
    main()
