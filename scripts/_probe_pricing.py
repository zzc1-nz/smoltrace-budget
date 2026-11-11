"""Check which tier model ids genai_otel's CostCalculator can actually price.

SMOLTRACE's `extract_traces` prices spans from `llm.model_name` via
genai_otel's CostCalculator. An unpriced id silently contributes $0, which would
make cost comparisons meaningless — so the tier ids must be verified, not guessed.
"""

from _common import PROJECT_ROOT  # noqa: F401

CANDIDATES = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-3-5-sonnet-20241022",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
]


def main() -> None:
    from genai_otel.cost_calculator import CostCalculator

    calc = CostCalculator()
    usage = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}

    print(f"{'model id':<42} {'cost for 1k in / 500 out':>26}")
    print("-" * 70)
    for model in CANDIDATES:
        try:
            info = calc.calculate_granular_cost(model=model, usage=usage, call_type="chat")
            total = (info or {}).get("total", 0.0)
            marker = "OK " if total > 0 else "ZERO"
            print(f"{model:<42} {marker} ${total:.6f}")
        except Exception as exc:
            print(f"{model:<42} ERROR {type(exc).__name__}: {str(exc)[:40]}")

    print()
    print("For reference, a ~3000-token high-tier task at claude-sonnet-4-5 pricing:")


if __name__ == "__main__":
    main()
