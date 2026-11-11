"""Render the cost report as a standalone, readable Markdown document.

``analysis.build_cost_report`` produces the numbers; this module turns them into
the document a human actually reads, with the three leaderboard tables, the cost
breakdown, the rank-stability analysis and the headline finding.

Kept separate from ``analysis`` so the numbers stay testable without any
formatting concerns, and so the README can quote either.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["render_cost_report_markdown", "render_leaderboard_table", "render_cost_table"]

ARM_TITLES = {
    "all_high": "(a) all_high — every task at the high tier (gold standard)",
    "all_low": "(b) all_low — every task at the low tier (cheapest)",
    "allocated": "(c) allocated — budget allocator assigns a tier per task",
}

TIER_LABELS = {"low": "gpt-4o-mini", "mid": "gpt-4o", "high": "claude-sonnet"}


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_money(value: Any, digits: int = 4) -> str:
    try:
        return f"${float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def render_leaderboard_table(entries: List[Dict[str, Any]], *, title: str = "") -> str:
    """Markdown table for one leaderboard."""
    lines: List[str] = []
    if title:
        lines.append(f"**{title}**")
        lines.append("")

    if not entries:
        lines.append("_no leaderboard rows_")
        return "\n".join(lines)

    lines.append("| Rank | Model | Pass rate | Passed | Evaluated | Tokens | Cost | Avg latency |")
    lines.append("|-----:|-------|----------:|-------:|----------:|-------:|-----:|------------:|")
    for row in entries:
        lines.append(
            "| {rank} | {model} | {rate} | {passed} | {evaluated} | {tokens} | {cost} | {latency} ms |".format(
                rank=row.get("rank", ""),
                model=row.get("model", ""),
                rate=_fmt_pct(row.get("pass_rate", 0.0), 2),
                passed=_fmt_int(row.get("passed", 0)),
                evaluated=_fmt_int(row.get("evaluated", 0)),
                tokens=_fmt_int(row.get("total_tokens", 0)),
                cost=_fmt_money(row.get("total_cost_usd", 0.0), 4),
                latency=f"{float(row.get('avg_latency_ms', 0.0)):.0f}",
            )
        )
    return "\n".join(lines)


def render_cost_table(cost_tradeoff: Dict[str, Any], arm_totals: Dict[str, Dict[str, Any]]) -> str:
    """Markdown table comparing total cost across the three arms.

    The delta column is phrased as a *saving* ("cheaper by 90.0%") rather than a
    signed percentage, because a negative percentage against a baseline reads as
    a cost increase to most people.
    """
    lines = [
        "| Arm | Total tokens | Total cost | Cost vs gold standard |",
        "|-----|-------------:|-----------:|----------------------:|",
    ]
    baseline = float(cost_tradeoff.get("baseline_cost_usd") or 0.0)
    for arm in ("all_high", "all_low", "allocated"):
        totals = arm_totals.get(arm, {})
        cost = float(totals.get("total_cost_usd") or totals.get("cost") or 0.0)
        tokens = totals.get("total_tokens", totals.get("tokens", 0))
        if not totals or (not cost and not tokens):
            # An arm that produced no records is absent, not free.
            delta = "not run"
        elif arm == "all_high":
            delta = "baseline"
        elif baseline > 0:
            delta_pct = (cost - baseline) / baseline * 100.0
            if delta_pct < 0:
                delta = f"cheaper by {abs(delta_pct):.1f}%"
            elif delta_pct > 0:
                delta = f"**more expensive** by {delta_pct:.1f}%"
            else:
                delta = "same cost"
        else:
            delta = "n/a (no baseline)"
        lines.append(f"| `{arm}` | {_fmt_int(tokens)} | {_fmt_money(cost)} | {delta} |")
    return "\n".join(lines)


def render_category_table(cost_by_category: Dict[str, Dict[str, Any]]) -> str:
    """Markdown table of the cost distribution by task category."""
    if not cost_by_category:
        return "_no category data_"
    lines = [
        "| Category | Records | Tokens | Token share | Cost | Cost share |",
        "|----------|--------:|-------:|------------:|-----:|-----------:|",
    ]
    for category in sorted(cost_by_category):
        row = cost_by_category[category]
        # analysis emits `tokens`/`cost`; the `total_*` names are accepted too so
        # the renderer works against either shape.
        tokens = row.get("tokens", row.get("total_tokens", 0))
        cost = row.get("cost", row.get("total_cost_usd", 0.0))
        lines.append(
            "| {cat} | {n} | {tok} | {tokp} | {cost} | {costp} |".format(
                cat=category,
                n=_fmt_int(row.get("n_records", 0)),
                tok=_fmt_int(tokens),
                tokp=_fmt_pct(row.get("share_of_tokens", 0.0)),
                cost=_fmt_money(cost, 4),
                costp=_fmt_pct(row.get("share_of_cost", 0.0)),
            )
        )
    return "\n".join(lines)


def render_stability_section(stability: Dict[str, Any]) -> str:
    """Markdown narrative + table for the rank-stability analysis."""
    lines: List[str] = []
    lines.append("| Arm | Top-3 overlap | Ordering identical | Kendall τ | Stability | Δ vs gold |")
    lines.append("|-----|--------------:|-------------------:|----------:|----------:|----------:|")

    for arm in ("all_low", "allocated"):
        entry = stability.get(arm) or stability.get("arms", {}).get(arm) or {}
        if not entry:
            continue
        overlap = entry.get("top3_membership_overlap", 0)
        # A 2-model pool cannot have a top-3: report the honest denominator.
        denominator = entry.get("top3_membership_overlap_denominator") or 3
        tau = entry.get("kendall_tau")
        lines.append(
            "| `{arm}` | {overlap}/{den} | {ident} | {tau} | {stab} | {delta} |".format(
                arm=arm,
                overlap=overlap,
                den=denominator,
                ident="yes" if entry.get("top3_ordering_identical") else "no",
                tau="n/a" if tau is None else f"{float(tau):.3f}",
                stab=_fmt_pct(entry.get("stability_pct", 0.0)),
                delta=_fmt_pct(entry.get("stability_delta_pct", 0.0)),
            )
        )
    return "\n".join(lines)


def render_cost_report_markdown(report: Dict[str, Any]) -> str:
    """Full Markdown cost report from ``analysis.build_cost_report`` output."""
    out: List[str] = []
    out.append("# Evaluation Cost Report")
    out.append("")
    out.append(
        "Generated by the cost-aware evaluation scheduling layer on top of "
        "SMOLTRACE. All token and cost figures come from SMOLTRACE's own "
        "OpenTelemetry spans, priced against the real model id of each "
        "evaluation tier."
    )
    out.append("")

    headline = report.get("headline")
    if headline:
        out.append("## Headline")
        out.append("")
        out.append(f"> {headline}")
        out.append("")

    # --- cost ---
    tradeoff = report.get("cost_rank_tradeoff") or {}
    arm_totals = report.get("arm_totals") or report.get("totals") or {}
    if tradeoff or arm_totals:
        out.append("## Cost of running the evaluation")
        out.append("")
        out.append(render_cost_table(tradeoff, arm_totals))
        out.append("")
        if tradeoff.get("interpretation"):
            out.append(str(tradeoff["interpretation"]))
            out.append("")

    # --- the three leaderboards ---
    leaderboards = report.get("leaderboards") or {}
    out.append("## Three leaderboards")
    out.append("")
    out.append(
        "The same candidate pool is ranked in every arm, so the tables are "
        "directly comparable. Only the evaluation fidelity differs."
    )
    out.append("")
    for arm in ("all_high", "all_low", "allocated"):
        arm_lb = leaderboards.get(arm)
        if not arm_lb:
            continue
        entries = arm_lb.get("mean") if isinstance(arm_lb, dict) else arm_lb
        if isinstance(entries, dict):
            entries = list(entries.values())
        out.append(f"### {ARM_TITLES.get(arm, arm)}")
        out.append("")
        out.append(render_leaderboard_table(entries or []))
        out.append("")

    # --- stability ---
    stability = report.get("rank_stability") or {}
    if stability:
        out.append("## Ranking stability versus the gold standard")
        out.append("")
        out.append(render_stability_section(stability))
        out.append("")

    # --- the headline finding ---
    hallucination = report.get("cheap_arm_hallucination") or {}
    if hallucination:
        out.append("## What cheap evaluation gets wrong")
        out.append("")
        if hallucination.get("same"):
            out.append(
                "The cheap arm ranked the same model first as the gold standard, "
                "so in this run cheap evaluation did not invert the top of the "
                "leaderboard."
            )
        else:
            out.append(
                "**The cheap arm crowns a different winner than the gold standard.**"
            )
            out.append("")
            out.append(
                f"- Gold standard winner: **{hallucination.get('all_high_winner')}** "
                f"(pass rate {_fmt_pct(hallucination.get('true_winner_pass_rate_all_high', 0.0))})"
            )
            out.append(
                f"- Cheap-evaluation winner: **{hallucination.get('all_low_winner')}** "
                f"(pass rate {_fmt_pct(hallucination.get('wrongly_crowned_pass_rate_all_high', 0.0))} under the gold standard)"
            )
        if hallucination.get("explanation"):
            out.append("")
            out.append(str(hallucination["explanation"]))
        out.append("")

    # --- category cost distribution ---
    by_category = report.get("cost_by_category") or {}
    if by_category:
        out.append("## Cost distribution by task category")
        out.append("")
        for arm in ("all_high", "allocated"):
            arm_cat = by_category.get(arm)
            if not arm_cat:
                continue
            out.append(f"**{arm}**")
            out.append("")
            out.append(render_category_table(arm_cat))
            out.append("")

    out.append("## Method note")
    out.append("")
    out.append(
        "Fidelity tiers bundle measurement knobs, not models: `low` "
        "(max_steps=2, tools off, keyword grader), `mid` (max_steps=4, tools on, "
        "keyword+semantic), `high` (max_steps=6, all tools, strict semantic). "
        "Each tier is priced at its real model class — "
        + ", ".join(f"`{k}` = {v}" for k, v in TIER_LABELS.items())
        + " — because that is what you would pay to obtain that fidelity. Cost "
        "therefore follows the tier, while the *ranking* follows the candidate "
        "models being evaluated. That separation is the whole point."
    )
    out.append("")
    return "\n".join(out)
