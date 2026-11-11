"""Tests for the Markdown cost-report renderer.

The renderer is fed by ``analysis.build_cost_report``; these tests pin the two
shapes it must tolerate (``tokens``/``cost`` and ``total_tokens``/``total_cost_usd``)
so a key rename on either side fails loudly here instead of silently rendering
zeros in the published report.
"""

from smoltrace_budget.report_render import (
    render_category_table,
    render_cost_report_markdown,
    render_cost_table,
    render_leaderboard_table,
    render_stability_section,
)


def _entry(rank, model, rate, tokens=1000, cost=0.01):
    return {
        "rank": rank,
        "model": model,
        "pass_rate": rate,
        "passed": 5,
        "evaluated": 10,
        "total_tokens": tokens,
        "total_cost_usd": cost,
        "avg_latency_ms": 800.0,
    }


# ---------------------------------------------------------------------------
# Leaderboard table
# ---------------------------------------------------------------------------


def test_leaderboard_table_renders_every_row():
    table = render_leaderboard_table(
        [_entry(1, "thorough-32b", 69.6), _entry(2, "balanced-13b", 61.9)],
        title="all_high",
    )
    assert "thorough-32b" in table
    assert "balanced-13b" in table
    assert "69.60%" in table
    assert "| Rank | Model |" in table
    assert table.count("\n") >= 4  # title, blank, header, separator, rows


def test_leaderboard_table_handles_empty_input():
    assert "no leaderboard rows" in render_leaderboard_table([])


# ---------------------------------------------------------------------------
# Cost table: sign convention is the thing most likely to mislead a reader
# ---------------------------------------------------------------------------


def test_cost_table_phrases_savings_as_cheaper_not_negative():
    tradeoff = {"baseline_cost_usd": 100.0}
    totals = {
        "all_high": {"total_tokens": 10_000, "total_cost_usd": 100.0},
        "all_low": {"total_tokens": 2_000, "total_cost_usd": 10.0},
        "allocated": {"total_tokens": 4_000, "total_cost_usd": 40.0},
    }
    table = render_cost_table(tradeoff, totals)
    assert "cheaper by 90.0%" in table
    assert "cheaper by 60.0%" in table
    assert "baseline" in table
    # A reader must never see a bare -90.0% and interpret it as a cost increase.
    assert "-90.0%" not in table


def test_cost_table_flags_a_more_expensive_arm():
    tradeoff = {"baseline_cost_usd": 10.0}
    totals = {
        "all_high": {"total_tokens": 1, "total_cost_usd": 10.0},
        "allocated": {"total_tokens": 1, "total_cost_usd": 15.0},
    }
    table = render_cost_table(tradeoff, totals)
    assert "more expensive" in table
    assert "50.0%" in table


def test_cost_table_without_baseline_does_not_claim_a_saving():
    totals = {"allocated": {"total_tokens": 1, "total_cost_usd": 5.0}}
    table = render_cost_table({}, totals)
    assert "n/a (no baseline)" in table
    assert "cheaper" not in table


def test_cost_table_accepts_both_key_shapes():
    tradeoff = {"baseline_cost_usd": 100.0}
    totals = {
        "all_high": {"total_tokens": 10_000, "total_cost_usd": 100.0},
        "allocated": {"tokens": 3_000, "cost": 25.0},
    }
    table = render_cost_table(tradeoff, totals)
    assert "3,000" in table
    assert "$25.0000" in table
    assert "cheaper by 75.0%" in table


def test_cost_table_marks_an_arm_with_no_records_as_absent():
    """An arm that produced nothing must not read as '100% cheaper'."""
    tradeoff = {"baseline_cost_usd": 100.0}
    totals = {
        "all_high": {"total_tokens": 10_000, "total_cost_usd": 100.0},
        "allocated": {"total_tokens": 0, "total_cost_usd": 0.0},
    }
    table = render_cost_table(tradeoff, totals)
    assert "not run" in table
    assert "cheaper by 100.0%" not in table


# ---------------------------------------------------------------------------
# Category table: the `tokens`/`cost` vs `total_*` bug
# ---------------------------------------------------------------------------


def test_category_table_reads_canonical_tokens_and_cost_keys():
    by_category = {
        "gaia": {
            "n_records": 96,
            "tokens": 400_000,
            "cost": 12.5,
            "share_of_tokens": 40.0,
            "share_of_cost": 62.5,
        }
    }
    table = render_category_table(by_category)
    assert "400,000" in table
    assert "$12.5000" in table
    assert "gaia" in table


def test_category_table_still_reads_total_aliases():
    by_category = {
        "math": {
            "n_records": 10,
            "total_tokens": 123_456,
            "total_cost_usd": 1.25,
            "share_of_tokens": 10.0,
            "share_of_cost": 5.0,
        }
    }
    table = render_category_table(by_category)
    assert "123,456" in table
    assert "$1.2500" in table


def test_category_table_never_silently_renders_zero_for_present_data():
    """Regression guard: a key mismatch used to print 0 / $0.0000."""
    table = render_category_table(
        {"gaia": {"n_records": 1, "tokens": 999, "cost": 7.0}}
    )
    assert "999" in table
    assert "0 | $0.0000" not in table


# ---------------------------------------------------------------------------
# Stability section: honest denominators
# ---------------------------------------------------------------------------


def test_stability_section_uses_supplied_denominator_for_small_pools():
    stability = {
        "all_low": {
            "top3_membership_overlap": 1,
            "top3_membership_overlap_denominator": 2,
            "top3_ordering_identical": False,
            "kendall_tau": 0.2,
            "stability_pct": 60.0,
            "stability_delta_pct": -40.0,
        }
    }
    table = render_stability_section(stability)
    assert "1/2" in table
    assert "1/3" not in table


def test_stability_section_defaults_to_three_when_denominator_absent():
    stability = {
        "allocated": {
            "top3_membership_overlap": 3,
            "top3_ordering_identical": True,
            "kendall_tau": 1.0,
            "stability_pct": 100.0,
            "stability_delta_pct": 0.0,
        }
    }
    table = render_stability_section(stability)
    assert "3/3" in table
    assert "yes" in table


def test_stability_section_renders_missing_tau_as_na():
    stability = {
        "all_low": {
            "top3_membership_overlap": 2,
            "top3_ordering_identical": False,
            "kendall_tau": None,
            "stability_pct": 70.0,
            "stability_delta_pct": -30.0,
        }
    }
    assert "n/a" in render_stability_section(stability)


# ---------------------------------------------------------------------------
# Full document
# ---------------------------------------------------------------------------


def test_full_report_contains_the_headline_and_all_sections():
    report = {
        "headline": "Allocation cuts evaluation cost by 62.1% while top-3 ranking is unchanged.",
        "cost_rank_tradeoff": {
            "baseline_cost_usd": 100.0,
            "interpretation": "Cheaper without changing the top of the leaderboard.",
        },
        "arm_totals": {
            "all_high": {"total_tokens": 10_000, "total_cost_usd": 100.0},
            "all_low": {"total_tokens": 2_000, "total_cost_usd": 10.0},
            "allocated": {"total_tokens": 4_000, "total_cost_usd": 40.0},
        },
        "leaderboards": {
            "all_high": {"mean": [_entry(1, "thorough-32b", 69.6)]},
            "all_low": {"mean": [_entry(1, "quickthink-7b", 58.6)]},
            "allocated": {"mean": [_entry(1, "thorough-32b", 69.5)]},
        },
        "rank_stability": {
            "allocated": {
                "top3_membership_overlap": 3,
                "top3_ordering_identical": True,
                "kendall_tau": 1.0,
                "stability_pct": 100.0,
                "stability_delta_pct": 0.0,
            }
        },
        "cheap_arm_hallucination": {
            "same": False,
            "all_high_winner": "thorough-32b",
            "all_low_winner": "quickthink-7b",
            "true_winner_pass_rate_all_high": 69.6,
            "wrongly_crowned_pass_rate_all_high": 58.6,
            "explanation": "Cheap evaluation rewards models that need no tools.",
        },
        "cost_by_category": {
            "all_high": {
                "gaia": {"n_records": 1, "tokens": 4000, "cost": 2.0, "share_of_tokens": 40.0, "share_of_cost": 50.0}
            }
        },
    }

    doc = render_cost_report_markdown(report)

    assert doc.startswith("# Evaluation Cost Report")
    assert "## Headline" in doc
    assert "62.1%" in doc
    assert "## Cost of running the evaluation" in doc
    assert "cheaper by 90.0%" in doc
    assert "## Three leaderboards" in doc
    assert "all_high" in doc and "all_low" in doc and "allocated" in doc
    assert "## Reading" not in doc  # section names stay stable
    assert "## Ranking stability versus the gold standard" in doc
    assert "## What cheap evaluation gets wrong" in doc
    assert "thorough-32b" in doc
    assert "quickthink-7b" in doc
    assert "## Cost distribution by task category" in doc
    assert "## Method note" in doc


def test_full_report_degrades_gracefully_when_nothing_is_computed():
    doc = render_cost_report_markdown({})
    assert "# Evaluation Cost Report" in doc
    assert "## Method note" in doc


def test_report_states_same_winner_when_cheap_arm_agrees():
    report = {
        "cheap_arm_hallucination": {
            "same": True,
            "all_high_winner": "thorough-32b",
            "all_low_winner": "thorough-32b",
        }
    }
    doc = render_cost_report_markdown(report)
    assert "same model first" in doc
    assert "**The cheap arm crowns a different winner" not in doc
