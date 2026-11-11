"""Audit every number quoted in README.md against the generated report.

A published report that disagrees with its own artifacts is worse than no report,
so this compares the README's claims to artifacts/report/cost_report.json and to
the raw records, and fails loudly on any mismatch.
"""

from _common import PROJECT_ROOT  # noqa: F401

import json
import re

PROJECT = PROJECT_ROOT
REPORT = PROJECT / "artifacts" / "report" / "cost_report.json"
README = PROJECT / "README.md"
RECORDS = PROJECT / "artifacts" / "records"


def load_records(arm: str):
    rows = []
    for path in sorted(RECORDS.glob(f"{arm}_r*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")

    checks: list[tuple[str, bool, str]] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        checks.append((label, condition, detail))

    tradeoff = report["cost_rank_tradeoff"]
    totals = report["totals_by_arm"]
    # Stability is nested under "arms"; the reference arm's own entry is implicit.
    stability = report["rank_stability"]["arms"]

    # --- headline cost figures -------------------------------------------
    check(
        "cost reduction 56.79%",
        f"{tradeoff['cost_reduction_pct']:.2f}" == "56.79",
        f"report says {tradeoff['cost_reduction_pct']}",
    )
    check(
        "README quotes 56.79%",
        "56.79%" in readme,
        "string missing from README",
    )
    check(
        "all_high cost $20.3497",
        f"{totals['all_high']['total_cost_usd']:.4f}" == "20.3497",
        f"report says {totals['all_high']['total_cost_usd']}",
    )
    check("README quotes $20.3497", "$20.3497" in readme)
    check(
        "allocated cost $8.7925",
        f"{totals['allocated']['total_cost_usd']:.4f}" == "8.7925",
        f"report says {totals['allocated']['total_cost_usd']}",
    )
    check("README quotes $8.7925", "$8.7925" in readme)
    check(
        "all_low cost $0.6975",
        f"{totals['all_low']['total_cost_usd']:.4f}" == "0.6975",
        f"report says {totals['all_low']['total_cost_usd']}",
    )
    check("README quotes $0.6975", "$0.6975" in readme)

    # --- token totals -----------------------------------------------------
    for arm, expected in (
        ("all_high", 11_375_912),
        ("all_low", 8_169_544),
        ("allocated", 11_375_786),
    ):
        actual = totals[arm]["total_tokens"]
        check(f"{arm} tokens {expected:,}", actual == expected, f"report says {actual:,}")
        check(f"README quotes {expected:,}", f"{expected:,}" in readme)

    # --- stability --------------------------------------------------------
    # Compare to what the report *renders* (2dp formatting of the stored value),
    # not to eyeballed rounding: 68.335 formats to 68.33.
    def rendered_2dp(value: float) -> str:
        return format(value, ".2f")

    check(
        "allocated stability renders 68.33%",
        rendered_2dp(stability["allocated"]["stability_pct"]) == "68.33",
        f"renders {rendered_2dp(stability['allocated']['stability_pct'])} "
        f"from {stability['allocated']['stability_pct']}",
    )
    check("README quotes 68.33%", "68.33%" in readme)
    check(
        "allocated delta -31.66",
        rendered_2dp(stability["allocated"]["stability_delta_pct"]) == "-31.66",
        f"renders {rendered_2dp(stability['allocated']['stability_delta_pct'])}",
    )
    check("README quotes 31.66", "31.66" in readme)
    check(
        "all_low stability renders 63.34%",
        rendered_2dp(stability["all_low"]["stability_pct"]) == "63.34",
        f"renders {rendered_2dp(stability['all_low']['stability_pct'])}",
    )
    check("README quotes 63.34%", "63.34%" in readme)

    # --- the wrong-winner claim ------------------------------------------
    halluc = report["cheap_arm_hallucination"]
    check("gold winner is thorough-32b", halluc["all_high_winner"] == "thorough-32b",
          f"report says {halluc['all_high_winner']}")
    check("cheap winner is balanced-13b", halluc["all_low_winner"] == "balanced-13b",
          f"report says {halluc['all_low_winner']}")
    check("winners differ", halluc["same"] is False)

    # --- the leaderboard tables ------------------------------------------
    pooled = report["pooled_leaderboards"]
    for arm in ("all_high", "all_low", "allocated"):
        rows = pooled[arm]
        check(f"{arm} has 5 ranked models", len(rows) == 5, f"has {len(rows)}")
        for row in rows:
            rate_str = f"{row['pass_rate']:.2f}%"
            check(f"{arm}/{row['model']} rate {rate_str}", rate_str in readme,
                  f"{rate_str} not found in README")

    # --- rank-1 winners per arm ------------------------------------------
    check("all_high rank1 thorough-32b", pooled["all_high"][0]["model"] == "thorough-32b")
    check("all_low rank1 balanced-13b", pooled["all_low"][0]["model"] == "balanced-13b")
    check("allocated rank1 balanced-13b", pooled["allocated"][0]["model"] == "balanced-13b")

    # --- thorough-32b's fall from 1st to 4th ------------------------------
    low_rows = {r["model"]: r for r in pooled["all_low"]}
    check("thorough-32b is rank 4 in all_low", low_rows["thorough-32b"]["rank"] == 4,
          f"rank {low_rows['thorough-32b']['rank']}")
    check(
        "thorough-32b all_low rate 47.22%",
        f"{low_rows['thorough-32b']['pass_rate']:.2f}" == "47.22",
        f"report says {low_rows['thorough-32b']['pass_rate']}",
    )
    check("README quotes 47.22%", "47.22%" in readme)

    # --- the price-effect claim ------------------------------------------
    h_tokens = totals["all_high"]["total_tokens"]
    a_tokens = totals["allocated"]["total_tokens"]
    token_delta_pct = (a_tokens - h_tokens) / h_tokens * 100
    check(
        "allocated token change is ~0%",
        abs(token_delta_pct) < 0.01,
        f"actual {token_delta_pct:.4f}%",
    )
    h_price = totals["all_high"]["total_cost_usd"] / h_tokens * 1e6
    a_price = totals["allocated"]["total_cost_usd"] / a_tokens * 1e6
    check("README price $1.79/Mtok", f"{h_price:.2f}" == "1.79", f"actual {h_price:.4f}")
    check("README price $0.77/Mtok", f"{a_price:.2f}" == "0.77", f"actual {a_price:.4f}")

    # --- the 20x underestimated-cost finding ------------------------------
    n_records = report["n_records_total"]
    check("5940 evaluations", n_records == 5940, f"report says {n_records}")
    check("README quotes 5,940", "5,940" in readme)
    total_tokens_all = n_records and sum(t["total_tokens"] for t in totals.values())
    check(
        "~20x over the 1.5-2.0M estimate",
        15 <= total_tokens_all / 2_000_000 <= 25,
        f"{total_tokens_all / 2_000_000:.1f}x the upper estimate",
    )
    check(
        "README quotes 20x",
        "20×" in readme or "20x" in readme,
    )

    # --- GAIA absorbing 84% of allocated budget --------------------------
    alloc_cat = report["cost_by_category"]["allocated"]
    gaia_cost_share = alloc_cat["gaia"]["share_of_cost"]
    check(
        "gaia cost share 83.9%",
        f"{gaia_cost_share:.1f}" == "83.9",
        f"report says {gaia_cost_share}",
    )
    gaia_token_share = alloc_cat["gaia"]["share_of_tokens"]
    check(
        "gaia token share 36.3%",
        f"{gaia_token_share:.1f}" == "36.3",
        f"report says {gaia_token_share}",
    )

    # --- raw records agree with the report -------------------------------
    raw_tokens = sum(r["total_tokens"] for r in load_records("all_high"))
    check(
        "raw all_high records match report tokens",
        raw_tokens == totals["all_high"]["total_tokens"],
        f"raw {raw_tokens:,} vs report {totals['all_high']['total_tokens']:,}",
    )

    # --- every arm evaluated the same pairs ------------------------------
    def pair_set(arm):
        return {(r["round_index"], r["task_id"], r["candidate_model"]) for r in load_records(arm)}

    high_pairs = pair_set("all_high")
    for arm in ("all_low", "allocated"):
        check(f"{arm} covers identical (round, task, model) pairs",
              pair_set(arm) == high_pairs,
              f"{len(pair_set(arm))} vs {len(high_pairs)}")

    # --- per-round honesty table ------------------------------------------
    per_round = report["per_round"]["by_arm"]
    alloc_rounds = per_round["allocated"]
    check(
        "allocated per-round stability spread 15.00",
        f"{alloc_rounds['spread_pct']:.2f}" == "15.00",
        f"report says {alloc_rounds['spread_pct']}",
    )
    check("README quotes 15.00 spread", "15.00" in readme)
    for value in alloc_rounds["stability_pct_per_round"]:
        rendered = f"{value:.2f}"
        check(f"README quotes per-round stability {rendered}", rendered in readme)
    check(
        "README range low 58.34", f"{alloc_rounds['min_stability_pct']:.2f}" == "58.34"
    )
    check(
        "README range high 73.33", f"{alloc_rounds['max_stability_pct']:.2f}" == "73.33"
    )
    for arm in ("all_high", "all_low"):
        check(
            f"{arm} fixed arm has zero spread (fully reproducible)",
            per_round[arm]["spread_pct"] == 0.0,
            f"spread {per_round[arm]['spread_pct']}",
        )

    # --- report ----------------------------------------------------------
    failed = [c for c in checks if not c[1]]
    print(f"{'PASS' if not failed else 'FAIL'}  {len(checks) - len(failed)}/{len(checks)} checks")
    print()
    for label, ok, detail in checks:
        if not ok:
            print(f"  FAIL  {label}   {detail}")
    if failed:
        raise SystemExit(f"{len(failed)} README/report inconsistencies")


if __name__ == "__main__":
    main()
