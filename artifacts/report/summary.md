# Cost-Aware Evaluation Scheduling — Analysis Report

Arms: `all_high`, `all_low`, `allocated` · Rounds: 3 · Records: 5,940 · Gold standard: `all_high`

## Headline

**Cost down 56.79%, top-3 rank stability changes -31.66% versus the all_high gold standard. Measured cheaply, the leaderboard wrongly crowns balanced-13b first.**

Cheap evaluation wrongly crowns balanced-13b first (56.8182% pass rate under all_low), but under the all_high gold standard balanced-13b is #2 of 5 at 62.6263%; the true winner is thorough-32b at 71.9697%. Measuring every task cheaply flips the top of the leaderboard.

## Leaderboards

Ranked by pass rate over an identical model pool. Ordering is `pass_rate` descending, then `total_cost_usd` ascending, then model name; rounds are pooled here, per-round tables are in the appendix.

### `all_high` — every task measured at the high tier (gold standard)

| rank | model | pass rate % | passed / evaluated | tokens | cost USD | avg latency ms |
|---|---|---|---|---|---|---|
| 1 | `thorough-32b` | 71.97 | 285 / 396 | 5,281,154 | 9.8742 | 169.3 |
| 2 | `balanced-13b` | 62.63 | 248 / 396 | 1,943,620 | 3.3887 | 107.5 |
| 3 | `quickthink-7b` | 58.08 | 230 / 396 | 1,079,030 | 1.7447 | 98.7 |
| 4 | `mathwhiz-9b` | 56.57 | 224 / 396 | 2,192,268 | 3.9365 | 106.4 |
| 5 | `tinyfast-3b` | 33.08 | 131 / 396 | 879,840 | 1.4055 | 95.5 |

### `all_low` — every task measured at the low tier (cheapest possible)

| rank | model | pass rate % | passed / evaluated | tokens | cost USD | avg latency ms |
|---|---|---|---|---|---|---|
| 1 | `balanced-13b` | 56.82 | 225 / 396 | 1,705,826 | 0.1437 | 119.1 |
| 2 | `mathwhiz-9b` | 56.06 | 222 / 396 | 1,924,326 | 0.1660 | 120.4 |
| 3 | `quickthink-7b` | 53.28 | 211 / 396 | 1,078,954 | 0.0856 | 119.2 |
| 4 | `thorough-32b` | 47.22 | 187 / 396 | 2,580,564 | 0.2329 | 123.1 |
| 5 | `tinyfast-3b` | 36.36 | 144 / 396 | 879,874 | 0.0692 | 120.7 |

### `allocated` — the budget allocator assigns a tier per task

| rank | model | pass rate % | passed / evaluated | tokens | cost USD | avg latency ms |
|---|---|---|---|---|---|---|
| 1 | `balanced-13b` | 63.89 | 253 / 396 | 1,943,598 | 1.3806 | 119.4 |
| 2 | `mathwhiz-9b` | 63.38 | 251 / 396 | 2,192,272 | 1.5986 | 125.7 |
| 3 | `thorough-32b` | 61.87 | 245 / 396 | 5,281,018 | 4.7744 | 188.3 |
| 4 | `quickthink-7b` | 60.86 | 241 / 396 | 1,079,044 | 0.5750 | 107.2 |
| 5 | `tinyfast-3b` | 35.86 | 142 / 396 | 879,854 | 0.4640 | 107.7 |

## Cost by arm

| arm | records | tasks | models | total tokens | total cost USD | cost vs all_high | tokens vs all_high |
|---|---|---|---|---|---|---|---|
| `all_high` | 1,980 | 132 | 5 | 11,375,912 | 20.3497 | 100.00 % | 100.00 % |
| `all_low` | 1,980 | 132 | 5 | 8,169,544 | 0.6975 | 3.43 % | 71.81 % |
| `allocated` | 1,980 | 132 | 5 | 11,375,786 | 8.7925 | 43.21 % | 100.00 % |

## Cost by task category

| arm | category | records | tokens | token share | cost USD | cost share |
|---|---|---|---|---|---|---|
| `all_high` | gaia | 480 | 4,131,392 | 36.32 % | 7.3760 | 36.25 % |
| `all_high` | math | 750 | 4,216,704 | 37.07 % | 7.5750 | 37.22 % |
| `all_high` | simpleqa | 750 | 3,027,816 | 26.62 % | 5.3986 | 26.53 % |
| `all_low` | gaia | 480 | 2,049,186 | 25.08 % | 0.1742 | 24.98 % |
| `all_low` | math | 750 | 3,092,524 | 37.85 % | 0.2640 | 37.85 % |
| `all_low` | simpleqa | 750 | 3,027,834 | 37.06 % | 0.2592 | 37.17 % |
| `allocated` | gaia | 480 | 4,131,250 | 36.32 % | 7.3750 | 83.88 % |
| `allocated` | math | 750 | 4,216,678 | 37.07 % | 1.1583 | 13.17 % |
| `allocated` | simpleqa | 750 | 3,027,858 | 26.62 % | 0.2592 | 2.95 % |

## Rank stability vs the gold standard

Stability is computed on the **pooled** leaderboard (all rounds combined), which is far less noisy than any single round.

| arm | stability % | delta vs 100 | top-3 overlap | overlap ratio | Kendall tau-b | same order | top-3 |
|---|---|---|---|---|---|---|---|
| `all_high` | 100.00 | +0.00 | 3/3 | 1.000 | 1.000 | yes | `thorough-32b`, `balanced-13b`, `quickthink-7b` |
| `all_low` | 63.34 | -36.66 | 2/3 | 0.667 | 0.200 | no | `balanced-13b`, `mathwhiz-9b`, `quickthink-7b` |
| `allocated` | 68.33 | -31.66 | 2/3 | 0.667 | 0.400 | no | `balanced-13b`, `mathwhiz-9b`, `thorough-32b` |

### Is the finding stable across rounds?

Each round produces its own leaderboard and its own stability score; a pooled number is only trustworthy if the rounds agree.

| arm | rounds | stability % per round | min | max | spread | same top-3 order every round |
|---|---|---|---|---|---|---|
| `all_high` | 3 | 100.00, 100.00, 100.00 | 100.00 | 100.00 | 0.00 | yes |
| `all_low` | 3 | 63.34, 63.34, 63.34 | 63.34 | 63.34 | 0.00 | no |
| `allocated` | 3 | 73.33, 58.34, 68.33 | 58.34 | 73.33 | 15.00 | no |

## Which model does the cheap arm wrongly crown?

- gold standard (`all_high`) winner: **thorough-32b** at 71.97% pass rate
- cheap arm (`all_low`) winner: **balanced-13b**
- wrongly crowned: **balanced-13b** — true rank under the gold standard: 2 (62.63% pass rate)

Cheap evaluation wrongly crowns balanced-13b first (56.8182% pass rate under all_low), but under the all_high gold standard balanced-13b is #2 of 5 at 62.6263%; the true winner is thorough-32b at 71.9697%. Measuring every task cheaply flips the top of the leaderboard.

## The X% / Y% trade-off

The allocated arm cuts evaluation cost by 56.79% versus the all_high gold standard (20.3497 USD / 11,375,912 tokens -> 8.7925 USD / 11,375,786 tokens) and moves top-3 rank stability by -31.66%.

- cost reduction vs `all_high`: **56.79%** (20.3497 USD -> 8.7925 USD)
- token reduction vs `all_high`: **0.00%** (11,375,912 -> 11,375,786 tokens)
- top-3 rank stability change: **-31.66%** (allocated arm scores 68.33/100)

## Methodology

- **arm_cost**: total_tokens and total_cost_usd are math.fsum over the arm's records, so they are independent of record order
- **cost_reduction_pct**: 100 * (baseline_cost_usd - allocated_cost_usd) / baseline_cost_usd against ARM_ALL_HIGH
- **kendall_tau**: Kendall tau-b = (C - D) / sqrt((n0 - n1) * (n0 - n2)), n0 = n(n-1)/2, n1/n2 = pairs tied on each side; None for <2 common models or a zero denominator
- **leaderboard_order**: pass_rate descending, then total_cost_usd ascending, then model name ascending (deterministic, no exceptions)
- **pass_rate**: 100 * passed / evaluated, rounded to 4 dp before ranking
- **stability_delta_pct**: stability_pct - 100 (reference arm is 0.0 by definition)
- **stability_pct**: 100 * (0.5 * top3_membership_overlap_ratio + 0.5 * (kendall_tau + 1)/2); falls back to 100 * overlap_ratio when kendall_tau is undefined
- **top3_membership_overlap_ratio**: |top3(arm) ∩ top3(reference)| / min(3, len(top3(reference)), len(top3(arm)))

## Appendix — per-round leaderboards

### `all_high` — round 0

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `thorough-32b` | 72.73 | 96 / 132 | 1,760,406 | 3.2916 |
| 2 | `balanced-13b` | 62.88 | 83 / 132 | 647,890 | 1.1297 |
| 3 | `mathwhiz-9b` | 57.58 | 76 / 132 | 730,738 | 1.3120 |
| 4 | `quickthink-7b` | 56.06 | 74 / 132 | 359,678 | 0.5816 |
| 5 | `tinyfast-3b` | 28.79 | 38 / 132 | 293,282 | 0.4685 |

### `all_high` — round 1

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `thorough-32b` | 71.21 | 94 / 132 | 1,760,438 | 3.2918 |
| 2 | `balanced-13b` | 62.12 | 82 / 132 | 647,880 | 1.1296 |
| 3 | `quickthink-7b` | 59.09 | 78 / 132 | 359,684 | 0.5816 |
| 4 | `mathwhiz-9b` | 54.55 | 72 / 132 | 730,744 | 1.3121 |
| 5 | `tinyfast-3b` | 35.61 | 47 / 132 | 293,270 | 0.4684 |

### `all_high` — round 2

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `thorough-32b` | 71.97 | 95 / 132 | 1,760,310 | 3.2908 |
| 2 | `balanced-13b` | 62.88 | 83 / 132 | 647,850 | 1.1294 |
| 3 | `quickthink-7b` | 59.09 | 78 / 132 | 359,668 | 0.5815 |
| 4 | `mathwhiz-9b` | 57.58 | 76 / 132 | 730,786 | 1.3124 |
| 5 | `tinyfast-3b` | 34.85 | 46 / 132 | 293,288 | 0.4686 |

### `all_low` — round 0

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `mathwhiz-9b` | 62.88 | 83 / 132 | 641,446 | 0.0553 |
| 2 | `balanced-13b` | 56.82 | 75 / 132 | 568,638 | 0.0479 |
| 3 | `quickthink-7b` | 51.52 | 68 / 132 | 359,634 | 0.0285 |
| 4 | `thorough-32b` | 49.24 | 65 / 132 | 860,192 | 0.0776 |
| 5 | `tinyfast-3b` | 40.91 | 54 / 132 | 293,292 | 0.0231 |

### `all_low` — round 1

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `balanced-13b` | 62.12 | 82 / 132 | 568,616 | 0.0479 |
| 2 | `mathwhiz-9b` | 53.79 | 71 / 132 | 641,418 | 0.0553 |
| 3 | `quickthink-7b` | 51.52 | 68 / 132 | 359,682 | 0.0286 |
| 4 | `thorough-32b` | 46.21 | 61 / 132 | 860,192 | 0.0776 |
| 5 | `tinyfast-3b` | 38.64 | 51 / 132 | 293,296 | 0.0231 |

### `all_low` — round 2

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `quickthink-7b` | 56.82 | 75 / 132 | 359,638 | 0.0285 |
| 2 | `balanced-13b` | 51.52 | 68 / 132 | 568,572 | 0.0479 |
| 3 | `mathwhiz-9b` | 51.52 | 68 / 132 | 641,462 | 0.0553 |
| 4 | `thorough-32b` | 46.21 | 61 / 132 | 860,180 | 0.0776 |
| 5 | `tinyfast-3b` | 29.55 | 39 / 132 | 293,286 | 0.0231 |

### `allocated` — round 0

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `balanced-13b` | 66.67 | 88 / 132 | 647,842 | 0.4599 |
| 2 | `thorough-32b` | 65.15 | 86 / 132 | 1,760,372 | 1.5917 |
| 3 | `quickthink-7b` | 59.85 | 79 / 132 | 359,702 | 0.1918 |
| 4 | `mathwhiz-9b` | 58.33 | 77 / 132 | 730,718 | 0.5326 |
| 5 | `tinyfast-3b` | 35.61 | 47 / 132 | 293,274 | 0.1546 |

### `allocated` — round 1

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `mathwhiz-9b` | 71.21 | 94 / 132 | 730,778 | 0.5329 |
| 2 | `balanced-13b` | 66.67 | 88 / 132 | 647,902 | 0.4605 |
| 3 | `quickthink-7b` | 58.33 | 77 / 132 | 359,664 | 0.1915 |
| 4 | `thorough-32b` | 58.33 | 77 / 132 | 1,760,318 | 1.5914 |
| 5 | `tinyfast-3b` | 37.88 | 50 / 132 | 293,288 | 0.1547 |

### `allocated` — round 2

| rank | model | pass rate % | passed / evaluated | tokens | cost USD |
|---|---|---|---|---|---|
| 1 | `quickthink-7b` | 64.39 | 85 / 132 | 359,678 | 0.1916 |
| 2 | `thorough-32b` | 62.12 | 82 / 132 | 1,760,328 | 1.5913 |
| 3 | `mathwhiz-9b` | 60.61 | 80 / 132 | 730,776 | 0.5330 |
| 4 | `balanced-13b` | 58.33 | 77 / 132 | 647,854 | 0.4602 |
| 5 | `tinyfast-3b` | 34.09 | 45 / 132 | 293,292 | 0.1547 |
