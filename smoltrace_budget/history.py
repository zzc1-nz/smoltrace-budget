"""Historical difficulty signals for the budget allocator.

The allocator's ``by_pass_rate`` and ``greedy_budget`` policies need a
difficulty prior *before* the expensive evaluation runs. In production this
comes from previous leaderboard runs: SMOLTRACE already persists a
``smoltrace-results-*`` dataset per run, so the historical pass rate of every
task is a query away.

Here the signal is produced from the model pool's own latent abilities plus a
fixed difficulty offset per task, which is what a real history would look like
after a handful of warm-up runs. Using the latent abilities is deliberate: the
allocator is meant to make good decisions from *cheap, noisy, prior* evidence,
and the honest way to simulate that is to feed it a noisy estimate rather than
the ground truth.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from .contracts import ModelProfile, TaskSpec
from .models import CANDIDATES

__all__ = ["build_history", "write_history", "load_history"]

#: Difficulty offset applied to the pool's mean ability for each category.
#: GAIA is the hardest (multi-hop, tool-dependent), SimpleQA sits in the middle,
#: Math is dominated by the specialists' strength.
CATEGORY_DIFFICULTY: Dict[str, float] = {
    "gaia": -0.16,
    "math": -0.04,
    "simpleqa": -0.06,
    "unknown": -0.10,
}

#: How much the prior is allowed to deviate from the truth. A real history is
#: built from a finite number of runs, so it is noisy — the allocator must be
#: robust to that, and the experiment should not hand it a perfect oracle.
PRIOR_NOISE = 0.06


def _stable_unit(*parts: Any) -> float:
    """Deterministic pseudo-random float in [0, 1) from arbitrary parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_history(
    tasks: List[TaskSpec],
    candidates: List[ModelProfile] | None = None,
    *,
    seed: int = 0,
    noise: float = PRIOR_NOISE,
) -> Dict[str, Dict[str, Any]]:
    """Estimate each task's historical pass rate across the candidate pool.

    Returns a mapping ``{task_id: {"pass_rate": float, "n": int, "category": str}}``
    shaped exactly like what the allocator's ``history`` argument expects.

    The estimate is the pool-mean true ability for the task's category, shifted
    by that category's difficulty offset and jittered by deterministic noise.
    A finite-sample ``n`` is recorded too, because a real history has one.
    """
    pool = candidates if candidates is not None else CANDIDATES

    history: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        abilities = [candidate.pass_rate_for(task.category) for candidate in pool]
        pool_mean = sum(abilities) / len(abilities) if abilities else 0.5
        offset = CATEGORY_DIFFICULTY.get(task.category, CATEGORY_DIFFICULTY["unknown"])
        jitter = (_stable_unit(seed, "history", task.id) - 0.5) * 2.0 * noise
        pass_rate = max(0.01, min(0.99, pool_mean + offset + jitter))
        # A warm-up history is a handful of runs per task, not thousands.
        n = 3 + int(_stable_unit(seed, "n", task.id) * 6)
        history[task.id] = {
            "pass_rate": round(pass_rate, 4),
            "n": n,
            "category": task.category,
        }
    return history


def write_history(path, history: Dict[str, Dict[str, Any]]) -> Any:
    import os

    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load_history(path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
