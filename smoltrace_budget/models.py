"""The candidate model pool under evaluation.

These are the models the leaderboard ranks. Each carries a *true* latent ability
per task category — the ground truth an ideal, infinitely expensive evaluation
would recover. Every measured arm is an attempt to recover this ranking under a
budget.

The pool is deliberately built to contain the failure mode this project studies:

    a model that looks great under cheap measurement and is merely average under
    honest measurement.

`QuickThink-7B` is that model: it is fast, concise, answers immediately without
consulting tools, and therefore *benefits* from a low-fidelity evaluation. It
has little to gain from extra steps, so truncating at `max_steps=2` costs it
almost nothing.

`Thorough-32B` is its mirror image: it needs tools and multi-step deliberation
to reach its true ability, so a low-fidelity evaluation systematically
under-measures it.

The remaining models sit in between, so the top of the leaderboard is genuinely
contested and a small measurement bias can flip it.
"""

from __future__ import annotations

from typing import Dict, List

from .contracts import ModelProfile

__all__ = ["CANDIDATES", "CANDIDATE_BY_ID", "get_candidate"]


CANDIDATES: List[ModelProfile] = [
    ModelProfile(
        model_id="quickthink-7b",
        display_name="QuickThink-7B",
        api_model_id="openai/quickthink-7b",
        # Genuinely decent on shallow lookups, weaker where multi-hop reasoning
        # is required. Chosen so its measured rank is highly evaluation-sensitive.
        true_pass_rate={"gaia": 0.34, "math": 0.61, "simpleqa": 0.72, "default": 0.55},
        tool_reliance=0.15,
        deliberation=0.10,
        verbosity=0.55,
    ),
    ModelProfile(
        model_id="balanced-13b",
        display_name="Balanced-13B",
        api_model_id="openai/balanced-13b",
        true_pass_rate={"gaia": 0.46, "math": 0.66, "simpleqa": 0.68, "default": 0.60},
        tool_reliance=0.45,
        deliberation=0.45,
        verbosity=0.85,
    ),
    ModelProfile(
        model_id="thorough-32b",
        display_name="Thorough-32B",
        api_model_id="openai/thorough-32b",
        # The strongest model overall, but only if you actually let it think and
        # use tools. Cheap evaluation hides this.
        true_pass_rate={"gaia": 0.62, "math": 0.74, "simpleqa": 0.70, "default": 0.68},
        tool_reliance=0.92,
        deliberation=0.90,
        verbosity=1.25,
    ),
    ModelProfile(
        model_id="mathwhiz-9b",
        display_name="MathWhiz-9B",
        api_model_id="openai/mathwhiz-9b",
        # Specialist: strong on math, weak elsewhere. Good for showing that
        # category-blind evaluation mismeasures specialists.
        true_pass_rate={"gaia": 0.30, "math": 0.81, "simpleqa": 0.55, "default": 0.52},
        tool_reliance=0.30,
        deliberation=0.60,
        verbosity=0.95,
    ),
    ModelProfile(
        model_id="tinyfast-3b",
        display_name="TinyFast-3B",
        api_model_id="openai/tinyfast-3b",
        # Weak everywhere: the control. Should stay near the bottom in every arm.
        true_pass_rate={"gaia": 0.16, "math": 0.35, "simpleqa": 0.44, "default": 0.30},
        tool_reliance=0.10,
        deliberation=0.05,
        verbosity=0.45,
    ),
]

CANDIDATE_BY_ID: Dict[str, ModelProfile] = {c.model_id: c for c in CANDIDATES}


def get_candidate(model_id: str) -> ModelProfile:
    """Look up a candidate by its ``model_id`` or ``api_model_id``."""
    if model_id in CANDIDATE_BY_ID:
        return CANDIDATE_BY_ID[model_id]
    for candidate in CANDIDATES:
        if candidate.api_model_id == model_id or candidate.display_name == model_id:
            return candidate
    raise KeyError(f"unknown candidate model {model_id!r}")
