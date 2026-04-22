"""
game/task_generator.py — Generates referential tasks (target concepts) for each round.

A concept is a combination of shape + color + position.
The generator can produce random concepts or cycle through a fixed set —
both modes are useful for different experimental conditions.
"""

import random
from config import SHAPES, COLORS, POSITIONS


def generate_concept() -> dict:
    """
    Generate one random target concept.

    Returns:
        A concept dict like {"shape": "triangle", "color": "red", "position": "top-left"}
    """
    return {
        "shape": random.choice(SHAPES),
        "color": random.choice(COLORS),
        "position": random.choice(POSITIONS)
    }


def generate_concept_set(n: int, allow_repeats: bool = True) -> list[dict]:
    """
    Generate a set of N concepts for use across rounds.

    Args:
        n: number of concepts to generate
        allow_repeats: if True, concepts may repeat across rounds (default).
                       if False, returns up to N unique concepts from the full space.

    Returns:
        List of concept dicts.
    """
    if allow_repeats:
        return [generate_concept() for _ in range(n)]
    else:
        # Build the full concept space and sample without replacement
        full_space = [
            {"shape": s, "color": c, "position": p}
            for s in SHAPES
            for c in COLORS
            for p in POSITIONS
        ]
        n = min(n, len(full_space))  # can't request more than exist
        return random.sample(full_space, n)


def evaluate_decode(target: dict, decoded: dict | None) -> dict:
    """
    Compare the target concept to Agent B's decoded concept.

    Returns a result dict with:
    - success: True only if all three attributes match exactly
    - attribute_scores: per-attribute match (useful for partial credit analysis later)
    - match_count: how many attributes were correct (0, 1, 2, or 3)

    NOTE: We log partial scores here but the main loop only uses `success`
    for lexicon updates. Partial credit analysis lives in analysis/metrics.py.
    """
    if decoded is None:
        # Agent B failed to produce parseable output
        return {
            "success": False,
            "attribute_scores": {"shape": False, "color": False, "position": False},
            "match_count": 0,
            "parse_error": True
        }

    attribute_scores = {
        attr: (target.get(attr) == decoded.get(attr))
        for attr in ("shape", "color", "position")
    }

    match_count = sum(attribute_scores.values())
    success = match_count == 3  # All attributes must match for full success

    return {
        "success": success,
        "attribute_scores": attribute_scores,
        "match_count": match_count,
        "parse_error": False
    }