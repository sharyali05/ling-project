"""
game/lexicon.py — Shared lexicon state management.

The lexicon is the agents' only memory between rounds.
It maps symbol strings to meanings and is passed into every prompt.

Design decision: one shared JSON file (not two separate agent lexicons).
This is an idealization — in reality agents may diverge in their internal
models — but it's the right starting point. You can revisit this later
when analyzing whether agent behavior actually matches the lexicon.
"""

import json
import os
from config import LEXICON_DIR


def initialize_lexicon() -> dict:
    """
    Seed the lexicon with a minimal structural convention —
    which symbol letter maps to which attribute dimension.
    This breaks the cold-start deadlock without giving away meanings.
    """
    return {
        #"_convention": "F=shape, G=color, H=position. Numbers indicate specific values within each dimension."
    }


def update_lexicon(lexicon: dict, symbol_message: str, concept: dict, success: bool) -> dict:
    """
    Update the shared lexicon after each round.

    Current strategy: only add entries on successful communication.
    If Agent B decoded correctly, we record the symbol → meaning mapping.
    Failed rounds don't update the lexicon — we don't want to reinforce
    a broken convention.

    NOTE: This is the simplest possible update strategy. More sophisticated
    approaches (partial credit updates, confidence weighting, conflict detection)
    belong in analysis/metrics.py once you have data to work with.

    Args:
        lexicon: current lexicon dict
        symbol_message: the symbol string Agent A produced, e.g. "F1-G3"
        concept: the target concept dict
        success: whether Agent B decoded correctly

    Returns:
        Updated lexicon dict.
    """
    if not success:
        # Don't update on failure — don't reinforce broken conventions
        return lexicon

    # Map the full symbol string to the full concept description
    # Also try to map individual tokens to individual attributes
    # e.g. if "F1-G3-H2" correctly encoded {shape:triangle, color:red, position:top-left}
    # we can tentatively record each token too (useful for TopSim analysis later)

    # Full message → full concept
    concept_str = f"{concept['shape']}, {concept['color']}, {concept['position']}"
    lexicon[symbol_message] = concept_str

    # Individual token → attribute value (speculative — may conflict across rounds)
    # We prefix these with "?" to flag them as inferred, not confirmed
    tokens = symbol_message.split("-")
    attribute_values = list(concept.values())  # [shape_val, color_val, position_val]

    # Only attempt 1:1 token→attribute mapping if message length matches attribute count
    # if len(tokens) == len(attribute_values):
    #     for token, value in zip(tokens, attribute_values):
    #         inferred_key = f"?{token}"
    #         # Only record if we haven't seen a conflict yet
    #         if inferred_key not in lexicon or lexicon[inferred_key] == value:
    #             lexicon[inferred_key] = value

    return lexicon


def save_lexicon(lexicon: dict, path: str) -> None:
    """
    Write the lexicon to a JSON file.

    Args:
        lexicon: the lexicon dict to save
        path: file path to write to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(lexicon, f, indent=2)


def load_lexicon(path: str) -> dict:
    """
    Load a lexicon from a JSON file (used for stress testing with Agent C).

    Args:
        path: file path to read from

    Returns:
        Lexicon dict.
    """
    with open(path, "r") as f:
        return json.load(f)