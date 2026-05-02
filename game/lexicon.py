"""
game/lexicon.py — Shared lexicon state management.

"""

import json
import os
from config import LEXICON_DIR


def initialize_lexicon() -> dict:
    return {}


def update_lexicon(lexicon: dict, symbol_message: str, concept: dict, success: bool) -> dict:
    marker = "$" if success else "%"
    key = f"{marker}{symbol_message}"
    lexicon[key] = marker
    return lexicon


def get_agent_b_lexicon_view(lexicon: dict) -> str:
    """
    Returns only the symbol strings with their outcome markers.
    Agent B sees ONLY whether each string succeeded or failed.
    No concept descriptions, no natural language.
    Example output:
        $K1-L3-M2
        %K3-L3-M4
        $K4-L2-M3
    """
    if not lexicon:
        return "No communication history yet."
    lines = []
    for key in lexicon.keys():
        lines.append(key)
    return "\n".join(lines)


def get_agent_a_lexicon_view(agent_a_history: dict) -> str:
    """
    Returns Agent A's private history: symbol string + outcome + concept.
    Only Agent A sees this. Never passed to Agent B.
    Example output:
        $K1-L3-M2: square, blue, top-right
        %K3-L3-M4: star, blue, bottom-right
        $K4-L2-M3: triangle, yellow, bottom-left
    """
    if not agent_a_history:
        return "No communication history yet."
    lines = []
    for key, concept_str in agent_a_history.items():
        lines.append(f"{key}: {concept_str}")
    return "\n".join(lines)


def save_lexicon(lexicon: dict, path: str) -> None:
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(lexicon, f, indent=2)


def load_lexicon(path: str) -> dict:
    import json
    with open(path) as f:
        return json.load(f)