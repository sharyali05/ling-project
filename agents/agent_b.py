"""
agents/agent_b.py — Agent B: the Listener.

Agent B receives a symbol string from Agent A and must decode it
back into the original concept. It never sees the target concept directly —
only the symbol string and the shared lexicon.
"""

import json
from agents.base_agent import BaseAgent
from config import SYMBOLS, NUMBERS, SHAPES, COLORS, POSITIONS


class ListenerAgent(BaseAgent):
    """
    Agent B: receives a symbol string, outputs a decoded concept.

    Output must be valid JSON matching the concept schema.
    We enforce this by prompting for JSON-only output and catching
    parse errors in the communication loop.
    """

    def __init__(self):
        super().__init__(name="Agent B (Listener)")

    def decode(self, symbol_message: str, lexicon: dict) -> tuple[dict | None, str]:
        """
        Decode a symbol string into a concept.

        Args:
            symbol_message: the symbol string received from Agent A, e.g. "F1-G3-H2"
            lexicon: shared lexicon containing only outcome markers ($ or %)
                    with no concept descriptions — Agent B must infer meaning
                    purely from patterns across successes and failures

        Returns:
            A tuple of (parsed concept dict or None if parsing failed, raw model output)
        """
        prompt = self._build_prompt(symbol_message, lexicon)
        raw_output = self.call(prompt)
        decoded_concept = self._parse_concept(raw_output)
        return decoded_concept, raw_output

    def _build_prompt(self, symbol_message: str, lexicon: dict) -> str:
        """
        Construct the full prompt for Agent B.

        Key decisions:
        - Agent B sees ONLY outcome markers — no concept descriptions
        - No valid_values injected — Agent B must infer possible values from patterns
        - No dimensional hints — Agent B cannot be told which letters mean shape/color/position
        - JSON output only for reliable parsing
        """
        from game.lexicon import get_agent_b_lexicon_view

        lexicon_section = get_agent_b_lexicon_view(lexicon)

        prompt = f"""You are Agent B, the Listener in a referential communication experiment.

    Agent A has sent you this exact symbol string: {symbol_message}

    YOUR ONLY JOB is to decode what concept this symbol string represents.

    SYMBOL VOCABULARY (the channel Agent A used):
    Available letters: {SYMBOLS}
    Available numbers: {NUMBERS}
    Tokens are letter-number pairs separated by hyphens.

    COMMUNICATION HISTORY ($ = this string led to a correct decode, % = this string led to a wrong decode):
    {lexicon_section}

    CRITICAL INSTRUCTIONS:
    - Study the communication history carefully to infer what the symbols mean
    - A $ entry means that exact symbol string was decoded correctly that round
    - A % entry means that exact symbol string led to a wrong decode that round
    - Look for patterns across multiple $ entries to infer what each token encodes
    - Do NOT default to any assumed values — decode based only on observed patterns
    - Output ONLY a JSON object with keys: "shape", "color", "position"
    - No explanation, no markdown fences, just the JSON
    - Start your response with {{ and end with }}

    Decode {symbol_message} now:"""
        return prompt

    def _parse_concept(self, raw_output: str) -> dict | None:
        """
        Parse the model's JSON output into a concept dict.

        Returns None if parsing fails — the communication loop will
        log this as a failure and flag it for later analysis.
        """
        import re
        json_match = re.search(r'\{[^{}]+\}', raw_output, re.DOTALL)
        if json_match:
            try:
                concept = json.loads(json_match.group())
                if all(k in concept for k in ("shape", "color", "position")):
                    return concept
            except json.JSONDecodeError:
                pass
        return None