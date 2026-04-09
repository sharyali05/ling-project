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
            lexicon: current shared lexicon

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
        - Agent B does NOT see the original concept
        - We tell it the valid output values so it can't hallucinate new ones
        - We ask for JSON output only so we can parse it reliably
        """

        lexicon_section = (
            json.dumps(lexicon, indent=2)
            if lexicon
            else "(empty — no conventions established yet)"
        )

        # tell the agent exactly what values are valid
        valid_values = {
            "shape": SHAPES,
            "color": COLORS,
            "position": POSITIONS
        }

        prompt = f"""You are Agent B, the Listener in a referential communication experiment about emergent language.

                    YOUR ROLE:
                    Agent A has encoded a concept using a shared symbol vocabulary and sent you a symbol string.
                    Your job is to decode that symbol string back into the original concept.
                    You have access to the shared lexicon of conventions built up over prior rounds.

                    CURRENT SHARED LEXICON:
                    {lexicon_section}

                    VALID OUTPUT VALUES:
                    {json.dumps(valid_values, indent=2)}

                    INSTRUCTIONS:
                    - Use the lexicon to interpret the symbol string
                    - If a symbol has no lexicon entry, make your best inference based on patterns you observe
                    - Output ONLY a JSON object — no explanation, no extra text
                    - Your JSON must use exactly these keys: "shape", "color", "position"
                    - Values must come from the valid output values listed above

                    SYMBOL MESSAGE RECEIVED:
                    {symbol_message}

                    Output ONLY the JSON object.
                    """
        return prompt

    def _parse_concept(self, raw_output: str) -> dict | None:
        """
        Parse the model's JSON output into a concept dict.

        Returns None if parsing fails — the communication loop will
        log this as a failure and flag it for later analysis.
        """
        # Strip any accidental markdown code fences the model might add
        cleaned = raw_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            concept = json.loads(cleaned)
            # Validate that required keys are present
            if all(k in concept for k in ("shape", "color", "position")):
                return concept
            else:
                return None
        except json.JSONDecodeError:
            return None