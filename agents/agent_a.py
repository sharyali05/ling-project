"""
agents/agent_a.py — Agent A: the Speaker.

Agent A sees the target concept and must encode it as a symbol string.
It cannot use natural language in its output — only symbols.

"""

import json
from agents.base_agent import BaseAgent
from config import SYMBOLS, NUMBERS, MAX_MESSAGE_LENGTH


# human-readable description of allowed vocabulary so agent knows whats allowed
VOCAB_DESCRIPTION = (
    f"Available symbol letters: {SYMBOLS}\n"
    f"Available numbers: {NUMBERS}\n"
    f"Combine them freely into tokens, separate tokens with hyphens.\n"
    f"You decide how many tokens to use and which symbols mean what."
)


class SpeakerAgent(BaseAgent):
    """
    Agent A: receives a concept, outputs a symbol string.

    The only output we care about is the symbol string on the final line.
    We allow chain-of-thought reasoning before the output — this makes
    behavior more structured and easier to debug — but we parse only
    the last line when extracting the symbol message.
    """

    def __init__(self):
        super().__init__(name="Agent A (Speaker)")

    def encode(self, concept: dict, agent_a_history: dict) -> tuple[str, str]:
        """
        Encode a concept into a symbol string.
        
        Args:
            concept: dict like {"shape": "triangle", "color": "red", "position": "top-left"}
            agent_a_history: Agent A's private history (symbol → concept, never shown to Agent B)
        Returns:
            A tuple of (symbol string, raw model output)
        """
        prompt = self._build_prompt(concept, agent_a_history)
        raw_output = self.call(prompt)
        symbol_message = self._extract_symbol(raw_output)
        return symbol_message, raw_output

    def _build_prompt(self, concept: dict, agent_a_history: dict) -> str:
        """
        Construct the full prompt for Agent A.

        Key prompt engineering decisions:
        - Agent A sees its own private history with full concept descriptions
        - Chain-of-thought reasoning is allowed in English
        - Only the final output line is the symbol string
        - No valid_values injected — Agent A invents the encoding
        """
        from game.lexicon import get_agent_a_lexicon_view

        history_section = get_agent_a_lexicon_view(agent_a_history)

        prompt = f"""You are Agent A, the Speaker in a referential communication experiment about emergent language.

    YOUR ROLE:
    You will be shown a target concept. Your job is to encode it using ONLY the symbol vocabulary below.
    Agent B (the Listener) will receive your symbol string and must decode the original concept.
    Neither of you may use natural language in the communication channel.

    SYMBOL VOCABULARY:
    {VOCAB_DESCRIPTION}

    YOUR COMMUNICATION HISTORY ($ confirmed convention, % failed attempt):
    {history_section}

    Use $ entries as reliable conventions.
    Use % entries to avoid repeating failed encodings.

    INSTRUCTIONS:
    - You MAY reason about your encoding strategy in plain English first (this helps you be systematic)
    - Your final output line MUST be ONLY the symbol string — nothing else
    - Use the existing history where possible to stay consistent
    - If you need to encode a concept that has no history entry yet, invent a new symbol mapping
    - Do NOT use more symbols than necessary

    TARGET CONCEPT TO ENCODE:
    {json.dumps(concept, indent=2)}

    Think through your encoding strategy above. Then on the very last line, output ONLY the symbol string — no explanation, no punctuation, no labels. Just the raw symbol string."""
        return prompt

    def _extract_symbol(self, raw_output: str) -> str:
        """
        Pull the symbol string from the model's output.

        Take the last non-empty line, which should be the symbol string.
        Downstream validation in communication_loop.py will flag if it looks wrong.
        """
        lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
        if not lines:
            return ""
        return lines[-1]

        # If it looks like English, flag it
        if len(last_line) > 20 or " " in last_line:
            # Try to find a valid-looking symbol line anywhere in the output
            for line in reversed(lines):
                if len(line) <= 15 and " " not in line and "-" in line or len(line) <= 3:
                    return line
            return "INVALID"  # Will be caught in logs for analysis
        
        return last_line