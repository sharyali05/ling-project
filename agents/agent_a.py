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
    f"Symbols: {SYMBOLS}\n"
    f"Numbers: {NUMBERS}\n"
    f"Format: pair one symbol with one number, e.g. 'F1', 'G3'\n"
    f"Combine up to {MAX_MESSAGE_LENGTH} pairs with hyphens, e.g. 'F1-G3-H2'"
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

    def encode(self, concept: dict, lexicon: dict) -> str:
        """
        Encode a concept into a symbol string.

        Args:
            concept: dict like {"shape": "triangle", "color": "red", "position": "top-left"}
            lexicon: current shared lexicon (symbol → meaning mappings)

        Returns:
            A symbol string like "F1-G3-H2", or raw model output if parsing fails.
        """
        prompt = self._build_prompt(concept, lexicon)
        raw_output = self.call(prompt)

        # extract just the symbol message — the last non-empty line of output.
        # only pass the final symbol string to Agent B.
        symbol_message = self._extract_symbol(raw_output)
        return symbol_message, raw_output  # return both for logging

    def _build_prompt(self, concept: dict, lexicon: dict) -> str:
        """
        Construct the full prompt for Agent A.

        Key prompt engineering decisions made here:
        - Chain-of-thought allowed (helps structure but is in English — only the OUTPUT is symbolic)
        - Lexicon injected as JSON so agent can build on prior conventions
        - Explicit output format instruction on the final line
        """

        lexicon_section = (
            json.dumps(lexicon, indent=2)
            if lexicon
            else "(empty — you are inventing the language from scratch this round)"
        )

        prompt = f"""You are Agent A, the Speaker in a referential communication experiment about emergent language.

                    YOUR ROLE:
                    You will be shown a target concept. Your job is to encode it using ONLY the symbol vocabulary below.
                    Agent B (the Listener) will receive your symbol string and must decode the original concept.
                    Neither of you may use natural language in the communication channel.

                    SYMBOL VOCABULARY:
                    {VOCAB_DESCRIPTION}

                    CURRENT SHARED LEXICON (conventions established in prior rounds):
                    {lexicon_section}

                    INSTRUCTIONS:
                    - You MAY reason about your encoding strategy in plain English first (this helps you be systematic)
                    - Your final output line MUST be ONLY the symbol string — nothing else
                    - Use the existing lexicon where possible to stay consistent
                    - If you need to encode a concept that has no lexicon entry yet, invent a new symbol mapping
                    - Do NOT use more symbols than necessary

                    TARGET CONCEPT TO ENCODE:
                    {json.dumps(concept, indent=2)}

                    Think through your encoding, then output ONLY the symbol string on the very last line.
                    """
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