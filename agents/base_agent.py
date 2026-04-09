"""
agents/base_agent.py — Shared base class for all agents (A, B, C).

Handles the actual Anthropic API call. Both the Speaker and Listener
inherit from this so API logic lives in one place.
"""

import anthropic
from config import ANTHROPIC_API_KEY, MODEL, MAX_TOKENS


class BaseAgent:
    """
    Wraps a single Anthropic API call.

    Every call is stateless — the agent has no memory between rounds.
    Memory is handled externally by passing the lexicon into each prompt.
    This is intentional: it mirrors the iterated learning setup where
    shared context (the lexicon) is the only continuity between rounds.
    """

    def __init__(self, name: str):
        self.name = name
        # One shared client instance per agent — reused across all rounds
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def call(self, prompt: str) -> str:
        """
        Send a prompt to the model and return the raw text response.

        Args:
            prompt: The full prompt string (includes role, rules, lexicon, task)

        Returns:
            The model's response as a plain string, stripped of whitespace.
        """
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        raw_text = response.content[0].text.strip()
        return raw_text