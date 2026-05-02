"""
analysis/stress_test.py — Agent C learnability stress test.

Introduces a naive third agent (Agent C) into the communication game.
Agent C has NO access to the accumulated lexicon built by Agents A and B.
It must learn to decode Agent A's symbol strings through interaction alone,
receiving only binary right/wrong feedback after each round.

This tests whether the emergent language is genuinely compositional and
learnable --- mirroring Kirby's iterated learning learnability bottleneck.

If Agent C can generalize beyond the specific symbol strings it has seen,
the language is rule-governed. If it can only get exact matches right,
the language is a holistic lookup table.

Usage:
    # Run against a specific final lexicon (Agent A uses that lexicon)
    python game/stress_test.py --lexicon data/lexicons/lexicon_final_RUNID.json

    # Run N rounds of stress test
    python game/stress_test.py --lexicon data/lexicons/lexicon_final_RUNID.json --rounds 50

    # Save results to a specific output file
    python game/stress_test.py --lexicon data/lexicons/lexicon_final_RUNID.json --out data/results/stress_test_RUNID.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_a import SpeakerAgent
from agents.base_agent import BaseAgent
from game.task_generator import generate_concept, evaluate_decode
from game.lexicon import load_lexicon
from config import SHAPES, COLORS, POSITIONS


# -----------------------------------------------------------------------
# Agent C: the naive learner
# -----------------------------------------------------------------------

class NaiveLearnerAgent(BaseAgent):
    """
    Agent C: a naive listener with no lexicon access.

    Differences from Agent B:
    - Receives NO shared lexicon — must infer structure from the symbol
      string alone, plus feedback accumulated across its own rounds.
    - Maintains its own internal observation log: a list of
      (symbol_string, feedback) pairs from prior rounds that it uses
      to build up its own understanding of the conventions.
    - Receives binary right/wrong feedback only — no partial credit,
      no attribute-level breakdown.

    This mirrors the situation of a new language learner entering a
    community with an established language: the conventions exist, but
    the learner must infer them from exposure and feedback rather than
    being handed a dictionary.
    """

    def __init__(self):
        super().__init__(name="Agent C (Naive Learner)")
        # Agent C's own accumulated observations across rounds
        # Format: list of {"symbol": str, "decoded": dict, "correct": bool, "target": dict}
        # Target is only added when correct=True (we don't reveal the answer on failure)
        self.observations = []

    def decode(self, symbol_message: str) -> tuple[dict | None, str]:
        """
        Attempt to decode a symbol string with no lexicon.

        Args:
            symbol_message: the symbol string from Agent A

        Returns:
            (parsed concept dict or None, raw model output)
        """
        prompt = self._build_prompt(symbol_message)
        raw_output = self.call(prompt)
        decoded = self._parse_concept(raw_output)
        return decoded, raw_output

    def record_feedback(self, symbol_message: str, decoded: dict | None,
                        correct: bool, target: dict | None):
        """
        Store the outcome of a round in Agent C's observation log.

        We only reveal the target concept when the agent was correct ---
        on failures, Agent C only learns that its guess was wrong,
        not what the right answer was. This is the binary feedback condition.

        Args:
            symbol_message: the symbol string Agent A produced
            decoded: what Agent C decoded it as
            correct: whether the decode was correct
            target: the actual target concept (only stored if correct=True)
        """
        entry = {
            "symbol": symbol_message,
            "decoded": decoded,
            "correct": correct,
            # confirmed_meaning removed — revealing concept labels to Agent C
            # would leak natural language into the symbol channel
        }
        self.observations.append(entry)

    def _build_prompt(self, symbol_message: str) -> str:
        """
        Build Agent C's prompt.

        Key design decisions:
        - No lexicon is provided — Agent C must infer structure from its
          own observation history and the symbol string structure itself.
        - Prior observations are injected so Agent C can learn across rounds.
        - The symbol vocabulary structure (letter + number format) is
          described so Agent C understands the channel, but NO dimensional
          hints (F=shape etc.) are given --- that would defeat the test.
        - Binary feedback only: Agent C knows which prior guesses were
          right or wrong, and what the correct meaning was when it guessed right.
        """

        valid_values = {
            "shape": SHAPES,
            "color": COLORS,
            "position": POSITIONS
        }

        # Format prior observations for the prompt
        # Only show the last 20 to keep the prompt from growing indefinitely
        recent_obs = self.observations[-20:] if len(self.observations) > 20 else self.observations

        if recent_obs:
            obs_lines = []
            for obs in recent_obs:
                marker = "$" if obs["correct"] else "%"
                line = f"{marker}{obs['symbol']}"
                obs_lines.append(line)
            observations_section = "\n".join(obs_lines)
        else:
            observations_section = "(none yet — this is your first round)"

        prompt = f"""You are Agent C, a new participant in a referential communication experiment.

YOUR SITUATION:
Two other agents (Agent A and Agent B) have been playing a communication game and have
developed a shared symbolic language. You are now joining as a new listener.
You have NO access to their shared lexicon or any record of their prior conversations.
You must figure out how their symbol language works purely from the symbol strings
you receive and the right/wrong feedback you get after each guess.

HOW THE SYMBOL LANGUAGE WORKS (channel rules only):
- Symbols are pairs of a letter and a number, e.g. F1, G3, H2
- Multiple pairs are joined by hyphens, e.g. F1-G3-H2
- Each concept has three attributes: shape, color, and position
- You must figure out which letters and numbers map to which attribute values

YOUR PRIOR OBSERVATIONS:
Each line shows a symbol string and whether your decode was correct ($) or wrong (%).
{observations_section}

INSTRUCTIONS:
- Study your prior observations carefully
- Look for patterns across $ entries — tokens that appear consistently in successful rounds
  likely encode the same attribute value each time
- % entries tell you something went wrong — avoid repeating those interpretations
- Do NOT assume any default values — infer everything from observed patterns
- Output ONLY a JSON object with keys: "shape", "color", "position"
- No explanation, no extra text — just the JSON

SYMBOL STRING TO DECODE NOW:
{symbol_message}

CRITICAL: Your response must be ONLY the JSON object.
No reasoning, no code fences, no explanation.
Start your response with {{ and end with }}
"""
        return prompt

    def _parse_concept(self, raw_output: str) -> dict | None:
        """Parse JSON output, same logic as Agent B."""
        # Find JSON object anywhere in the output
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


# -----------------------------------------------------------------------
# Stress test loop
# -----------------------------------------------------------------------

def run_stress_test(lexicon: dict, num_rounds: int) -> list[dict]:
    """
    Run the Agent C stress test.

    Agent A encodes concepts using the fully established lexicon.
    Agent C attempts to decode them with no lexicon access,
    learning only from binary feedback.

    Args:
        lexicon: the final established lexicon from Agents A and B's experiment
        num_rounds: number of stress test rounds to run

    Returns:
        List of round log dicts
    """
    speaker = SpeakerAgent()
    learner = NaiveLearnerAgent()

    round_logs = []
    success_count = 0

    print(f"\nStarting Agent C stress test: {num_rounds} rounds")
    print(f"Lexicon size Agent A is using: {len(lexicon)} entries\n")

    for round_num in tqdm(range(1, num_rounds + 1), desc="Agent C Rounds"):

        # Step 1: generate a concept
        target_concept = generate_concept()

        # Step 2: Agent A encodes it using the established lexicon
        # Agent A has the full lexicon --- it communicates as normal
        symbol_message, speaker_raw = speaker.encode(target_concept, lexicon)

        # Step 3: Agent C attempts to decode with NO lexicon
        decoded_concept, learner_raw = learner.decode(symbol_message)

        # Step 4: evaluate
        result = evaluate_decode(target_concept, decoded_concept)
        if result["success"]:
            success_count += 1

        # Step 5: give Agent C binary feedback and record observation
        # We reveal the target only on success --- binary feedback condition
        learner.record_feedback(
            symbol_message=symbol_message,
            decoded=decoded_concept,
            correct=result["success"],
            target=target_concept if result["success"] else None
        )

        # Step 6: log the round
        round_log = {
            "round": round_num,
            "target_concept": target_concept,
            "symbol_message": symbol_message,
            "agent_c_decoded": decoded_concept,
            "result": result,
            "cumulative_accuracy": success_count / round_num,
            "agent_c_observation_count": len(learner.observations),
            "speaker_raw_output": speaker_raw,
            "learner_raw_output": learner_raw,
        }
        round_logs.append(round_log)

        # Print round summary
        status = "✓" if result["success"] else "✗"
        print(
            f"  Round {round_num:03d} {status} | "
            f"Concept: {target_concept['shape']}/{target_concept['color']}/{target_concept['position']} | "
            f"Message: {symbol_message} | "
            f"C decoded: {decoded_concept} | "
            f"Accuracy: {success_count/round_num:.0%}"
        )

        time.sleep(0.5)

    print(f"\nStress test complete.")
    print(f"Agent C final accuracy: {success_count}/{num_rounds} = {success_count/num_rounds:.1%}")
    print(f"(For reference: Agents A+B reached ~90%+ over 100 rounds with full lexicon access)")

    return round_logs


# -----------------------------------------------------------------------
# Per-attribute breakdown for stress test results
# -----------------------------------------------------------------------

def stress_test_attribute_accuracy(round_logs: list) -> dict:
    """
    Compute per-attribute cumulative accuracy for Agent C.
    Useful for checking whether position > color > shape ordering holds.
    """
    attrs = ["shape", "color", "position"]
    hits = {a: 0 for a in attrs}
    result = {a: [] for a in attrs}

    for i, r in enumerate(round_logs):
        scores = r["result"].get("attribute_scores", {})
        for a in attrs:
            if scores.get(a, False):
                hits[a] += 1
            result[a].append(hits[a] / (i + 1))

    return result


def print_stress_test_summary(round_logs: list):
    """Print a readable summary of Agent C's performance."""
    total = len(round_logs)
    successes = sum(1 for r in round_logs if r["result"]["success"])
    attr_acc = stress_test_attribute_accuracy(round_logs)

    # Find the round at which Agent C first hit 50% rolling accuracy (window=10)
    first_50_round = None
    for i in range(10, total):
        window = round_logs[max(0, i-10):i]
        roll = sum(1 for r in window if r["result"]["success"]) / len(window)
        if roll >= 0.5 and first_50_round is None:
            first_50_round = round_logs[i]["round"]

    print("\n" + "=" * 55)
    print("  AGENT C STRESS TEST SUMMARY")
    print("=" * 55)
    print(f"  Total rounds           : {total}")
    print(f"  Final accuracy         : {successes/total:.1%}")
    print(f"  First round ≥50% roll  : {first_50_round or 'never reached'}")
    print(f"  Final shape accuracy   : {attr_acc['shape'][-1]:.1%}")
    print(f"  Final color accuracy   : {attr_acc['color'][-1]:.1%}")
    print(f"  Final position accuracy: {attr_acc['position'][-1]:.1%}")
    print("=" * 55 + "\n")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Agent C stress test on an established emergent language."
    )
    parser.add_argument(
        "--lexicon",
        required=True,
        help="Path to the final lexicon JSON file from a completed experiment run"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of stress test rounds (default: 50)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for stress test log JSON (default: data/results/stress_test_TIMESTAMP.json)"
    )
    args = parser.parse_args()

    # Load the established lexicon
    if not os.path.exists(args.lexicon):
        print(f"Error: lexicon file not found: {args.lexicon}")
        sys.exit(1)

    print(f"\nLoading lexicon from: {args.lexicon}")
    lexicon = load_lexicon(args.lexicon)
    print(f"Lexicon loaded: {len(lexicon)} entries")

    # Run the stress test
    round_logs = run_stress_test(lexicon, args.rounds)

    # Print summary
    print_stress_test_summary(round_logs)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"data/results/stress_test_{timestamp}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(round_logs, f, indent=2)
    print(f"Stress test log saved to: {out_path}\n")


if __name__ == "__main__":
    main()
