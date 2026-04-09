"""
game/communication_loop.py — The main game loop.

Runs N rounds of the referential communication game between Agent A and Agent B.
Each round:
  1. Generate a target concept
  2. Agent A encodes it as a symbol string
  3. Agent B decodes the symbol string back to a concept
  4. Evaluate success
  5. Update the shared lexicon
  6. Log everything

"""

import json
import os
import time
from tqdm import tqdm

from agents.agent_a import SpeakerAgent
from agents.agent_b import ListenerAgent
from game.task_generator import generate_concept, evaluate_decode
from game.lexicon import update_lexicon, save_lexicon
from config import LEXICON_DIR


def run_experiment(
    lexicon: dict,
    num_rounds: int,
    num_concepts: int,
    checkpoint_interval: int,
    run_id: str
) -> tuple[dict, list]:
    """
    Run full communication experiment.

    Args:
        lexicon: starting lexicon (empty at beginning of experiment)
        num_rounds: total number of rounds to run
        num_concepts: unused directly here (task generator draws from full space)
                      reserved for future fixed-concept-set mode
        checkpoint_interval: save a lexicon snapshot every N rounds
        run_id: unique identifier for this run (used in filenames)

    Returns:
        (final_lexicon, round_logs) — the evolved lexicon and full log of every round
    """

    # Initialize the two agents
    # Each has its own API client but shares the same lexicon via prompts
    speaker = SpeakerAgent()
    listener = ListenerAgent()

    round_logs = []   # We'll store one dict per round here
    success_count = 0

    print(f"Starting experiment: {num_rounds} rounds\n")

    for round_num in tqdm(range(1, num_rounds + 1), desc="Rounds"):

        # generate target concept
        target_concept = generate_concept()

        # Agent A encodes the concept
        symbol_message, speaker_raw = speaker.encode(target_concept, lexicon)

        # Agent B decodes the symbol string
        decoded_concept, listener_raw = listener.decode(symbol_message, lexicon)

        # evaluate
        result = evaluate_decode(target_concept, decoded_concept)
        if result["success"]:
            success_count += 1

        # update shared lexicon
        # Only updates on success — see lexicon.py for the update strategy
        lexicon = update_lexicon(lexicon, symbol_message, target_concept, result["success"])

        # lLog everything
        # We log raw model output too so you can analyze agent reasoning later
        round_log = {
            "round": round_num,
            "target_concept": target_concept,
            "symbol_message": symbol_message,
            "decoded_concept": decoded_concept,
            "result": result,
            "lexicon_size": len(lexicon),
            "cumulative_accuracy": success_count / round_num,
            # raw outputs for debugging and later analysis
            "speaker_raw_output": speaker_raw,
            "listener_raw_output": listener_raw,
        }
        round_logs.append(round_log)

        # Print a brief summary every round so you can watch it unfold
        status = "correct" if result["success"] else "wrong"
        print(
            f"  Round {round_num:03d} {status} | "
            f"Concept: {target_concept['shape']}/{target_concept['color']}/{target_concept['position']} | "
            f"Message: {symbol_message} | "
            f"Decoded: {decoded_concept} | "
            f"Accuracy: {success_count/round_num:.0%}"
        )

        # checkpoint
        if round_num % checkpoint_interval == 0:
            checkpoint_path = os.path.join(LEXICON_DIR, f"checkpoint_{run_id}_round{round_num:04d}.json")
            save_lexicon(lexicon, checkpoint_path)
            print(f"  [Checkpoint saved: {checkpoint_path}]")

        # Small delay between API calls to avoid rate limiting
        time.sleep(0.5)

    print(f"\nExperiment complete.")
    print(f"Final accuracy: {success_count}/{num_rounds} = {success_count/num_rounds:.1%}")
    print(f"Lexicon size: {len(lexicon)} entries")

    return lexicon, round_logs