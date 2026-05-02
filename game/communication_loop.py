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
import random

from agents.agent_a import SpeakerAgent
from agents.agent_b import ListenerAgent
from game.task_generator import generate_concept, evaluate_decode, generate_concept_set
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
        num_concepts: size of the fixed concept pool agents draw from
        checkpoint_interval: save a lexicon snapshot every N rounds
        run_id: unique identifier for this run (used in filenames)

    Returns:
        (final_lexicon, round_logs) — the evolved lexicon and full log of every round
    """

    speaker = SpeakerAgent()
    listener = ListenerAgent()

    round_logs = []
    success_count = 0

    # Agent A's private history — maps "$symbol" or "%symbol" to concept string
    # This is NEVER passed to Agent B
    agent_a_history = {}

    print(f"Starting experiment: {num_rounds} rounds\n")

    concept_pool = generate_concept_set(num_concepts, allow_repeats=False)
    print(f"Concept pool ({num_concepts} concepts): {concept_pool}\n")

    for round_num in tqdm(range(1, num_rounds + 1), desc="Rounds"):

        target_concept = random.choice(concept_pool)

        # Agent A encodes — receives its own private history (symbol + concept)
        symbol_message, speaker_raw = speaker.encode(target_concept, agent_a_history)

        # Agent B decodes — receives only outcome markers, no concept descriptions
        decoded_concept, listener_raw = listener.decode(symbol_message, lexicon)

        # evaluate
        result = evaluate_decode(target_concept, decoded_concept)
        if result["success"]:
            success_count += 1

        # Update shared lexicon (outcome markers only — what Agent B can see)
        lexicon = update_lexicon(lexicon, symbol_message, target_concept, result["success"])

        # Update Agent A's private history (full concept descriptions — never shown to B)
        marker = "$" if result["success"] else "%"
        concept_str = f"{target_concept['shape']}, {target_concept['color']}, {target_concept['position']}"
        agent_a_history[f"{marker}{symbol_message}"] = concept_str

        # Log everything
        round_log = {
            "round": round_num,
            "target_concept": target_concept,
            "symbol_message": symbol_message,
            "decoded_concept": decoded_concept,
            "result": result,
            "lexicon_size": len(lexicon),
            "cumulative_accuracy": success_count / round_num,
            "speaker_raw_output": speaker_raw,
            "listener_raw_output": listener_raw,
        }
        round_logs.append(round_log)

        status = "correct" if result["success"] else "wrong"
        print(
            f"  Round {round_num:03d} {status} | "
            f"Concept: {target_concept['shape']}/{target_concept['color']}/{target_concept['position']} | "
            f"Message: {symbol_message} | "
            f"Decoded: {decoded_concept} | "
            f"Accuracy: {success_count/round_num:.0%}"
        )

        if round_num % checkpoint_interval == 0:
            checkpoint_path = os.path.join(LEXICON_DIR, f"checkpoint_{run_id}_round{round_num:04d}.json")
            save_lexicon(lexicon, checkpoint_path)
            print(f"  [Checkpoint saved: {checkpoint_path}]")

        time.sleep(0.5)

    print(f"\nExperiment complete.")
    print(f"Final accuracy: {success_count}/{num_rounds} = {success_count/num_rounds:.1%}")
    print(f"Lexicon size: {len(lexicon)} entries")

    return lexicon, round_logs