"""
main.py — Entry point for the Linguagenesis experiment.

Run this file to start the communication game between Agent A and Agent B.
All configuration (rounds, concepts, symbols) lives in config.py.

Usage:
    python main.py --rounds 20 --concepts 3
    python main.py --rounds 100 --concepts 10 --log-interval 10
"""

import argparse
import json
import os
from datetime import datetime

from game.communication_loop import run_experiment
from game.lexicon import initialize_lexicon, save_lexicon
from config import NUM_ROUNDS, CHECKPOINT_INTERVAL


def parse_args():
    """
    Parse command-line arguments so you can configure the experiment
    without editing config.py each time.
    """
    parser = argparse.ArgumentParser(description="Run the Linguagenesis emergent language experiment.")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS,
                        help="Number of communication rounds to run")
    parser.add_argument("--concepts", type=int, default=3,
                        help="Number of distinct concepts in the task space")
    parser.add_argument("--log-interval", type=int, default=CHECKPOINT_INTERVAL,
                        help="Save a lexicon snapshot every N rounds")
    return parser.parse_args()


def setup_output_dirs():
    """
    Make sure all output directories exist before we start writing logs.
    These mirror the folder structure in the README.
    """
    dirs = ["data/logs", "data/lexicons", "data/results"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    setup_output_dirs()

    # Timestamp this run so logs don't overwrite each other
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== Linguagenesis Experiment ===")
    print(f"Run ID     : {run_id}")
    print(f"Rounds     : {args.rounds}")
    print(f"Concepts   : {args.concepts}")
    print(f"Log every  : {args.log_interval} rounds\n")

    # Start with an empty shared lexicon — agents build from scratch
    lexicon = initialize_lexicon()

    # Hand off to the communication loop
    # It returns the final lexicon and a full log of every round
    final_lexicon, round_logs = run_experiment(
        lexicon=lexicon,
        num_rounds=args.rounds,
        num_concepts=args.concepts,
        checkpoint_interval=args.log_interval,
        run_id=run_id
    )

    # Save the final evolved lexicon
    final_lexicon_path = f"data/lexicons/lexicon_final_{run_id}.json"
    save_lexicon(final_lexicon, final_lexicon_path)
    print(f"\nFinal lexicon saved to: {final_lexicon_path}")

    # Save the full round-by-round log
    log_path = f"data/logs/run_{run_id}.json"
    with open(log_path, "w") as f:
        json.dump(round_logs, f, indent=2)
    print(f"Round logs saved to   : {log_path}")

    print("\n=== Experiment Complete ===\n")


if __name__ == "__main__":
    main()