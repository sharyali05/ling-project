"""
config.py — Central configuration for the Linguagenesis experiment.

Adjust these values to change experimental conditions without touching
any other files. All other modules import from here.
"""

import os
from dotenv import load_dotenv

# Load API key from .env file in the project root.
# .env file should contain one line: ANTHROPIC_API_KEY=your_key_here
# NEVER hardcode your key here — .env is in .gitignore
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not found. "
        "Make sure you have a .env file in the project root with: ANTHROPIC_API_KEY=your_key_here"
    )

# Model 
# use the same model for both agents so behavior is comparable
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024  # 200 was too low

# Symbol Vocabulary
# These are the ONLY characters agents may use in their messages.
# Deliberately chosen to be visually non-iconic (avoid A=triangle, O=circle, etc.)
SYMBOLS = ["F", "G", "H", "J", "K", "L", "M", "N"]
NUMBERS = ["1", "2", "3", "4"]
MAX_MESSAGE_LENGTH = 4  # Max number of symbol-number pairs per message

# Task Space
# The full universe of concepts agents will need to communicate about.
# A concept is one item from each category, e.g. {shape: circle, color: red, position: top-left}
SHAPES    = ["circle", "triangle", "square", "star"]
COLORS    = ["red", "blue", "green", "yellow"]
POSITIONS = ["top-left", "top-right", "bottom-left", "bottom-right"]

# Experiment Parameters
NUM_ROUNDS = 100           # Total communication rounds
CHECKPOINT_INTERVAL = 10   # Save a lexicon snapshot every N rounds

# Output Paths
LOG_DIR     = "data/logs"
LEXICON_DIR = "data/lexicons"
RESULTS_DIR = "data/results"