# ling-project - Emergent Language Agent Experiment 🧬

> A multi-agent AI system where two LLM agents invent a shared language from scratch through a referential communication game — and we analyze whether the emergent system mirrors properties of natural human language.

---

## Academic Context

Lit Review + Project Inspo:
- **Simon Kirby's Iterated Learning Model** — language structure emerges from repeated transmission
- **Nicaraguan Sign Language** — spontaneous language emergence in deaf communities
- **Lewis Signaling Games** — agents develop shared conventions through repeated interaction
- **Compositionality in neural agents** — recent NLP research on emergent communication (Lazaridou et al., 2018)

### Our Core Research Question:
> *When two LLM agents are forced to communicate through a constrained symbolic channel, does the emergent language exhibit structural properties of human language — compositionality, arbitrariness, economy — or does it collapse into something alien?*

---

## File Tree

```
linguagenesis/
│
├── README.md
│
├── agents/
│   ├── agent_a.py              # Agent A — "Speaker" logic & prompt templates
│   ├── agent_b.py              # Agent B — "Listener" logic & prompt templates
│   └── base_agent.py           # Shared agent class (API calls, lexicon management)
│
├── game/
│   ├── communication_loop.py   # Main game loop — runs N rounds between agents
│   ├── task_generator.py       # Generates referential tasks (shapes, colors, positions)
│   └── lexicon.py              # Shared lexicon state (JSON-based, updated each round)
│
├── analysis/
│   ├── metrics.py              # Topographic similarity, entropy, MI calculations
│   ├── visualize.py            # Plots: lexicon graph, entropy over time, symbol heatmaps
│   └── stress_test.py          # Introduces Agent C to test if emergent language is learnable
│
├── data/
│   ├── logs/                   # Raw round-by-round communication logs (auto-generated)
│   ├── lexicons/               # Snapshots of lexicon state at each checkpoint
│   └── results/                # Final analysis outputs and plots
│
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook for exploratory analysis & writeup
│
├── config.py                   # API keys, model settings, hyperparameters
├── main.py                     # Entry point — run the full experiment
├── requirements.txt
└── .env                        # API keys (never commit this)
```

---

## Setup

### Prerequisites
- Python 3.10+
- An Anthropic API key ([get one here](https://console.anthropic.com/))

### Installation

```bash
git clone https://github.com/your-username/linguagenesis.git
cd linguagenesis

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

---

## Dependencies (`requirements.txt`)

```
anthropic>=0.25.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
networkx>=3.2.0
python-dotenv>=1.0.0
tqdm>=4.66.0
jupyter>=1.0.0
```

---

## How It Works

### 1. The Communication Task

Each round, Agent A is shown a **target concept** — a combination of attributes like shape, color, and position (e.g. `{shape: triangle, color: red, position: top-left}`). Agent A must encode this concept using only **tokens from a constrained symbol vocabulary** (e.g. sequences like `X3-Y1-Z2`). Agent B receives the token sequence and must decode the original concept. Neither agent may use natural language in the communication channel.

```
Round N:
  Task Generator  →  target concept  →  Agent A
  Agent A         →  symbol string   →  Agent B
  Agent B         →  decoded concept →  Evaluator
  Evaluator       →  success/fail + feedback  →  both agents
  Both agents     →  update lexicon JSON
```

### 2. Symbol Vocabulary Constraint

This is the most important design decision. Agents communicate **only** using tokens from a fixed set, e.g.:

```
Symbols: [A, B, C, D, E, F, G, H] + [1, 2, 3, 4]
Messages: up to 4 tokens long, e.g. "A2-C4-B1"
```

This constraint forces true invention — agents cannot fall back on English.

### 3. Lexicon State

Each agent maintains a running `lexicon.json` that maps symbols to meanings. This is passed in the system prompt each round so agents can build on prior conventions:

```json
{
  "A1": "triangle",
  "B3": "red",
  "C2": "top-left",
  "A1-B3": "red triangle",
  "A1-B3-C2": "red triangle in top-left"
}
```

### 4. Analysis

After N rounds, the analysis pipeline measures:

| Metric | What it tells us |
|---|---|
| **Communication accuracy** | Did the language actually work? |
| **Topographic similarity (TopSim)** | Are similar meanings encoded with similar symbols? (compositionality) |
| **Entropy over time** | Is the symbol usage becoming more structured/predictable? |
| **Lexicon growth rate** | How fast did the vocabulary stabilize? |
| **Redundancy** | Are agents using the minimal symbols needed? (economy) |

---

## Running the Experiment

### Quick start (20 rounds, 3 concepts)
```bash
python main.py --rounds 20 --concepts 3
```

### Full experiment (100 rounds, 10 concepts)
```bash
python main.py --rounds 100 --concepts 10 --log-interval 10
```

### Stress test (introduce Agent C)
```bash
python analysis/stress_test.py --lexicon data/lexicons/checkpoint_100.json
```

### Run analysis & generate plots
```bash
python analysis/metrics.py --log-dir data/logs/
python analysis/visualize.py --results-dir data/results/
```

### Launch Jupyter notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Expected Outputs

After running the experiment, `data/results/` will contain:

- `accuracy_over_time.png` — Communication success rate per round
- `entropy_curve.png` — Symbol entropy over rounds (should decrease as structure emerges)
- `lexicon_graph.png` — Network graph of symbol → meaning mappings
- `topsim_score.txt` — Final topographic similarity score
- `lexicon_final.json` — The fully evolved invented language

---

## Experiment Configurations

You can modify `config.py` to run different experimental conditions:

```python
# config.py

MODEL = "claude-sonnet-4-20250514"

# Symbol vocabulary
SYMBOLS = ["A", "B", "C", "D", "E", "F"]
NUMBERS = ["1", "2", "3", "4"]
MAX_MESSAGE_LENGTH = 4       # Max tokens per message

# Task space
SHAPES    = ["circle", "triangle", "square", "star"]
COLORS    = ["red", "blue", "green", "yellow"]
POSITIONS = ["top-left", "top-right", "bottom-left", "bottom-right"]

# Experiment
NUM_ROUNDS = 100
CHECKPOINT_INTERVAL = 10     # Save lexicon snapshot every N rounds
```

---

## Linguistic Analysis Guide

When writing up your results, evaluate the emergent language against these criteria:

### Greenberg's Universals (partial checklist)
- [ ] Does the language show **consistent word order**?
- [ ] Are there **compositional rules** (combining smaller symbols for complex meanings)?
- [ ] Does the language show **economy** (shorter symbols for more frequent concepts)?

### Hockett's Design Features
- [ ] **Arbitrariness** — is the symbol-meaning mapping non-iconic?
- [ ] **Productivity** — can novel concepts be expressed from existing symbols?
- [ ] **Displacement** — can agents refer to concepts not present in the current task?

### Comparison to Human Language
- Does the emergent language show **more or less** compositionality than expected?
- Does constraining the channel (symbols only) push the agents toward **human-like efficiency**?
- What happens when Agent C tries to learn the language — does it generalize?

---

## Our Responsibilities

| Task | Owner |
|---|---|
| Agent architecture & prompt engineering | Person A |
| Communication game loop | Person A |
| Lexicon state management | Person A |
| Metrics & analysis pipeline | Person B |
| Visualization | Person B |
| Stress test (Agent C) | Person B |
| Jupyter notebook & writeup | Both |

---

## References so far:

- Lazaridou, A., Baroni, M. (2020). *Emergent Multi-Agent Communication in the Deep Learning Era.* arXiv.
- Kirby, S., Hurford, J. (2002). *The Emergence of Linguistic Structure.* Edinburgh.
- Lewis, D. (1969). *Convention.* Harvard University Press.
- Mordatch, I., Abbeel, P. (2018). *Emergence of Grounded Compositional Language in Multi-Agent Populations.* AAAI.

---

## 📝 License

MIT License — free to use and modify for academic purposes.
