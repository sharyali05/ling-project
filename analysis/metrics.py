"""
analysis/metrics.py — Core quantitative metrics for the Linguagenesis experiment.

Computes:
  - Communication accuracy over time
  - Lexicon growth rate
  - Symbol entropy over time
  - Topographic similarity (TopSim) — compositionality measure
  - Per-attribute accuracy (shape / color / position)

Usage:
    python analysis/metrics.py --log-dir data/logs/
    python analysis/metrics.py --log-dir data/logs/ --out data/results/
"""

import argparse
import json
import math
import os
from collections import Counter
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_logs(log_dir: str) -> dict[str, list]:
    """Load all JSON run logs from a directory. Returns {filename: rounds_list}."""
    logs = {}
    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith(".json"):
            path = os.path.join(log_dir, fname)
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0 and "round" in data[0]:
                logs[fname] = data
                print(f"  Loaded {fname}: {len(data)} rounds")
    return logs

def normalize_stress_test_log(rounds: list) -> list:
    """
    Rename agent_c_decoded to decoded_concept so stress test
    logs can be passed through the standard metrics pipeline.
    """
    normalized = []
    for r in rounds:
        r_copy = r.copy()
        if "agent_c_decoded" in r_copy:
            r_copy["decoded_concept"] = r_copy.pop("agent_c_decoded")
        normalized.append(r_copy)
    return normalized



# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def accuracy_over_time(rounds: list) -> list[float]:
    """Return per-round binary success (1.0 or 0.0)."""
    return [1.0 if r["result"]["success"] else 0.0 for r in rounds]


def cumulative_accuracy(rounds: list) -> list[float]:
    """Return cumulative accuracy at each round."""
    hits = 0
    result = []
    for i, r in enumerate(rounds):
        if r["result"]["success"]:
            hits += 1
        result.append(hits / (i + 1))
    return result


def rolling_accuracy(rounds: list, window: int = 10) -> list[float]:
    """Return rolling window accuracy."""
    raw = accuracy_over_time(rounds)
    result = []
    for i in range(len(raw)):
        start = max(0, i - window + 1)
        result.append(sum(raw[start:i+1]) / (i - start + 1))
    return result


def per_attribute_accuracy(rounds: list) -> dict[str, list[float]]:
    """Cumulative accuracy broken down by attribute (shape, color, position)."""
    attrs = ["shape", "color", "position"]
    hits = {a: 0 for a in attrs}
    result = {a: [] for a in attrs}
    for i, r in enumerate(rounds):
        scores = r["result"].get("attribute_scores", {})
        for a in attrs:
            if scores.get(a, False):
                hits[a] += 1
            result[a].append(hits[a] / (i + 1))
    return result


# ---------------------------------------------------------------------------
# Lexicon metrics
# ---------------------------------------------------------------------------

def lexicon_growth(rounds: list) -> list[int]:
    """Lexicon size at each round."""
    return [r["lexicon_size"] for r in rounds]


def lexicon_stabilization_round(rounds: list, window: int = 10) -> int:
    """
    Round at which lexicon growth rate stabilizes (< 1 new entry per window rounds).
    Returns -1 if it never stabilizes.
    """
    sizes = lexicon_growth(rounds)
    for i in range(window, len(sizes)):
        growth = sizes[i] - sizes[i - window]
        if growth <= 1:
            return rounds[i]["round"]
    return -1


# ---------------------------------------------------------------------------
# Symbol entropy
# ---------------------------------------------------------------------------

def symbol_entropy_over_time(rounds: list, window: int = 20) -> list[float]:
    """
    Shannon entropy of symbol token distribution in a rolling window.
    Lower entropy = more structured, predictable symbol usage.
    """
    entropies = []
    for i in range(len(rounds)):
        start = max(0, i - window + 1)
        slice_ = rounds[start:i+1]
        tokens = []
        for r in slice_:
            msg = r.get("symbol_message", "")
            tokens.extend(msg.split("-"))
        counter = Counter(tokens)
        total = sum(counter.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in counter.values())
        entropies.append(entropy)
    return entropies


def message_length_over_time(rounds: list) -> list[int]:
    """Number of tokens in each symbol message."""
    return [len(r.get("symbol_message", "").split("-")) for r in rounds]


# ---------------------------------------------------------------------------
# Topographic Similarity (TopSim)
# ---------------------------------------------------------------------------

def _concept_distance(c1: dict, c2: dict) -> float:
    """
    Hamming distance between two concepts (0–3).
    Each attribute that differs contributes 1.
    """
    attrs = ["shape", "color", "position"]
    return sum(1 for a in attrs if c1.get(a) != c2.get(a))


def _symbol_distance(s1: str, s2: str) -> float:
    """
    Edit distance between two symbol strings at the token level.
    Tokens are split by '-'. Uses simple token-level Levenshtein.
    """
    t1 = s1.split("-")
    t2 = s2.split("-")
    # Build DP table
    m, n = len(t1), len(t2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if t1[i-1] == t2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return float(dp[m][n])


def topographic_similarity(rounds: list, min_rounds: int = 20) -> float:
    """
    Compute TopSim: Spearman correlation between pairwise meaning distances
    and pairwise symbol distances, over all successful rounds.

    TopSim close to 1.0 = highly compositional (similar meanings → similar symbols)
    TopSim close to 0.0 = arbitrary / holistic

    Only uses successful rounds to avoid noise from failed communications.
    """
    successful = [r for r in rounds if r["result"]["success"]]
    if len(successful) < min_rounds:
        print(f"  Warning: only {len(successful)} successful rounds for TopSim (need {min_rounds}+)")
        return float("nan")

    # Deduplicate by symbol message (keep last occurrence — most stable mapping)
    seen = {}
    for r in successful:
        seen[r["symbol_message"]] = r["target_concept"]
    pairs_data = list(seen.items())  # [(symbol, concept), ...]

    if len(pairs_data) < 2:
        return float("nan")

    meaning_dists = []
    symbol_dists = []
    for (s1, c1), (s2, c2) in combinations(pairs_data, 2):
        meaning_dists.append(_concept_distance(c1, c2))
        symbol_dists.append(_symbol_distance(s1, s2))

    corr, pval = spearmanr(meaning_dists, symbol_dists)
    return float(corr)


def topsim_over_time(rounds: list, window: int = 30, step: int = 5) -> list[tuple[int, float]]:
    """
    TopSim computed in a rolling window to show how compositionality evolves.
    Returns list of (round_number, topsim) tuples.
    """
    results = []
    for i in range(window, len(rounds) + 1, step):
        slice_ = rounds[max(0, i - window):i]
        ts = topographic_similarity(slice_, min_rounds=10)
        results.append((rounds[i-1]["round"], ts))
    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def compute_all_metrics(rounds: list, run_name: str = "") -> dict:
    """Compute and return all metrics as a dict."""
    final_acc = cumulative_accuracy(rounds)[-1]
    unique_concepts = len({
        f"{r['target_concept']['shape']}-{r['target_concept']['color']}-{r['target_concept']['position']}"
        for r in rounds
    })
    stab_round = lexicon_stabilization_round(rounds)
    final_lexicon_size = lexicon_growth(rounds)[-1]
    ts = topographic_similarity(rounds)
    avg_msg_len = sum(message_length_over_time(rounds)) / len(rounds)

    report = {
        "run": run_name,
        "total_rounds": len(rounds),
        "final_cumulative_accuracy": round(final_acc, 4),
        "unique_concepts_seen": unique_concepts,
        "final_lexicon_size": final_lexicon_size,
        "lexicon_stabilization_round": stab_round,
        "topographic_similarity": round(ts, 4) if not math.isnan(ts) else "N/A",
        "avg_message_length": round(avg_msg_len, 2),
    }

    # Per-attribute final accuracy
    attr_acc = per_attribute_accuracy(rounds)
    for attr, acc_list in attr_acc.items():
        report[f"final_{attr}_accuracy"] = round(acc_list[-1], 4)

    return report


def print_report(metrics: dict):
    print("\n" + "=" * 55)
    print(f"  RUN: {metrics['run']}")
    print("=" * 55)
    print(f"  Rounds                      : {metrics['total_rounds']}")
    print(f"  Final cumulative accuracy   : {metrics['final_cumulative_accuracy']:.1%}")
    print(f"  Final lexicon size          : {metrics['final_lexicon_size']}")
    print(f"  Lexicon stabilized at round : {metrics['lexicon_stabilization_round']}")
    print(f"  Topographic similarity      : {metrics['topographic_similarity']}")
    print(f"  Avg message length (tokens) : {metrics['avg_message_length']}")
    print(f"  Shape accuracy              : {metrics['final_shape_accuracy']:.1%}")
    print(f"  Color accuracy              : {metrics['final_color_accuracy']:.1%}")
    print(f"  Position accuracy           : {metrics['final_position_accuracy']:.1%}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute Linguagenesis metrics from run logs.")
    parser.add_argument("--log-dir", default="data/logs", help="Directory containing run JSON logs")
    parser.add_argument("--out", default="data/results", help="Output directory for results JSON")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\nLoading logs from: {args.log_dir}")
    logs = load_logs(args.log_dir)

    if not logs:
        print("No valid log files found.")
        return

    all_metrics = []
    for fname, rounds in logs.items():
        metrics = compute_all_metrics(rounds, run_name=fname)
        print_report(metrics)
        all_metrics.append(metrics)

    # Save summary
    out_path = os.path.join(args.out, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics summary saved to: {out_path}\n")


if __name__ == "__main__":
    main()
