"""
analysis/visualize.py — Plots for the Linguagenesis experiment.

Generates:
  1. accuracy_over_time.png       — cumulative + rolling accuracy per run
  2. lexicon_growth.png           — lexicon size over rounds per run
  3. entropy_curve.png            — symbol entropy over rounds per run
  4. per_attribute_accuracy.png   — shape vs color vs position accuracy
  5. topsim_over_time.png         — topographic similarity evolving over rounds
  6. message_length.png           — symbol message length over rounds
  7. symbol_heatmap.png           — which symbols were used most (final run)

Usage:
    python analysis/visualize.py --log-dir data/logs/
    python analysis/visualize.py --log-dir data/logs/ --out data/results/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import Counter

# Import our metrics module (must be run from project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.metrics import (
    load_logs,
    accuracy_over_time,
    cumulative_accuracy,
    rolling_accuracy,
    per_attribute_accuracy,
    lexicon_growth,
    symbol_entropy_over_time,
    topsim_over_time,
    message_length_over_time,
)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
ATTR_COLORS = {"shape": "#4C72B0", "color": "#DD8452", "position": "#55A868"}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f8",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

def short_name(fname: str) -> str:
    """Shorten a filename to a readable run label."""
    return fname.replace("run_", "").replace(".json", "")


# ---------------------------------------------------------------------------
# Plot 1: Accuracy over time
# ---------------------------------------------------------------------------

def plot_accuracy(logs: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Communication Accuracy Over Time", fontweight="bold", fontsize=14)

    for i, (fname, rounds) in enumerate(logs.items()):
        color = COLORS[i % len(COLORS)]
        label = short_name(fname)
        round_nums = [r["round"] for r in rounds]

        # Left: cumulative accuracy
        cum = cumulative_accuracy(rounds)
        axes[0].plot(round_nums, cum, color=color, label=label, linewidth=2)

        # Right: rolling accuracy (window=10)
        roll = rolling_accuracy(rounds, window=10)
        axes[1].plot(round_nums, roll, color=color, label=label, linewidth=2, alpha=0.85)

    for ax, title in zip(axes, ["Cumulative Accuracy", "Rolling Accuracy (window=10)"]):
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Lexicon growth
# ---------------------------------------------------------------------------

def plot_lexicon_growth(logs: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Lexicon Growth Over Rounds")

    for i, (fname, rounds) in enumerate(logs.items()):
        color = COLORS[i % len(COLORS)]
        round_nums = [r["round"] for r in rounds]
        sizes = lexicon_growth(rounds)
        ax.plot(round_nums, sizes, color=color, label=short_name(fname), linewidth=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Lexicon Size (entries)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "lexicon_growth.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Symbol entropy
# ---------------------------------------------------------------------------

def plot_entropy(logs: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Symbol Entropy Over Time\n(lower = more structured symbol usage)")

    for i, (fname, rounds) in enumerate(logs.items()):
        color = COLORS[i % len(COLORS)]
        round_nums = [r["round"] for r in rounds]
        entropies = symbol_entropy_over_time(rounds, window=20)
        ax.plot(round_nums, entropies, color=color, label=short_name(fname), linewidth=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "entropy_curve.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: Per-attribute accuracy
# ---------------------------------------------------------------------------

def plot_per_attribute(logs: dict, out_dir: str):
    n_runs = len(logs)
    fig, axes = plt.subplots(1, n_runs, figsize=(8 * n_runs, 5), squeeze=False)
    fig.suptitle("Per-Attribute Cumulative Accuracy", fontweight="bold", fontsize=14)

    for col, (fname, rounds) in enumerate(logs.items()):
        ax = axes[0][col]
        ax.set_title(short_name(fname))
        round_nums = [r["round"] for r in rounds]
        attr_acc = per_attribute_accuracy(rounds)
        for attr, acc_list in attr_acc.items():
            ax.plot(round_nums, acc_list,
                    color=ATTR_COLORS[attr], label=attr.capitalize(), linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "per_attribute_accuracy.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 5: TopSim over time
# ---------------------------------------------------------------------------

def plot_topsim(logs: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Topographic Similarity Over Time\n(higher = more compositional language)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    any_data = False
    for i, (fname, rounds) in enumerate(logs.items()):
        color = COLORS[i % len(COLORS)]
        ts_series = topsim_over_time(rounds, window=30, step=5)
        if ts_series:
            xs = [t[0] for t in ts_series]
            ys = [t[1] for t in ts_series if not (isinstance(t[1], float) and
                  (t[1] != t[1]))]  # filter NaN
            xs_clean = [xs[j] for j, t in enumerate(ts_series)
                        if not (isinstance(t[1], float) and t[1] != t[1])]
            if xs_clean:
                ax.plot(xs_clean, ys, color=color, label=short_name(fname),
                        linewidth=2, marker="o", markersize=4)
                any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "Not enough successful rounds\nto compute TopSim",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

    ax.set_xlabel("Round")
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "topsim_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 6: Message length over time
# ---------------------------------------------------------------------------

def plot_message_length(logs: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Symbol Message Length Over Rounds\n(convergence toward 3 tokens = F-G-H structure)")
    ax.axhline(y=3, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Expected (3 tokens)")

    for i, (fname, rounds) in enumerate(logs.items()):
        color = COLORS[i % len(COLORS)]
        round_nums = [r["round"] for r in rounds]
        lengths = message_length_over_time(rounds)
        # Rolling average to smooth noise
        window = 10
        smoothed = []
        for j in range(len(lengths)):
            start = max(0, j - window + 1)
            smoothed.append(sum(lengths[start:j+1]) / (j - start + 1))
        ax.plot(round_nums, smoothed, color=color, label=short_name(fname), linewidth=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Avg tokens per message (rolling)")
    ax.set_ylim(0, 6)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "message_length.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 7: Symbol usage heatmap (last run only)
# ---------------------------------------------------------------------------

def plot_symbol_heatmap(logs: dict, out_dir: str):
    # Use the last (most complete) run
    fname, rounds = list(logs.items())[-1]

    # Count token usage
    token_counts = Counter()
    for r in rounds:
        for token in r.get("symbol_message", "").split("-"):
            token_counts[token] += 1

    # Separate into letter groups
    letter_groups = {}
    for token, count in token_counts.items():
        if len(token) >= 2:
            letter = token[0]
            number = token[1:]
            letter_groups.setdefault(letter, {})[number] = count

    letters = sorted(letter_groups.keys())
    all_numbers = sorted({n for g in letter_groups.values() for n in g})

    if not letters or not all_numbers:
        print("  Skipping heatmap — no token data.")
        return

    matrix = np.zeros((len(letters), len(all_numbers)))
    for i, letter in enumerate(letters):
        for j, number in enumerate(all_numbers):
            matrix[i][j] = letter_groups[letter].get(number, 0)

    fig, ax = plt.subplots(figsize=(max(6, len(all_numbers) * 1.5), max(4, len(letters) * 1.2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(all_numbers)))
    ax.set_xticklabels(all_numbers)
    ax.set_yticks(range(len(letters)))
    ax.set_yticklabels(letters)
    ax.set_xlabel("Number suffix")
    ax.set_ylabel("Symbol letter")
    ax.set_title(f"Symbol Token Usage Heatmap\n({short_name(fname)})")

    plt.colorbar(im, ax=ax, label="Times used")

    # Annotate cells with counts
    for i in range(len(letters)):
        for j in range(len(all_numbers)):
            val = int(matrix[i][j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, color="black" if val < matrix.max() * 0.7 else "white")

    plt.tight_layout()
    path = os.path.join(out_dir, "symbol_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Linguagenesis visualizations.")
    parser.add_argument("--log-dir", default="data/logs", help="Directory with run JSON logs")
    parser.add_argument("--out", default="data/results", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    setup_style()

    print(f"\nLoading logs from: {args.log_dir}")
    logs = load_logs(args.log_dir)

    if not logs:
        print("No valid log files found.")
        return

    print(f"\nGenerating plots → {args.out}/")
    plot_accuracy(logs, args.out)
    plot_lexicon_growth(logs, args.out)
    plot_entropy(logs, args.out)
    plot_per_attribute(logs, args.out)
    plot_topsim(logs, args.out)
    plot_message_length(logs, args.out)
    plot_symbol_heatmap(logs, args.out)

    print("\nAll plots generated successfully.\n")


if __name__ == "__main__":
    main()
