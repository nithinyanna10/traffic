#!/usr/bin/env python3
"""
Visualization utilities for training analysis.

Generates matplotlib plots for:
  - Training curves (reward, success rate over episodes)
  - Policy evolution (action distribution over time)
  - Failure frequency over training

Usage:
    python analysis/visualize.py --metrics ./logs/metrics/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_training_curves(metrics: list[dict], output_dir: str) -> None:
    """Plot reward and success rate over episodes."""
    episodes = [m["episode"] for m in metrics]
    rewards = [m["total_reward"] for m in metrics]
    success = [m["success_rate"] for m in metrics]
    latency = [m["avg_latency"] for m in metrics]
    cost = [m["avg_cost"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RL-Traffic-Control: Training Curves", fontsize=14, fontweight="bold")

    # Reward
    ax = axes[0, 0]
    ax.plot(episodes, rewards, color="#3b82f6", linewidth=0.8, alpha=0.5)
    if len(rewards) > 20:
        window = max(len(rewards) // 50, 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(rewards)), smoothed, color="#3b82f6", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward")
    ax.grid(True, alpha=0.3)

    # Success Rate
    ax = axes[0, 1]
    ax.plot(episodes, success, color="#22d67a", linewidth=0.8, alpha=0.5)
    if len(success) > 20:
        smoothed = np.convolve(success, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(success)), smoothed, color="#22d67a", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("SLA Success Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Latency
    ax = axes[1, 0]
    ax.plot(episodes, latency, color="#f5a623", linewidth=0.8, alpha=0.5)
    if len(latency) > 20:
        smoothed = np.convolve(latency, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(latency)), smoothed, color="#f5a623", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Average Latency")
    ax.grid(True, alpha=0.3)

    # Cost
    ax = axes[1, 1]
    ax.plot(episodes, cost, color="#8b5cf6", linewidth=0.8, alpha=0.5)
    if len(cost) > 20:
        smoothed = np.convolve(cost, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(cost)), smoothed, color="#8b5cf6", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Cost")
    ax.set_title("Average Cost per Request")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_comparison(comparison_path: str, output_dir: str) -> None:
    """Plot comparison bar charts from compare.py output."""
    with open(comparison_path) as f:
        data = json.load(f)

    agents = list(data.keys())
    metrics_names = ["mean_reward", "mean_success_rate", "mean_p95_latency", "mean_cost", "collapse_rate"]
    labels = ["Reward", "Success Rate", "P95 Latency (ms)", "Cost", "Collapse Rate"]
    colors = ["#3b82f6", "#22d67a", "#f5a623", "#8b5cf6", "#ef4444"]

    fig, axes = plt.subplots(1, len(metrics_names), figsize=(4 * len(metrics_names), 5))
    fig.suptitle("Agent Comparison", fontsize=14, fontweight="bold")

    for i, (metric, label, color) in enumerate(zip(metrics_names, labels, colors)):
        ax = axes[i]
        values = [data[a].get(metric, 0) for a in agents]
        bars = ax.bar(agents, values, color=color, alpha=0.8)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--metrics", type=str, default="./logs/metrics/metrics.json")
    parser.add_argument("--comparison", type=str, default="./logs/comparison.json")
    parser.add_argument("--output-dir", type=str, default="./logs/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.metrics):
        with open(args.metrics) as f:
            metrics = json.load(f)
        plot_training_curves(metrics, args.output_dir)
    else:
        print(f"No metrics file found at {args.metrics}")

    if os.path.exists(args.comparison):
        plot_comparison(args.comparison, args.output_dir)
    else:
        print(f"No comparison file found at {args.comparison}")


if __name__ == "__main__":
    main()
