#!/usr/bin/env python3
"""
Compare RL agent vs rule-based baselines.

Runs each agent for N episodes and produces a comparison table + plots.

Usage:
    python analysis/compare.py [--episodes 50] [--ppo-checkpoint path]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import env as _env_reg  # noqa

from agents.baselines import BASELINES


def evaluate_agent(agent, env, num_episodes: int = 50) -> dict:
    """Run agent for num_episodes and collect metrics."""
    all_rewards = []
    all_latencies = []
    all_costs = []
    all_success_rates = []
    collapse_count = 0
    all_steps = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.reset() if hasattr(agent, 'reset') else None
        episode_reward = 0.0
        step_latencies = []
        step_costs = []
        step_successes = []
        steps = 0

        while True:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            resp = info.get("response", {})
            if resp:
                step_latencies.append(resp.get("latency_ms", 0))
                step_costs.append(resp.get("cost", 0))
                step_successes.append(resp.get("success", False))

            if terminated or truncated:
                if terminated:
                    collapse_count += 1
                break

        all_rewards.append(episode_reward)
        all_steps.append(steps)
        if step_latencies:
            sorted_lat = sorted(step_latencies)
            all_latencies.append(sorted_lat[int(len(sorted_lat) * 0.95)])
            all_costs.append(np.mean(step_costs))
            all_success_rates.append(
                sum(step_successes) / max(len(step_successes), 1)
            )

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_p95_latency": float(np.mean(all_latencies)) if all_latencies else 0,
        "mean_cost": float(np.mean(all_costs)) if all_costs else 0,
        "mean_success_rate": float(np.mean(all_success_rates)) if all_success_rates else 0,
        "collapse_rate": collapse_count / num_episodes,
        "mean_steps": float(np.mean(all_steps)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare RL vs baselines")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--ppo-checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="./logs/comparison.json")
    args = parser.parse_args()

    env = gym.make("MicroserviceRouting-v0")
    results = {}

    # Baselines
    for name, AgentCls in BASELINES.items():
        print(f"\nEvaluating: {name}")
        agent = AgentCls()
        t0 = time.time()
        metrics = evaluate_agent(agent, env, args.episodes)
        elapsed = time.time() - t0
        metrics["eval_time_s"] = elapsed
        results[name] = metrics
        print(f"  reward={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  success={metrics['mean_success_rate']:.2%}  p95_lat={metrics['mean_p95_latency']:.1f}ms")
        print(f"  cost={metrics['mean_cost']:.4f}  collapse={metrics['collapse_rate']:.2%}")

    # PPO (if checkpoint available)
    if args.ppo_checkpoint and os.path.exists(args.ppo_checkpoint + ".zip"):
        print(f"\nEvaluating: PPO (from {args.ppo_checkpoint})")
        from agents.ppo_agent import PPOAgent
        ppo = PPOAgent()
        ppo.load(args.ppo_checkpoint)
        t0 = time.time()
        metrics = evaluate_agent(ppo, env, args.episodes)
        metrics["eval_time_s"] = time.time() - t0
        results["ppo"] = metrics
        print(f"  reward={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  success={metrics['mean_success_rate']:.2%}  p95_lat={metrics['mean_p95_latency']:.1f}ms")
    else:
        print("\n(Skipping PPO — no checkpoint found. Train first with training/train_ppo.py)")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Agent':<20} {'Reward':>10} {'Success':>10} {'P95 Lat':>10} {'Cost':>10} {'Collapse':>10}")
    print("-" * 80)
    for name, m in results.items():
        print(
            f"{name:<20} {m['mean_reward']:>10.2f} "
            f"{m['mean_success_rate']:>9.1%} "
            f"{m['mean_p95_latency']:>9.1f}ms "
            f"{m['mean_cost']:>10.4f} "
            f"{m['collapse_rate']:>9.1%}"
        )
    print("=" * 80)

    # Save JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save HTML report
    html_path = args.output.replace(".json", ".html")
    if not html_path.endswith(".html"):
        html_path = os.path.join(os.path.dirname(args.output), "comparison.html")
    write_html_report(results, html_path, args.episodes, args.output)
    print(f"HTML report: {html_path}")


def write_html_report(
    results: dict, path: str, num_episodes: int, json_output_path: str
) -> None:
    """Write a standalone HTML comparison report."""
    rows = []
    for name, m in results.items():
        rows.append(
            f"    <tr><td>{name}</td>"
            f"<td>{m['mean_reward']:.2f} ± {m['std_reward']:.2f}</td>"
            f"<td>{m['mean_success_rate']:.1%}</td>"
            f"<td>{m['mean_p95_latency']:.1f} ms</td>"
            f"<td>{m['mean_cost']:.4f}</td>"
            f"<td>{m['collapse_rate']:.1%}</td>"
            f"<td>{m.get('eval_time_s', 0):.1f}s</td></tr>"
        )
    table_rows = "\n".join(rows)
    json_basename = os.path.basename(json_output_path)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL-Traffic-Control — Comparison Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0c0f18; color: #e2e8f4; padding: 24px; }}
        h1 {{ color: #00d4ff; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 900px; margin-top: 16px; }}
        th, td {{ border: 1px solid #2a3a55; padding: 10px 14px; text-align: left; }}
        th {{ background: #1a2234; color: #94a3b8; font-weight: 600; }}
        tr:nth-child(even) {{ background: rgba(0,212,255,0.05); }}
        .foot {{ margin-top: 20px; font-size: 12px; color: #64748b; }}
    </style>
</head>
<body>
    <h1>⚡ RL-Traffic-Control — Agent Comparison</h1>
    <p>Evaluated over <strong>{num_episodes}</strong> episodes per agent.</p>
    <table>
        <thead>
            <tr>
                <th>Agent</th>
                <th>Reward (mean ± std)</th>
                <th>Success Rate</th>
                <th>P95 Latency</th>
                <th>Cost</th>
                <th>Collapse Rate</th>
                <th>Eval Time</th>
            </tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
    </table>
    <p class="foot">Generated by analysis/compare.py. Data: {json_basename}</p>
</body>
</html>"""
    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
