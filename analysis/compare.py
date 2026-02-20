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

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
