#!/usr/bin/env python3
"""
Benchmark agents over multiple seeds; report mean ± std reward and success rate.

Usage:
    python scripts/benchmark.py [--seeds 0 1 2 3 4] [--episodes 20] [--agent ppo] [--checkpoint path]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import env as _env_reg  # noqa: F401
from agents.baselines import BASELINES


def run_agent(agent, env, num_episodes: int, seed: int) -> tuple[list[float], list[float]]:
    rewards = []
    success_rates = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep * 1000)
        if hasattr(agent, "reset"):
            agent.reset()
        episode_reward = 0.0
        step_successes = []
        while True:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if info.get("response", {}).get("success") is not None:
                step_successes.append(info["response"]["success"])
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        success_rates.append(np.mean(step_successes) if step_successes else 0.0)
    return rewards, success_rates


def main():
    parser = argparse.ArgumentParser(description="Benchmark agents over multiple seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--agent", type=str, default="round_robin", choices=["round_robin", "least_connections", "latency_heuristic", "ppo"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    env = gym.make("MicroserviceRouting-v0")
    if args.agent == "ppo":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(device="cpu")
        if args.checkpoint and os.path.isfile(args.checkpoint + ".zip"):
            agent.load(args.checkpoint)
    else:
        agent = BASELINES[args.agent]()

    all_rewards = []
    all_success = []
    for seed in args.seeds:
        r, s = run_agent(agent, env, args.episodes, seed)
        all_rewards.extend(r)
        all_success.extend(s)

    env.close()

    mean_r = np.mean(all_rewards)
    std_r = np.std(all_rewards)
    mean_s = np.mean(all_success)
    std_s = np.std(all_success)
    print(f"Agent: {args.agent} | Seeds: {args.seeds} | Episodes per seed: {args.episodes}")
    print(f"Reward:    {mean_r:.2f} ± {std_r:.2f}")
    print(f"Success:   {mean_s:.2%} ± {std_s:.2%}")


if __name__ == "__main__":
    main()
