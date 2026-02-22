#!/usr/bin/env python3
"""
Example: run a baseline agent for N steps and print summary stats.

Usage:
    python examples/run_baseline.py [--agent round_robin] [--steps 500]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import env as _env_reg  # noqa: F401
from agents.baselines import BASELINES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="round_robin", choices=list(BASELINES.keys()))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("MicroserviceRouting-v0")
    agent = BASELINES[args.agent]()

    obs, info = env.reset(seed=args.seed)
    total_reward = 0
    rewards_per_ep = []
    current_ep_reward = 0

    for step in range(args.steps):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        current_ep_reward += reward
        if terminated or truncated:
            rewards_per_ep.append(current_ep_reward)
            current_ep_reward = 0
            obs, info = env.reset()

    env.close()

    if rewards_per_ep:
        print(f"Agent: {args.agent} | Steps: {args.steps} | Episodes: {len(rewards_per_ep)}")
        print(f"Mean episode reward: {np.mean(rewards_per_ep):.2f} ± {np.std(rewards_per_ep):.2f}")
    else:
        print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
