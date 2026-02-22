#!/usr/bin/env python3
"""
Full pipeline example: train PPO → save → evaluate over seeds → print comparison table.

This script does not call compare.py or visualize.py; it implements a minimal
in-process training + evaluation and prints results. For full experiment
reporting use scripts/run_experiment.py.

Usage:
    python examples/full_pipeline.py [--train-steps 5000] [--eval-episodes 10] [--eval-seeds 3]
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
from agents.ppo_agent import PPOAgent
from agents.baselines import BASELINES


def train_ppo(env, steps: int, seed: int = 0) -> PPOAgent:
    agent = PPOAgent(device="cpu")
    obs, _ = env.reset(seed=seed)
    # Short in-process "training" by repeatedly stepping (no SB3 learn; for demo only)
    for _ in range(min(steps, 2000)):
        action = agent.predict(obs, deterministic=False)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    return agent


def evaluate(agent, env, num_episodes: int, seeds: list[int], deterministic: bool = True) -> dict:
    rewards = []
    success_rates = []
    for seed in seeds:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep * 1000)
            if hasattr(agent, "reset"):
                agent.reset()
            ep_rew = 0.0
            succ = []
            while True:
                action = agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rew += reward
                if info.get("response", {}).get("success") is not None:
                    succ.append(info["response"]["success"])
                if terminated or truncated:
                    break
            rewards.append(ep_rew)
            success_rates.append(np.mean(succ) if succ else 0.0)
    return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards), "mean_success": np.mean(success_rates), "std_success": np.std(success_rates)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="./logs/pipeline_demo")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = gym.make("MicroserviceRouting-v0")

    # 1. Train PPO (lightweight: just step loop for demo; for real training use train_ppo.py)
    print("Training PPO (demo loop)...")
    t0 = time.time()
    agent = train_ppo(env, args.train_steps)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 2. Evaluate PPO
    seeds = list(range(args.eval_seeds))
    ppo_metrics = evaluate(agent, env, args.eval_episodes, seeds)
    print(f"  PPO reward: {ppo_metrics['mean_reward']:.2f} ± {ppo_metrics['std_reward']:.2f}  success: {ppo_metrics['mean_success']:.2%}")

    # 3. Evaluate baselines
    results = {"ppo": ppo_metrics}
    for name, AgentCls in BASELINES.items():
        a = AgentCls()
        m = evaluate(a, env, args.eval_episodes, seeds)
        results[name] = m
        print(f"  {name}: reward {m['mean_reward']:.2f} ± {m['std_reward']:.2f}  success {m['mean_success']:.2%}")

    env.close()

    # 4. Table
    print("\n" + "=" * 60)
    print(f"{'Agent':<22} {'Reward':>14} {'Success':>14}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<22} {m['mean_reward']:>8.2f} ± {m['std_reward']:<5.2f} {m['mean_success']:>8.2%} ± {m['std_success']:.2%}")
    print("=" * 60)
    print("Output dir:", args.out_dir)


if __name__ == "__main__":
    main()
