#!/usr/bin/env python3
"""
Evaluate all agents (baselines + optional PPO/DQN) over multiple seeds and print a table.

Usage:
    python scripts/eval_all_agents.py [--episodes 20] [--seeds 0 1 2] [--ppo path] [--dqn path]
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


def run_agent(agent, env, num_episodes: int, seed: int) -> dict:
    rewards = []
    success_rates = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep * 1000)
        if hasattr(agent, "reset"):
            agent.reset()
        ep_rew = 0.0
        successes = []
        while True:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            r = info.get("response")
            if r is not None and "success" in r:
                successes.append(r["success"])
            if terminated or truncated:
                break
        rewards.append(ep_rew)
        success_rates.append(np.mean(successes) if successes else 0.0)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_success": float(np.mean(success_rates)),
        "std_success": float(np.std(success_rates)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ppo", type=str, default=None)
    parser.add_argument("--dqn", type=str, default=None)
    args = parser.parse_args()

    env = gym.make("MicroserviceRouting-v0")
    results = {}

    for name, AgentCls in BASELINES.items():
        agent = AgentCls()
        all_r, all_s = [], []
        for seed in args.seeds:
            r, s = [], []
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=seed + ep * 1000)
                if hasattr(agent, "reset"):
                    agent.reset()
                ep_rew = 0.0
                succ = []
                while True:
                    action = agent.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_rew += reward
                    if info.get("response", {}).get("success") is not None:
                        succ.append(info["response"]["success"])
                    if terminated or truncated:
                        break
                r.append(ep_rew)
                s.append(np.mean(succ) if succ else 0.0)
            all_r.extend(r)
            all_s.extend(s)
        results[name] = {"mean_reward": np.mean(all_r), "std_reward": np.std(all_r), "mean_success": np.mean(all_s), "std_success": np.std(all_s)}

    if args.ppo and os.path.isfile((args.ppo or "") + ".zip"):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(device="cpu")
        agent.load(args.ppo)
        all_r, all_s = [], []
        for seed in args.seeds:
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=seed + ep * 1000)
                ep_rew = 0.0
                succ = []
                while True:
                    action = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_rew += reward
                    if info.get("response", {}).get("success") is not None:
                        succ.append(info["response"]["success"])
                    if terminated or truncated:
                        break
                all_r.append(ep_rew)
                all_s.append(np.mean(succ) if succ else 0.0)
        results["ppo"] = {"mean_reward": np.mean(all_r), "std_reward": np.std(all_r), "mean_success": np.mean(all_s), "std_success": np.std(all_s)}

    if args.dqn and os.path.isfile((args.dqn or "") + ".zip"):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(device="cpu")
        agent.load(args.dqn)
        all_r, all_s = [], []
        for seed in args.seeds:
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=seed + ep * 1000)
                ep_rew = 0.0
                succ = []
                while True:
                    action = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_rew += reward
                    if info.get("response", {}).get("success") is not None:
                        succ.append(info["response"]["success"])
                    if terminated or truncated:
                        break
                all_r.append(ep_rew)
                all_s.append(np.mean(succ) if succ else 0.0)
        results["dqn"] = {"mean_reward": np.mean(all_r), "std_reward": np.std(all_r), "mean_success": np.mean(all_s), "std_success": np.std(all_s)}

    env.close()

    print(f"\n{'Agent':<22} {'Reward':>14} {'Success':>14}")
    print("-" * 52)
    for name, m in results.items():
        print(f"{name:<22} {m['mean_reward']:>8.2f} ± {m['std_reward']:<5.2f} {m['mean_success']:>8.2%} ± {m['std_success']:.2%}")
    print("-" * 52)


if __name__ == "__main__":
    main()
