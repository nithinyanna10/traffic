#!/usr/bin/env python3
"""
Unified simulator runner.

Runs an agent in the MicroserviceRoutingEnv and optionally streams state
to the dashboard via WebSocket.

Usage:
    python simulator/runner.py --agent ppo --steps 5000 [--dashboard]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import env as _env_reg  # noqa

from agents.baselines import BASELINES
from training.metrics import MetricsTracker


def load_agent(agent_type: str, checkpoint: str = None):
    """Load an agent by type."""
    if agent_type == "ppo":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent()
        if checkpoint:
            agent.load(checkpoint)
        return agent
    elif agent_type == "impala":
        from agents.impala_agent import IMPALAAgent
        agent = IMPALAAgent()
        if checkpoint:
            agent.load(checkpoint)
        return agent
    elif agent_type in BASELINES:
        return BASELINES[agent_type]()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_simulation(
    agent,
    env,
    num_steps: int = 5000,
    dashboard_callback=None,
    verbose: bool = True,
):
    """Run a simulation, collecting metrics."""
    tracker = MetricsTracker()
    obs, info = env.reset()
    episode = 0
    episode_reward = 0.0

    for step in range(num_steps):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        tracker.on_step(reward, info)

        if dashboard_callback:
            state = info.copy()
            state["episode"] = episode
            state["episode_reward"] = episode_reward
            state["global_step"] = step
            dashboard_callback(state)

        if terminated or truncated:
            ep_metrics = tracker.on_episode_end(episode, collapsed=terminated)
            if verbose and episode % 10 == 0:
                print(
                    f"Episode {episode:4d} | "
                    f"reward={ep_metrics.total_reward:8.2f} | "
                    f"success={ep_metrics.success_rate:.2%} | "
                    f"p95_lat={ep_metrics.p95_latency:.1f}ms | "
                    f"cost={ep_metrics.avg_cost:.4f} | "
                    f"{'COLLAPSED' if ep_metrics.collapsed else 'OK'}"
                )
            obs, info = env.reset()
            episode_reward = 0.0
            episode += 1

    tracker.save()
    tracker.close()
    print(f"\nCompleted {episode} episodes in {num_steps} steps.")
    print(f"Rolling stats: {json.dumps(tracker.get_rolling_stats(), indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--agent", type=str, default="round_robin",
                        choices=list(BASELINES.keys()) + ["ppo", "impala"])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()

    env = gym.make("MicroserviceRouting-v0")
    agent = load_agent(args.agent, args.checkpoint)

    dashboard_cb = None
    if args.dashboard:
        # Import dashboard and push state
        from dashboard.server import update_state
        dashboard_cb = update_state

    print(f"Running {args.agent} for {args.steps} steps...")
    run_simulation(agent, env, args.steps, dashboard_cb, verbose=True)


if __name__ == "__main__":
    main()
