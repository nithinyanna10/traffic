#!/usr/bin/env python3
"""
Quick healthcheck: create env, run one step with each baseline and PPO, no crashes.
Use in CI or after install to verify the project runs.

Usage:
    python scripts/healthcheck.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import env as _env_reg  # noqa: F401
from agents.baselines import BASELINES


def main():
    env = gym.make("MicroserviceRouting-v0")
    obs, _ = env.reset(seed=0)

    for name, AgentCls in BASELINES.items():
        agent = AgentCls()
        a = agent.predict(obs)
        assert 0 <= a < 9, f"{name} returned invalid action {a}"
    print("Baselines OK")

    from agents.ppo_agent import PPOAgent
    agent = PPOAgent(device="cpu")
    a = agent.predict(obs, deterministic=True)
    assert 0 <= a < 9, f"PPO returned invalid action {a}"
    print("PPO OK")

    env.close()
    print("Healthcheck passed.")


if __name__ == "__main__":
    main()
