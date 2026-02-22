#!/usr/bin/env python3
"""
Example: run environment with curriculum (manual loop, no RL).
Demonstrates set_traffic_rate, set_burst_config, set_failure_injection.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import env as _env_reg  # noqa: F401
from training.curriculum import CurriculumScheduler


def main():
    env = gym.make("MicroserviceRouting-v0")
    unwrapped = env.unwrapped
    scheduler = CurriculumScheduler()
    obs, _ = env.reset(seed=42)

    total_steps = 1000
    for step in range(total_steps):
        progress = step / total_steps
        config = scheduler.get_config(progress)
        unwrapped.set_traffic_rate(config.traffic_rate)
        unwrapped.set_burst_config(config.burst_probability, config.burst_size)
        unwrapped.set_failure_injection(config.failure_injection_prob)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print("Done: curriculum demo over", total_steps, "steps.")


if __name__ == "__main__":
    main()
