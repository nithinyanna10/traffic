"""
Integration tests: short training runs, compare pipeline, env + agent together.
"""

from __future__ import annotations

import os
import sys
import tempfile

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env as _env_reg  # noqa: F401
from agents.baselines import BASELINES
from training.curriculum import CurriculumScheduler, CurriculumConfig


class TestShortTrainingRun:
    """Run a few steps with PPO and baselines to ensure no crashes."""

    def test_ppo_100_steps(self):
        from agents.ppo_agent import PPOAgent
        env = gym.make("MicroserviceRouting-v0")
        agent = PPOAgent(device="cpu")
        obs, _ = env.reset(seed=0)
        for _ in range(100):
            action = agent.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    def test_round_robin_200_steps(self):
        env = gym.make("MicroserviceRouting-v0")
        agent = BASELINES["round_robin"]()
        obs, _ = env.reset(seed=1)
        total_reward = 0
        for _ in range(200):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        assert isinstance(total_reward, (int, float))

    def test_curriculum_affects_env(self):
        env = gym.make("MicroserviceRouting-v0")
        unwrapped = env.unwrapped
        scheduler = CurriculumScheduler()
        unwrapped.set_traffic_rate(0.5)
        unwrapped.set_burst_config(0.0, 2)
        unwrapped.set_failure_injection(0.0)
        obs, _ = env.reset(seed=2)
        for step in range(50):
            config = scheduler.get_config(step / 100.0)
            unwrapped.set_traffic_rate(config.traffic_rate)
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        env.close()


class TestComparePipeline:
    """Test that compare.py can be invoked and produces output."""

    def test_compare_baselines_only(self):
        from analysis.compare import evaluate_agent, write_html_report
        env = gym.make("MicroserviceRouting-v0")
        results = {}
        for name, AgentCls in list(BASELINES.items())[:2]:  # first 2 only for speed
            agent = AgentCls()
            metrics = evaluate_agent(agent, env, num_episodes=3)
            results[name] = metrics
        env.close()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            write_html_report(results, path, 3, os.path.join(tmp, "out.json"))
            assert os.path.isfile(path)
            content = open(path).read()
            assert "Comparison" in content or "Agent" in content
