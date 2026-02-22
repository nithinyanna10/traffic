"""
Tests for RL and baseline agents: interface, shape, deterministic behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

import gymnasium as gym
import env as _env_reg  # noqa: F401
from agents.baselines import (
    BASELINES,
    RoundRobinAgent,
    LeastConnectionsAgent,
    LatencyHeuristicAgent,
)


class TestBaselineAgents:
    """Test rule-based agents."""

    @pytest.mark.parametrize("name", ["round_robin", "least_connections", "latency_heuristic"])
    def test_baseline_interface(self, env, name):
        agent_cls = BASELINES[name]
        agent = agent_cls()
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert action in range(9)
        if hasattr(agent, "reset"):
            agent.reset()

    def test_round_robin_cycles(self, env):
        agent = RoundRobinAgent()
        obs, _ = env.reset()
        actions = [agent.predict(obs) for _ in range(15)]
        # First 5 should be 0,1,2,3,4 then repeat
        assert actions[0] == 0
        assert actions[1] == 1
        assert actions[5] == 0

    def test_least_connections_returns_valid_service_or_drop(self, env):
        agent = LeastConnectionsAgent()
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert action in range(9)
        # With healthy env, usually routes to one of 0-4
        assert 0 <= action <= 4 or action == 7

    def test_latency_heuristic_returns_action(self, env):
        agent = LatencyHeuristicAgent()
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert action in range(9)


class TestPPOAgent:
    """Test PPO agent (requires env and optional checkpoint)."""

    def test_ppo_predict_shape(self, env):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(device="cpu")
        obs, _ = env.reset()
        action = agent.predict(obs, deterministic=True)
        assert action in range(9)

    def test_ppo_save_load_roundtrip(self, env, tmp_path):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(device="cpu")
        obs, _ = env.reset()
        a1 = agent.predict(obs, deterministic=True)
        path = str(tmp_path / "ppo_test")
        agent.save(path)
        agent2 = PPOAgent(device="cpu")
        agent2.load(path)
        a2 = agent2.predict(obs, deterministic=True)
        assert a1 == a2
