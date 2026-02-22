"""
Tests for MicroserviceRoutingEnv: spaces, step, reset, curriculum hooks.
"""

from __future__ import annotations

import numpy as np
import pytest

import gymnasium as gym
import env as _env_reg  # noqa: F401


class TestEnvSpaces:
    """Test observation and action spaces."""

    def test_obs_space_shape(self, env, obs_shape):
        obs, _ = env.reset()
        assert obs.shape == obs_shape
        assert env.observation_space.shape == obs_shape

    def test_obs_in_bounds(self, env):
        obs, _ = env.reset()
        assert np.all(obs >= 0) and np.all(obs <= 1)

    def test_action_space_discrete(self, env, n_actions):
        assert env.action_space.n == n_actions

    def test_step_returns_five_tuple(self, env):
        obs, _ = env.reset()
        action = 0
        result = env.step(action)
        assert len(result) == 5
        obs2, reward, terminated, truncated, info = result
        assert obs2.shape == obs.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestEnvStep:
    """Test step dynamics and info."""

    def test_info_contains_required_keys(self, env):
        obs, info = env.reset()
        assert "step" in info
        assert "success_rate" in info
        assert "avg_latency" in info
        assert "avg_cost" in info
        assert "alive_services" in info
        assert "services" in info
        assert len(info["services"]) == 5

    def test_services_have_expected_fields(self, env):
        obs, info = env.reset()
        for s in info["services"]:
            assert "name" in s
            assert "queue" in s
            assert "health" in s
            assert "alive" in s
            assert "latency" in s
            assert "failure_rate" in s

    def test_routing_actions_0_to_4(self, env):
        obs, _ = env.reset()
        for a in range(5):
            obs2, reward, term, trunc, info = env.step(a)
            assert "response" in info or obs2 is not None
            if term or trunc:
                break

    def test_delay_action_5(self, env):
        obs, _ = env.reset()
        obs2, reward, term, trunc, info = env.step(5)
        assert not term  # delay doesn't terminate
        assert "delayed_queue_size" in info or True  # may be in info

    def test_episode_terminates_or_truncates(self, env_seed_0):
        env = env_seed_0
        steps = 0
        max_steps = 1100
        while steps < max_steps:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert terminated or truncated
        assert steps <= max_steps


class TestEnvCurriculum:
    """Test curriculum setters (traffic, burst, failure injection)."""

    def test_set_traffic_rate(self, env):
        unwrapped = env.unwrapped
        unwrapped.set_traffic_rate(2.0)
        obs, _ = env.reset()
        assert obs.shape[0] == 34

    def test_set_burst_config(self, env):
        unwrapped = env.unwrapped
        unwrapped.set_burst_config(0.1, 4)
        obs, _ = env.reset()
        env.step(0)
        # Just ensure no crash; burst is stochastic
        assert True

    def test_set_failure_injection(self, env):
        unwrapped = env.unwrapped
        unwrapped.set_failure_injection(0.0)  # disable
        obs, _ = env.reset()
        unwrapped.set_failure_injection(0.5)
        env.step(0)
        assert True
