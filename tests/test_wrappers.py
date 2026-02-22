"""Tests for env/wrappers.py."""

from __future__ import annotations

import numpy as np
import pytest

import gymnasium as gym
import env as _env_reg  # noqa: F401
from env.wrappers import NormalizeObservation, TimeLimitOverride


def test_time_limit_override():
    env = gym.make("MicroserviceRouting-v0")
    wrapped = TimeLimitOverride(env, max_episode_steps=5)
    obs, _ = wrapped.reset(seed=0)
    steps = 0
    while steps < 20:
        obs, _, term, trunc, _ = wrapped.step(0)
        steps += 1
        if term or trunc:
            break
    assert steps == 5
    assert trunc
    env.close()


def test_normalize_observation():
    env = gym.make("MicroserviceRouting-v0")
    wrapped = NormalizeObservation(env, mode="minmax")
    obs, _ = wrapped.reset(seed=0)
    for _ in range(10):
        obs, _, term, trunc, _ = wrapped.step(env.action_space.sample())
        assert obs.shape == (34,)
        if term or trunc:
            wrapped.reset()
    env.close()
