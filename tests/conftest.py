"""
Pytest configuration and shared fixtures for RL-Traffic-Control.
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env as _env_reg  # noqa: F401


@pytest.fixture
def env():
    """Create a fresh MicroserviceRoutingEnv with fixed seed."""
    e = gym.make("MicroserviceRouting-v0")
    obs, info = e.reset(seed=42)
    yield e
    e.close()


@pytest.fixture
def env_seed_0():
    """Env with seed 0 for deterministic tests."""
    e = gym.make("MicroserviceRouting-v0")
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture
def obs_shape():
    """Expected observation shape (34-dim)."""
    return (34,)


@pytest.fixture
def n_actions():
    """Discrete action space size."""
    return 9


@pytest.fixture
def n_services():
    """Number of services in the env."""
    return 5


@pytest.fixture
def valid_actions():
    """All valid discrete actions."""
    return list(range(9))
