"""Tests for env/config.py EnvConfig."""

from __future__ import annotations

import pytest

from env.config import EnvConfig, DEFAULT_ENV_CONFIG


def test_default_config():
    c = EnvConfig()
    assert c.traffic_rate == 1.0
    assert c.max_steps == 1000
    assert c.burst_probability == 0.0
    assert c.burst_size == 3


def test_to_dict():
    c = EnvConfig(traffic_rate=2.0, max_steps=500)
    d = c.to_dict()
    assert d["traffic_rate"] == 2.0
    assert d["max_steps"] == 500
    assert "seed" in d


def test_from_dict():
    d = {"traffic_rate": 1.5, "max_steps": 2000, "burst_size": 5}
    c = EnvConfig.from_dict(d)
    assert c.traffic_rate == 1.5
    assert c.max_steps == 2000
    assert c.burst_size == 5


def test_from_dict_partial():
    c = EnvConfig.from_dict({})
    assert c.traffic_rate == 1.0
    assert c.max_steps == 1000
