"""
Tests for CurriculumScheduler: config interpolation, bounds.
"""

from __future__ import annotations

import pytest

from training.curriculum import CurriculumScheduler, CurriculumConfig


class TestCurriculumScheduler:
    """Test curriculum config generation."""

    def test_get_config_at_start(self):
        s = CurriculumScheduler()
        c = s.get_config(0.0)
        assert c.traffic_rate >= 0
        assert 0 <= c.burst_probability <= 1
        assert 0 <= c.failure_injection_prob <= 1
        assert c.burst_size >= 1

    def test_get_config_at_end(self):
        s = CurriculumScheduler()
        c = s.get_config(1.0)
        assert c.traffic_rate >= s.start.traffic_rate
        assert c.burst_probability >= s.start.burst_probability
        assert c.failure_injection_prob >= s.start.failure_injection_prob

    def test_warmup_holds_start_config(self):
        s = CurriculumScheduler(warmup_fraction=0.2)
        c0 = s.get_config(0.0)
        c_warmup = s.get_config(0.1)
        assert c_warmup.traffic_rate == c0.traffic_rate
        assert c_warmup.burst_probability == c0.burst_probability

    def test_progress_clamped(self):
        s = CurriculumScheduler()
        c_neg = s.get_config(-0.5)
        c_over = s.get_config(1.5)
        assert c_neg.traffic_rate >= 0
        assert c_over.burst_size >= 1

    def test_custom_start_end(self):
        start = CurriculumConfig(traffic_rate=0.5, burst_probability=0.0, burst_size=1)
        end = CurriculumConfig(traffic_rate=5.0, burst_probability=0.2, burst_size=10)
        s = CurriculumScheduler(start=start, end=end)
        c0 = s.get_config(0.0)
        c1 = s.get_config(1.0)
        assert c0.traffic_rate == 0.5
        assert c1.traffic_rate == 5.0
        assert c1.burst_size == 10
