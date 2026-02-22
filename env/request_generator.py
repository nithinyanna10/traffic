"""
Request generator for the MicroserviceRoutingEnv.

Generates requests with randomized SLA deadlines, complexity, priority,
and cost sensitivity. Supports curriculum-driven traffic rate scaling.
"""

from __future__ import annotations

import random
from typing import Optional, List

from env.services import Request


class RequestGenerator:
    """Generates incoming API requests with varied characteristics."""

    def __init__(
        self,
        traffic_rate: float = 1.0,
        seed: Optional[int] = None,
        burst_probability: float = 0.0,
        burst_size: int = 3,
    ):
        """
        Args:
            traffic_rate: Multiplier for request generation probability.
                          1.0 = one request per step on average.
            seed: Random seed for reproducibility.
            burst_probability: Probability each step of generating a burst (0 = off).
            burst_size: Number of requests in a burst when one occurs.
        """
        self.traffic_rate = traffic_rate
        self._burst_probability = burst_probability
        self._burst_size = burst_size
        self._rng = random.Random(seed)

    def set_traffic_rate(self, rate: float) -> None:
        self.traffic_rate = rate

    def set_burst_config(self, probability: float, size: int) -> None:
        """Set burst probability and size (curriculum)."""
        self._burst_probability = probability
        self._burst_size = max(1, size)

    def maybe_get_burst(self) -> Optional[List[Request]]:
        """With probability burst_probability return a burst of requests; else None."""
        if self._burst_probability <= 0 or self._rng.random() >= self._burst_probability:
            return None
        return self.generate_burst(self._burst_size)

    def generate(self) -> Optional[Request]:
        """Generate a request (or None if no request this step)."""
        # Poisson-like: probability of generating a request
        if self._rng.random() > self.traffic_rate:
            return None

        # SLA deadlines range from tight (100ms) to relaxed (2000ms)
        sla_deadline = self._rng.uniform(100, 2000)

        # Complexity affects processing time
        complexity = self._rng.random()

        # Priority: skew toward medium priority
        priority = min(1.0, max(0.0, self._rng.gauss(0.5, 0.2)))

        # Cost sensitivity
        cost_sensitivity = self._rng.random()

        return Request(
            sla_deadline_ms=sla_deadline,
            complexity=complexity,
            priority=priority,
            cost_sensitivity=cost_sensitivity,
        )

    def generate_burst(self, count: int) -> list[Request]:
        """Generate a burst of requests (for stress testing)."""
        return [
            Request(
                sla_deadline_ms=self._rng.uniform(100, 2000),
                complexity=self._rng.random(),
                priority=min(1.0, max(0.0, self._rng.gauss(0.5, 0.2))),
                cost_sensitivity=self._rng.random(),
            )
            for _ in range(count)
        ]
