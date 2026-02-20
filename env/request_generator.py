"""
Request generator for the MicroserviceRoutingEnv.

Generates requests with randomized SLA deadlines, complexity, priority,
and cost sensitivity. Supports curriculum-driven traffic rate scaling.
"""

from __future__ import annotations

import random
from typing import Optional

from env.services import Request


class RequestGenerator:
    """Generates incoming API requests with varied characteristics."""

    def __init__(
        self,
        traffic_rate: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            traffic_rate: Multiplier for request generation probability.
                          1.0 = one request per step on average.
            seed: Random seed for reproducibility.
        """
        self.traffic_rate = traffic_rate
        self._rng = random.Random(seed)

    def set_traffic_rate(self, rate: float) -> None:
        self.traffic_rate = rate

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
