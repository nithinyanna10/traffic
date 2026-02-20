"""
Rule-based baseline agents for comparison.

All agents implement the same interface:
  agent.select_action(obs) -> int
  agent.reset()
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class BaselineAgent(ABC):
    """Base class for rule-based routing agents."""

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        ...

    def reset(self) -> None:
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Alias for select_action to match RL agent interface."""
        return self.select_action(obs)


class RoundRobinAgent(BaselineAgent):
    """Routes requests to services in round-robin order (0-4)."""

    def __init__(self):
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def select_action(self, obs: np.ndarray) -> int:
        action = self._counter % 5
        self._counter += 1
        return action


class LeastConnectionsAgent(BaselineAgent):
    """Routes to the service with the smallest queue.

    Observation layout:
      [0:5]  = queue sizes
      [5:10] = health scores
    """

    def select_action(self, obs: np.ndarray) -> int:
        queue_sizes = obs[0:5]
        health = obs[5:10]
        # Only consider alive services (health > 0.1)
        candidates = [i for i in range(5) if health[i] > 0.1]
        if not candidates:
            return 7  # drop if everything is dead
        return int(min(candidates, key=lambda i: queue_sizes[i]))


class LatencyHeuristicAgent(BaselineAgent):
    """Routes to the service with lowest rolling latency, weighted by health.

    Observation layout:
      [5:10]  = health
      [10:15] = rolling latency
      [15:20] = failure rates
    """

    def select_action(self, obs: np.ndarray) -> int:
        health = obs[5:10]
        latency = obs[10:15]
        failure = obs[15:20]

        candidates = [i for i in range(5) if health[i] > 0.1]
        if not candidates:
            return 7  # drop

        # Score = latency * (1 + failure_rate), lower is better; penalize unhealthy
        def score(i):
            return latency[i] * (1.0 + failure[i] * 5.0) / max(health[i], 0.01)

        return int(min(candidates, key=score))


# Registry for easy access
BASELINES = {
    "round_robin": RoundRobinAgent,
    "least_connections": LeastConnectionsAgent,
    "latency_heuristic": LatencyHeuristicAgent,
}
