"""
Rare event replay buffer — prioritizes failure states and cascade events.

Stores transitions with elevated sampling probability for rare/important events
like service cascades, system collapses, and SLA breaches under high load.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Transition:
    """A single environment transition."""
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    info: dict
    priority: float = 1.0  # sampling priority


class RareEventReplayBuffer:
    """Prioritized replay buffer that over-samples rare failure events."""

    def __init__(
        self,
        capacity: int = 50_000,
        rare_event_capacity: int = 10_000,
        rare_priority_multiplier: float = 5.0,
    ):
        self._main_buffer: deque[Transition] = deque(maxlen=capacity)
        self._rare_buffer: deque[Transition] = deque(maxlen=rare_event_capacity)
        self._rare_priority = rare_priority_multiplier

    def add(self, transition: Transition) -> None:
        """Add a transition, automatically routing rare events."""
        if self._is_rare_event(transition):
            transition.priority = self._rare_priority
            self._rare_buffer.append(transition)
        self._main_buffer.append(transition)

    def _is_rare_event(self, t: Transition) -> bool:
        """Detect rare/important events worth prioritizing."""
        info = t.info
        # Cascade failure
        resp = info.get("response", {})
        if resp.get("cascade", False):
            return True
        # System collapse (episode terminated)
        if t.done and info.get("alive_services", 5) < 3:
            return True
        # Very large negative reward (catastrophic decision)
        if t.reward < -5.0:
            return True
        # Queue overflow
        if resp.get("queue_overflow", False):
            return True
        return False

    def sample(self, batch_size: int, rare_fraction: float = 0.3) -> list[Transition]:
        """
        Sample a batch with elevated proportion of rare events.

        Args:
            batch_size: Total number of transitions to sample.
            rare_fraction: Fraction of batch drawn from rare buffer.
        """
        n_rare = min(
            int(batch_size * rare_fraction),
            len(self._rare_buffer),
        )
        n_main = batch_size - n_rare

        samples = []
        if n_rare > 0:
            samples.extend(random.sample(list(self._rare_buffer), n_rare))
        if n_main > 0 and len(self._main_buffer) > 0:
            samples.extend(
                random.sample(
                    list(self._main_buffer),
                    min(n_main, len(self._main_buffer)),
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self._main_buffer)

    @property
    def rare_count(self) -> int:
        return len(self._rare_buffer)

    def get_stats(self) -> dict:
        return {
            "total_size": len(self._main_buffer),
            "rare_size": len(self._rare_buffer),
            "rare_fraction": len(self._rare_buffer) / max(len(self._main_buffer), 1),
        }
