"""
MicroserviceRoutingHardEnv — Harder variant: longer episodes, higher default traffic.

Same observation/action as MicroserviceRouting-v0 but MAX_STEPS=2000 and
default traffic_rate=1.5. Use for curriculum or evaluation at scale.
"""

from __future__ import annotations

from typing import Optional

from env.microservice_env import MicroserviceRoutingEnv


class MicroserviceRoutingHardEnv(MicroserviceRoutingEnv):
    """Harder routing env: 2000 steps per episode, default traffic 1.5."""

    MAX_STEPS = 2000

    def __init__(
        self,
        traffic_rate: float = 1.5,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__(traffic_rate=traffic_rate, seed=seed, render_mode=render_mode)
