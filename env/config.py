"""
Environment configuration: optional overrides for MicroserviceRoutingEnv.
Used by env factory and CLI to create envs with custom max_steps, traffic_rate, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Configuration for MicroserviceRoutingEnv (and variants)."""

    traffic_rate: float = 1.0
    max_steps: int = 1000
    seed: Optional[int] = None
    # Curriculum-related (can be overridden by curriculum scheduler)
    burst_probability: float = 0.0
    burst_size: int = 3
    failure_injection_prob: float = 0.0

    def to_dict(self) -> dict:
        return {
            "traffic_rate": self.traffic_rate,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "burst_probability": self.burst_probability,
            "burst_size": self.burst_size,
            "failure_injection_prob": self.failure_injection_prob,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        return cls(
            traffic_rate=d.get("traffic_rate", 1.0),
            max_steps=d.get("max_steps", 1000),
            seed=d.get("seed"),
            burst_probability=d.get("burst_probability", 0.0),
            burst_size=d.get("burst_size", 3),
            failure_injection_prob=d.get("failure_injection_prob", 0.0),
        )


# Default config used when no config is passed
DEFAULT_ENV_CONFIG = EnvConfig()
