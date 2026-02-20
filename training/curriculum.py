"""
Curriculum scheduler — gradually increases difficulty over training.

Difficulty knobs:
  - traffic_rate: requests per step (1.0 → 3.0+)
  - failure_injection: probability of forcing service crashes
  - burst_probability: chance of traffic bursts
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    """Curriculum parameters at a given stage."""
    traffic_rate: float = 1.0
    failure_injection_prob: float = 0.0
    burst_probability: float = 0.0
    burst_size: int = 3


class CurriculumScheduler:
    """Linearly interpolates difficulty parameters over training progress."""

    def __init__(
        self,
        start: CurriculumConfig | None = None,
        end: CurriculumConfig | None = None,
        warmup_fraction: float = 0.1,
    ):
        self.start = start or CurriculumConfig(
            traffic_rate=0.6,
            failure_injection_prob=0.0,
            burst_probability=0.0,
            burst_size=2,
        )
        self.end = end or CurriculumConfig(
            traffic_rate=3.0,
            failure_injection_prob=0.05,
            burst_probability=0.15,
            burst_size=8,
        )
        self.warmup_fraction = warmup_fraction

    def get_config(self, progress: float) -> CurriculumConfig:
        """
        Get curriculum config at given training progress.

        Args:
            progress: float in [0, 1], fraction of total training completed.

        Returns:
            Interpolated CurriculumConfig.
        """
        # During warmup, stay at start config
        if progress < self.warmup_fraction:
            return CurriculumConfig(**self.start.__dict__)

        # Linear interpolation after warmup
        t = (progress - self.warmup_fraction) / (1.0 - self.warmup_fraction)
        t = min(1.0, max(0.0, t))

        return CurriculumConfig(
            traffic_rate=self._lerp(self.start.traffic_rate, self.end.traffic_rate, t),
            failure_injection_prob=self._lerp(
                self.start.failure_injection_prob, self.end.failure_injection_prob, t
            ),
            burst_probability=self._lerp(
                self.start.burst_probability, self.end.burst_probability, t
            ),
            burst_size=int(self._lerp(self.start.burst_size, self.end.burst_size, t)),
        )

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t
