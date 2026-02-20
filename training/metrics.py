"""
Metric tracking and logging for training runs.

Tracks:
  - success_rate
  - p95_latency
  - cost_per_request
  - system_uptime (fraction of steps all services alive)
  - collapse_frequency (collapses per episode)
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int = 0
    total_reward: float = 0.0
    steps: int = 0
    success_rate: float = 0.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    avg_cost: float = 0.0
    system_uptime: float = 0.0
    collapsed: bool = False
    timestamp: float = 0.0


class MetricsTracker:
    """Tracks and logs training metrics."""

    def __init__(
        self,
        log_dir: str = "./logs/metrics",
        window_size: int = 100,
        tensorboard: bool = True,
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._window_size = window_size
        self._episodes: list[EpisodeMetrics] = []
        self._recent: deque[EpisodeMetrics] = deque(maxlen=window_size)

        # Per-step latency tracking within an episode
        self._step_latencies: list[float] = []
        self._step_costs: list[float] = []
        self._step_successes: list[bool] = []
        self._alive_steps = 0
        self._total_steps = 0
        self._episode_reward = 0.0
        self._collapse_count = 0

        # TensorBoard
        self._writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("[MetricsTracker] TensorBoard not available, skipping.")

    def on_step(self, reward: float, info: dict) -> None:
        """Called every environment step."""
        self._episode_reward += reward
        self._total_steps += 1

        resp = info.get("response", {})
        if resp:
            self._step_latencies.append(resp.get("latency_ms", 0))
            self._step_costs.append(resp.get("cost", 0))
            self._step_successes.append(resp.get("success", False))

        if info.get("alive_services", 5) == 5:
            self._alive_steps += 1

    def on_episode_end(self, episode: int, collapsed: bool = False) -> EpisodeMetrics:
        """Called at the end of each episode. Returns episode metrics."""
        if collapsed:
            self._collapse_count += 1

        latencies = self._step_latencies or [0.0]
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)

        metrics = EpisodeMetrics(
            episode=episode,
            total_reward=self._episode_reward,
            steps=self._total_steps,
            success_rate=(
                sum(self._step_successes) / max(len(self._step_successes), 1)
            ),
            avg_latency=sum(latencies) / max(len(latencies), 1),
            p95_latency=sorted_lat[min(p95_idx, len(sorted_lat) - 1)],
            avg_cost=sum(self._step_costs) / max(len(self._step_costs), 1),
            system_uptime=self._alive_steps / max(self._total_steps, 1),
            collapsed=collapsed,
            timestamp=time.time(),
        )

        self._episodes.append(metrics)
        self._recent.append(metrics)

        # Log to TensorBoard
        if self._writer:
            self._writer.add_scalar("episode/reward", metrics.total_reward, episode)
            self._writer.add_scalar("episode/success_rate", metrics.success_rate, episode)
            self._writer.add_scalar("episode/avg_latency", metrics.avg_latency, episode)
            self._writer.add_scalar("episode/p95_latency", metrics.p95_latency, episode)
            self._writer.add_scalar("episode/avg_cost", metrics.avg_cost, episode)
            self._writer.add_scalar("episode/system_uptime", metrics.system_uptime, episode)
            self._writer.add_scalar(
                "episode/collapse_rate",
                self._collapse_count / max(episode, 1),
                episode,
            )
            self._writer.flush()

        # Reset per-episode trackers
        self._step_latencies = []
        self._step_costs = []
        self._step_successes = []
        self._alive_steps = 0
        self._total_steps = 0
        self._episode_reward = 0.0

        return metrics

    def get_rolling_stats(self) -> dict:
        """Get rolling statistics over recent episodes."""
        if not self._recent:
            return {}
        return {
            "mean_reward": sum(m.total_reward for m in self._recent) / len(self._recent),
            "mean_success_rate": sum(m.success_rate for m in self._recent) / len(self._recent),
            "mean_p95_latency": sum(m.p95_latency for m in self._recent) / len(self._recent),
            "mean_cost": sum(m.avg_cost for m in self._recent) / len(self._recent),
            "mean_uptime": sum(m.system_uptime for m in self._recent) / len(self._recent),
            "collapse_rate": sum(1 for m in self._recent if m.collapsed) / len(self._recent),
            "episodes": len(self._recent),
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save all episode metrics to JSON."""
        path = path or os.path.join(self.log_dir, "metrics.json")
        data = []
        for m in self._episodes:
            data.append({
                "episode": m.episode,
                "total_reward": m.total_reward,
                "steps": m.steps,
                "success_rate": m.success_rate,
                "avg_latency": m.avg_latency,
                "p95_latency": m.p95_latency,
                "avg_cost": m.avg_cost,
                "system_uptime": m.system_uptime,
                "collapsed": m.collapsed,
                "timestamp": m.timestamp,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def close(self) -> None:
        if self._writer:
            self._writer.close()
