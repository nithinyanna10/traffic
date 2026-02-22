"""
Gymnasium wrappers for MicroserviceRoutingEnv.

- NormalizeObservation: optionally re-normalize obs (env already in [0,1])
- RecordEpisodeStatistics: record episode reward/length for logging
- TimeLimitOverride: override max_episode_steps at wrap time
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from gymnasium.wrappers import RecordEpisodeStatistics as _RecordEpisodeStatistics


class NormalizeObservation(Wrapper):
    """Normalize observation to zero mean, unit variance (or min-max to [0,1])."""

    def __init__(self, env: gym.Env, mode: str = "minmax"):
        super().__init__(env)
        self.mode = mode
        self._obs_mean = None
        self._obs_std = None
        self._obs_min = None
        self._obs_max = None
        self._n = 0

    def _update_stats(self, obs: np.ndarray) -> None:
        self._n += 1
        if self._obs_mean is None:
            self._obs_mean = np.zeros_like(obs)
            self._obs_std = np.ones_like(obs)
            self._obs_min = obs.copy()
            self._obs_max = obs.copy()
        else:
            alpha = 1.0 / self._n
            self._obs_mean = (1 - alpha) * self._obs_mean + alpha * obs
            self._obs_std = np.maximum(self._obs_std * 0.99, np.abs(obs - self._obs_mean) + 1e-8)
            self._obs_min = np.minimum(self._obs_min, obs)
            self._obs_max = np.maximum(self._obs_max, obs)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update_stats(obs)
        if self.mode == "minmax":
            r = self._obs_max - self._obs_min
            r = np.where(r < 1e-8, 1.0, r)
            return (obs - self._obs_min) / r
        else:
            return (obs - self._obs_mean) / (self._obs_std + 1e-8)


class RecordEpisodeStatistics(_RecordEpisodeStatistics):
    """Thin wrapper to expose episode_reward and episode_length in info."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.is_vector_env:
            # Vector env: info is a list
            pass
        return obs, reward, terminated, truncated, info


class TimeLimitOverride(Wrapper):
    """Override max_episode_steps for the wrapped env (if it supports it)."""

    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        return obs, reward, terminated, truncated, info
