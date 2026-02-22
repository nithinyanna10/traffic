"""
DQN Agent for MicroserviceRoutingEnv using Stable-Baselines3.

Useful for comparison with PPO and for environments where off-policy
sample efficiency matters. Supports same predict/save/load interface as PPOAgent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

import env  # noqa: F401


class DQNMetricsCallback(BaseCallback):
    """Logs env metrics to TensorBoard during DQN training."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "success_rate" in info:
                self.logger.record("env/success_rate", info["success_rate"])
                self.logger.record("env/avg_latency", info["avg_latency"])
                self.logger.record("env/avg_cost", info["avg_cost"])
        return True


class DQNAgent:
    """DQN agent for microservice routing (discrete action space)."""

    def __init__(
        self,
        env_id: str = "MicroserviceRouting-v0",
        learning_rate: float = 1e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.2,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        policy_kwargs: Optional[dict] = None,
        tensorboard_log: str = "./logs/dqn",
        device: str = "auto",
    ):
        import gymnasium as gym

        self.env_id = env_id
        self.env = gym.make(env_id)

        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[256, 256])

        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
            device=device,
        )

    def train(self, total_timesteps: int = 50_000, callback=None) -> None:
        callbacks = [DQNMetricsCallback()]
        if callback:
            callbacks.append(callback)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

    def predict(self, obs, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str = "./checkpoints/dqn_agent") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    def load(self, path: str = "./checkpoints/dqn_agent") -> None:
        self.model = DQN.load(path, env=self.env)

    @property
    def policy(self):
        return self.model.policy
