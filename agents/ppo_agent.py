"""
PPO Agent wrapping Stable-Baselines3 for the MicroserviceRoutingEnv.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import env  # noqa: F401  — triggers gymnasium registration


class MetricsCallback(BaseCallback):
    """Logs custom env metrics to TensorBoard during PPO training."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "success_rate" in info:
                self.logger.record("env/success_rate", info["success_rate"])
                self.logger.record("env/avg_latency", info["avg_latency"])
                self.logger.record("env/avg_cost", info["avg_cost"])
                self.logger.record("env/alive_services", info["alive_services"])
        return True


class PPOAgent:
    """PPO agent for microservice routing."""

    def __init__(
        self,
        env_id: str = "MicroserviceRouting-v0",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        policy_kwargs: Optional[dict] = None,
        tensorboard_log: str = "./logs/ppo",
        device: str = "auto",
    ):
        import gymnasium as gym

        self.env = gym.make(env_id)

        if policy_kwargs is None:
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            )

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
            device=device,
        )

    def train(self, total_timesteps: int = 100_000, callback=None) -> None:
        callbacks = [MetricsCallback()]
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

    def save(self, path: str = "./checkpoints/ppo_agent") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str = "./checkpoints/ppo_agent") -> None:
        self.model = PPO.load(path, env=self.env)

    @property
    def policy(self):
        return self.model.policy
