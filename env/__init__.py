"""RL-Traffic-Control: Custom Gymnasium environment for microservice routing."""

import gymnasium as gym

from env.microservice_env import MicroserviceRoutingEnv

gym.register(
    id="MicroserviceRouting-v0",
    entry_point="env.microservice_env:MicroserviceRoutingEnv",
    max_episode_steps=1000,
)

__all__ = ["MicroserviceRoutingEnv"]
