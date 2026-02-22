"""RL-Traffic-Control: Custom Gymnasium environment for microservice routing."""

import gymnasium as gym

from env.microservice_env import MicroserviceRoutingEnv
from env.microservice_env_hard import MicroserviceRoutingHardEnv

gym.register(
    id="MicroserviceRouting-v0",
    entry_point="env.microservice_env:MicroserviceRoutingEnv",
    max_episode_steps=1000,
)

gym.register(
    id="MicroserviceRoutingHard-v0",
    entry_point="env.microservice_env_hard:MicroserviceRoutingHardEnv",
    max_episode_steps=2000,
)

__all__ = ["MicroserviceRoutingEnv", "MicroserviceRoutingHardEnv"]
