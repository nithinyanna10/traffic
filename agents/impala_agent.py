"""
IMPALA-style distributed training agent.

Architecture:
  - Central learner with policy network
  - N rollout workers collecting trajectories in parallel
  - V-trace importance-weighted corrections for off-policy data
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import env as _env_reg  # noqa: F401


# ---------------------------------------------------------------------------
# Policy Network
# ---------------------------------------------------------------------------

class IMPALAPolicy(nn.Module):
    """Actor-critic network for IMPALA."""

    def __init__(self, obs_dim: int = 34, act_dim: int = 9, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0)
            logits, value = self(x)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                action = torch.multinomial(probs, 1).item()
            log_prob = F.log_softmax(logits, dim=-1)[0, action].item()
        return action, log_prob, value.item()


# ---------------------------------------------------------------------------
# Trajectory storage
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A complete trajectory from a rollout worker."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)  # behavior policy log probs
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# V-trace
# ---------------------------------------------------------------------------

def vtrace_returns(
    behavior_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: float,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace targets and advantages."""
    T = len(rewards)
    rhos = torch.exp(target_log_probs - behavior_log_probs)
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    # Temporal differences
    values_plus = torch.cat([values, torch.tensor([bootstrap_value])])
    deltas = clipped_rhos * (rewards + gamma * values_plus[1:] - values_plus[:-1])

    # V-trace targets (backwards)
    vs_minus_v = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        vs_minus_v[t] = deltas[t] + gamma * cs[t] * last
        last = vs_minus_v[t]

    vs = values + vs_minus_v
    advantages = clipped_rhos * (rewards + gamma * vs_minus_v + gamma * values[0:] - values)
    # Simplified advantage: vs - values
    advantages = vs - values

    return vs, advantages


# ---------------------------------------------------------------------------
# Rollout Worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _worker_fn(
    worker_id: int,
    param_queue: mp.Queue,
    traj_queue: mp.Queue,
    env_id: str,
    rollout_length: int,
    stop_event: mp.Event,
):
    """Rollout worker: collects trajectories and sends to learner."""
    environment = gym.make(env_id)
    policy = IMPALAPolicy()

    obs, _ = environment.reset()
    while not stop_event.is_set():
        # Check for new parameters
        try:
            state_dict = param_queue.get_nowait()
            policy.load_state_dict(state_dict)
        except queue.Empty:
            pass

        traj = Trajectory()
        for _ in range(rollout_length):
            action, log_prob, value = policy.get_action(obs)
            next_obs, reward, terminated, truncated, info = environment.step(action)

            traj.observations.append(obs.copy())
            traj.actions.append(action)
            traj.rewards.append(reward)
            traj.log_probs.append(log_prob)
            traj.values.append(value)
            traj.dones.append(terminated or truncated)

            obs = next_obs
            if terminated or truncated:
                obs, _ = environment.reset()

        traj_queue.put(traj)

    environment.close()


# ---------------------------------------------------------------------------
# Central Learner
# ---------------------------------------------------------------------------

class IMPALAAgent:
    """IMPALA-style distributed RL agent."""

    def __init__(
        self,
        env_id: str = "MicroserviceRouting-v0",
        num_workers: int = 4,
        rollout_length: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 40.0,
        device: str = "cpu",
    ):
        self.env_id = env_id
        self.num_workers = num_workers
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.policy = IMPALAPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self._workers = []
        self._param_queues = []
        self._traj_queue = mp.Queue(maxsize=num_workers * 4)
        self._stop_event = mp.Event()

    def start_workers(self) -> None:
        """Spawn rollout workers."""
        for i in range(self.num_workers):
            pq = mp.Queue(maxsize=2)
            w = mp.Process(
                target=_worker_fn,
                args=(i, pq, self._traj_queue, self.env_id,
                      self.rollout_length, self._stop_event),
                daemon=True,
            )
            w.start()
            self._workers.append(w)
            self._param_queues.append(pq)
            # Send initial params
            pq.put(self.policy.state_dict())

    def stop_workers(self) -> None:
        """Stop all workers."""
        self._stop_event.set()
        for w in self._workers:
            w.join(timeout=5)
        self._workers.clear()

    def broadcast_params(self) -> None:
        """Send current policy params to all workers."""
        state_dict = self.policy.state_dict()
        for pq in self._param_queues:
            try:
                pq.put_nowait(state_dict)
            except queue.Full:
                pass  # worker hasn't consumed last update yet

    def train_step(self, traj: Trajectory) -> dict:
        """One learner update from a single trajectory."""
        obs_t = torch.FloatTensor(np.array(traj.observations))
        act_t = torch.LongTensor(traj.actions)
        rew_t = torch.FloatTensor(traj.rewards)
        behavior_lp = torch.FloatTensor(traj.log_probs)

        # Forward pass with current (target) policy
        logits, values = self.policy(obs_t)
        values = values.squeeze(-1)
        log_probs_all = F.log_softmax(logits, dim=-1)
        target_lp = log_probs_all.gather(1, act_t.unsqueeze(1)).squeeze(1)

        # V-trace
        bootstrap = values[-1].item()
        vs, advantages = vtrace_returns(
            behavior_lp.detach(),
            target_lp.detach(),
            rew_t,
            values.detach(),
            bootstrap,
            gamma=self.gamma,
        )

        # Policy gradient loss
        pg_loss = -(target_lp * advantages.detach()).mean()

        # Value loss
        v_loss = F.mse_loss(values, vs.detach())

        # Entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()

        loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
            "mean_reward": rew_t.mean().item(),
        }

    def train(
        self,
        num_updates: int = 1000,
        log_interval: int = 10,
        checkpoint_dir: str = "./checkpoints/impala",
        callback=None,
    ) -> list[dict]:
        """Full training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.start_workers()
        all_metrics = []

        try:
            for update in range(1, num_updates + 1):
                # Collect trajectory
                traj = self._traj_queue.get(timeout=60)
                metrics = self.train_step(traj)
                metrics["update"] = update
                all_metrics.append(metrics)

                # Broadcast updated params
                if update % 4 == 0:
                    self.broadcast_params()

                # Log
                if update % log_interval == 0:
                    print(
                        f"[IMPALA] Update {update}/{num_updates} | "
                        f"loss={metrics['total_loss']:.4f} | "
                        f"reward={metrics['mean_reward']:.4f} | "
                        f"entropy={metrics['entropy']:.4f}"
                    )

                # Checkpoint
                if update % 200 == 0:
                    path = os.path.join(checkpoint_dir, f"impala_{update}.pt")
                    torch.save(self.policy.state_dict(), path)

                if callback:
                    callback(metrics)

        finally:
            self.stop_workers()

        return all_metrics

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _, _ = self.policy.get_action(obs, deterministic=deterministic)
        return action

    def save(self, path: str = "./checkpoints/impala/impala_final.pt") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, weights_only=True))
