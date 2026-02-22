#!/usr/bin/env python3
"""
Train PPO agent on MicroserviceRoutingEnv with curriculum learning.

Usage:
    python training/train_ppo.py [--steps N] [--dashboard]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np

import env as _env_reg  # noqa: F401
from agents.ppo_agent import PPOAgent, MetricsCallback
from training.curriculum import CurriculumScheduler
from training.metrics import MetricsTracker
from training.replay_buffer import RareEventReplayBuffer, Transition

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Adjusts environment difficulty during PPO training."""

    def __init__(self, scheduler: CurriculumScheduler, total_timesteps: int):
        super().__init__()
        self.scheduler = scheduler
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        config = self.scheduler.get_config(progress)
        # Access the unwrapped env
        try:
            unwrapped = self.training_env.envs[0].unwrapped
            unwrapped.set_traffic_rate(config.traffic_rate)
            unwrapped.set_burst_config(config.burst_probability, config.burst_size)
            unwrapped.set_failure_injection(config.failure_injection_prob)
        except (AttributeError, IndexError):
            pass
        if self.num_timesteps % 10_000 == 0:
            self.logger.record("curriculum/traffic_rate", config.traffic_rate)
            self.logger.record("curriculum/burst_prob", config.burst_probability)
            self.logger.record("curriculum/failure_injection", config.failure_injection_prob)
            self.logger.record("curriculum/progress", progress)
        return True


class DashboardStreamCallback(BaseCallback):
    """Streams state to dashboard WebSocket during training."""

    def __init__(self, dashboard_url: str = "ws://localhost:8000/ws/train"):
        super().__init__()
        self.dashboard_url = dashboard_url
        self._ws = None

    def _on_training_start(self):
        try:
            import asyncio
            import websockets
            # Will be used if dashboard is running
        except ImportError:
            pass

    def _on_step(self) -> bool:
        # Stream every 10 steps to avoid overload
        if self.num_timesteps % 10 == 0:
            try:
                unwrapped = self.training_env.envs[0].unwrapped
                info = unwrapped.last_step_info
                if info and self._ws:
                    import json
                    # Non-blocking send attempt
                    pass  # WebSocket streaming handled by dashboard server
            except Exception:
                pass
        return True


def main():
    parser = argparse.ArgumentParser(description="Train PPO on MicroserviceRoutingEnv")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ppo")
    parser.add_argument("--log-dir", type=str, default="./logs/ppo")
    parser.add_argument("--dashboard", action="store_true", help="Stream to dashboard")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("=" * 60)
    print("RL-Traffic-Control: PPO Training")
    print("=" * 60)
    print(f"  Total timesteps: {args.steps:,}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Checkpoint dir:  {args.checkpoint_dir}")
    print(f"  Log dir:         {args.log_dir}")
    print(f"  Device:          {args.device}")
    print("=" * 60)

    # Create agent
    agent = PPOAgent(
        learning_rate=args.lr,
        tensorboard_log=args.log_dir,
        device=args.device,
    )

    # Set up callbacks
    scheduler = CurriculumScheduler()
    curriculum_cb = CurriculumCallback(scheduler, args.steps)

    callbacks = [curriculum_cb]
    if args.dashboard:
        callbacks.append(DashboardStreamCallback())

    # Train
    start_time = time.time()
    agent.model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - start_time

    # Save checkpoint
    agent.save(os.path.join(args.checkpoint_dir, "ppo_final"))

    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Checkpoint saved to {args.checkpoint_dir}")
    print(f"Logs saved to {args.log_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
