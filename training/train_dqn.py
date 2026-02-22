#!/usr/bin/env python3
"""
Train DQN agent on MicroserviceRoutingEnv.

Usage:
    python training/train_dqn.py [--steps N] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env as _env_reg  # noqa: F401
from agents.dqn_agent import DQNAgent


def main():
    parser = argparse.ArgumentParser(description="Train DQN on MicroserviceRoutingEnv")
    parser.add_argument("--steps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/dqn")
    parser.add_argument("--log-dir", type=str, default="./logs/dqn")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("=" * 60)
    print("RL-Traffic-Control: DQN Training")
    print("=" * 60)
    print(f"  Total timesteps: {args.steps:,}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Checkpoint dir:  {args.checkpoint_dir}")
    print(f"  Log dir:         {args.log_dir}")
    print("=" * 60)

    agent = DQNAgent(
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        tensorboard_log=args.log_dir,
        device=args.device,
    )

    start_time = time.time()
    agent.train(total_timesteps=args.steps)
    elapsed = time.time() - start_time

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    agent.save(os.path.join(args.checkpoint_dir, "dqn_final"))

    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Checkpoint saved to {args.checkpoint_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
