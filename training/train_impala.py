#!/usr/bin/env python3
"""
Train IMPALA agent on MicroserviceRoutingEnv with distributed rollout workers.

Usage:
    python training/train_impala.py [--updates N] [--workers W]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.impala_agent import IMPALAAgent
from training.curriculum import CurriculumScheduler
from training.metrics import MetricsTracker


def main():
    parser = argparse.ArgumentParser(description="Train IMPALA on MicroserviceRoutingEnv")
    parser.add_argument("--updates", type=int, default=1000, help="Number of learner updates")
    parser.add_argument("--workers", type=int, default=4, help="Number of rollout workers")
    parser.add_argument("--rollout-length", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/impala")
    parser.add_argument("--log-dir", type=str, default="./logs/impala")
    args = parser.parse_args()

    print("=" * 60)
    print("RL-Traffic-Control: IMPALA Training")
    print("=" * 60)
    print(f"  Learner updates:  {args.updates:,}")
    print(f"  Workers:          {args.workers}")
    print(f"  Rollout length:   {args.rollout_length}")
    print(f"  Learning rate:    {args.lr}")
    print("=" * 60)

    # Metrics
    metrics_tracker = MetricsTracker(log_dir=args.log_dir)

    # Agent
    agent = IMPALAAgent(
        num_workers=args.workers,
        rollout_length=args.rollout_length,
        lr=args.lr,
    )

    start_time = time.time()

    def on_update(metrics: dict) -> None:
        update = metrics.get("update", 0)
        if update % 50 == 0:
            stats = metrics_tracker.get_rolling_stats()
            if stats:
                print(
                    f"  [Rolling] reward={stats.get('mean_reward', 0):.2f} "
                    f"uptime={stats.get('mean_uptime', 0):.2%}"
                )

    all_metrics = agent.train(
        num_updates=args.updates,
        log_interval=20,
        checkpoint_dir=args.checkpoint_dir,
        callback=on_update,
    )

    elapsed = time.time() - start_time

    # Save final
    agent.save(os.path.join(args.checkpoint_dir, "impala_final.pt"))
    metrics_tracker.save()
    metrics_tracker.close()

    print(f"\n{'=' * 60}")
    print(f"IMPALA training complete in {elapsed:.1f}s ({args.updates} updates)")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
