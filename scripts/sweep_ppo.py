#!/usr/bin/env python3
"""
Hyperparameter sweep for PPO: run multiple trainings with different lr and steps.
Results are written to logs/sweep_results.json and a summary table is printed.

Usage:
    python scripts/sweep_ppo.py --lrs 1e-4 3e-4 1e-3 --steps 20000 50000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_ppo(steps: int, lr: float, run_id: str, base_dir: str) -> dict:
    """Run train_ppo.py and return final metrics from TensorBoard or last checkpoint."""
    log_dir = os.path.join(base_dir, "sweep_logs", run_id)
    ckpt_dir = os.path.join(base_dir, "sweep_ckpts", run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "training/train_ppo.py",
        "--steps", str(steps),
        "--lr", str(lr),
        "--log-dir", log_dir,
        "--checkpoint-dir", ckpt_dir,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    return {
        "run_id": run_id,
        "steps": steps,
        "lr": lr,
        "success": result.returncode == 0,
        "elapsed_s": round(elapsed, 1),
        "stderr": result.stderr[-500:] if result.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep PPO hyperparameters")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--steps", type=int, nargs="+", default=[20_000, 50_000])
    parser.add_argument("--base-dir", type=str, default="./logs")
    parser.add_argument("--output", type=str, default="./logs/sweep_results.json")
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    results = []
    run_idx = 0

    for steps in args.steps:
        for lr in args.lrs:
            run_id = f"ppo_steps{steps}_lr{lr}_run{run_idx}"
            run_idx += 1
            print(f"\n>>> Running {run_id} ...")
            r = run_ppo(steps, lr, run_id, args.base_dir)
            results.append(r)
            print(f"    success={r['success']} elapsed={r['elapsed_s']}s")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 60)
    print(f"{'Run ID':<40} {'Success':<8} {'Time (s)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['run_id']:<40} {str(r['success']):<8} {r['elapsed_s']:<10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
