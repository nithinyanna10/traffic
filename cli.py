#!/usr/bin/env python3
"""
RL-Traffic-Control CLI: train, compare, visualize, serve.

Usage:
    python cli.py train ppo --steps 100000
    python cli.py train dqn --steps 50000
    python cli.py compare --episodes 50 --ppo-checkpoint ./checkpoints/ppo/ppo_final
    python cli.py visualize --metrics ./logs/metrics.json --output-dir ./logs/plots
    python cli.py serve [--port 8000]
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def cmd_train(args: argparse.Namespace) -> None:
    ckpt_dir = args.checkpoint_dir if args.agent == "ppo" else args.checkpoint_dir.replace("ppo", "dqn", 1)
    log_dir = args.log_dir if args.agent == "ppo" else args.log_dir.replace("ppo", "dqn", 1)
    if args.agent == "ppo":
        from training.train_ppo import main as ppo_main
        sys.argv = [
            "train_ppo.py",
            "--steps", str(args.steps),
            "--lr", str(args.lr),
            "--checkpoint-dir", ckpt_dir,
            "--log-dir", log_dir,
            "--device", args.device,
        ]
        if args.dashboard:
            sys.argv.append("--dashboard")
        ppo_main()
    elif args.agent == "dqn":
        from training.train_dqn import main as dqn_main
        sys.argv = [
            "train_dqn.py",
            "--steps", str(args.steps),
            "--lr", str(args.lr),
            "--checkpoint-dir", ckpt_dir,
            "--log-dir", log_dir,
            "--device", args.device,
        ]
        dqn_main()
    else:
        print(f"Unknown agent: {args.agent}. Use ppo or dqn.")
        sys.exit(1)


def cmd_compare(args: argparse.Namespace) -> None:
    from analysis.compare import main as compare_main
    sys.argv = [
        "compare.py",
        "--episodes", str(args.episodes),
        "--output", args.output,
    ]
    if args.ppo_checkpoint:
        sys.argv.extend(["--ppo-checkpoint", args.ppo_checkpoint])
    compare_main()


def cmd_visualize(args: argparse.Namespace) -> None:
    from analysis.visualize import main as viz_main
    sys.argv = [
        "visualize.py",
        "--metrics", args.metrics,
        "--comparison", args.comparison,
        "--output-dir", args.output_dir,
    ]
    viz_main()


def cmd_serve(args: argparse.Namespace) -> None:
    from dashboard.server import start_server
    start_server(host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="traffic",
        description="RL-Traffic-Control: train, compare, visualize, serve dashboard.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train an agent (ppo or dqn)")
    p_train.add_argument("agent", choices=["ppo", "dqn"], help="Agent type")
    p_train.add_argument("--steps", type=int, default=100_000, help="Timesteps (ppo default 100k, dqn 50k)")
    p_train.add_argument("--lr", type=float, default=None, help="Learning rate (default per agent)")
    p_train.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ppo")
    p_train.add_argument("--log-dir", type=str, default="./logs/ppo")
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--dashboard", action="store_true", help="(PPO only) Stream to dashboard")
    p_train.set_defaults(lr_ppo=3e-4, lr_dqn=1e-4)
    p_train.set_defaults(func=cmd_train)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare agents (baselines + optional PPO)")
    p_compare.add_argument("--episodes", type=int, default=50)
    p_compare.add_argument("--ppo-checkpoint", type=str, default=None)
    p_compare.add_argument("--output", type=str, default="./logs/comparison.json")
    p_compare.set_defaults(func=cmd_compare)

    # visualize
    p_viz = subparsers.add_parser("visualize", help="Plot training curves and comparison")
    p_viz.add_argument("--metrics", type=str, default="./logs/metrics/metrics.json")
    p_viz.add_argument("--comparison", type=str, default="./logs/comparison.json")
    p_viz.add_argument("--output-dir", type=str, default="./logs/plots")
    p_viz.set_defaults(func=cmd_visualize)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start dashboard server")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if args.command == "train":
        if args.lr is None:
            args.lr = args.lr_ppo if args.agent == "ppo" else args.lr_dqn
    args.func(args)


if __name__ == "__main__":
    main()
