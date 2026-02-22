#!/usr/bin/env python3
"""
End-to-end experiment runner: train → compare → visualize → HTML report.

Runs PPO (or DQN) for a given number of steps, then runs comparison with baselines
and optional PPO checkpoint, generates plots, and writes a single experiment report.

Usage:
    python scripts/run_experiment.py --agent ppo --train-steps 30000 --compare-episodes 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_cmd(cmd: list[str], cwd: str, timeout: int = 7200) -> tuple[bool, str]:
    """Run command; return (success, combined_output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Run full experiment: train, compare, visualize")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--train-steps", type=int, default=30_000)
    parser.add_argument("--compare-episodes", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="./logs/experiments")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing checkpoint")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"{args.agent}_{stamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "agent": args.agent,
        "train_steps": args.train_steps,
        "compare_episodes": args.compare_episodes,
        "timestamp": stamp,
        "steps": [],
        "results": {},
    }

    # 1. Train
    if not args.skip_train:
        ckpt_dir = root / "checkpoints" / args.agent
        log_dir = root / "logs" / args.agent
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        train_script = "training/train_ppo.py" if args.agent == "ppo" else "training/train_dqn.py"
        cmd = [
            sys.executable,
            train_script,
            "--steps", str(args.train_steps),
            "--checkpoint-dir", str(ckpt_dir),
            "--log-dir", str(log_dir),
        ]
        report["steps"].append({"name": "train", "cmd": " ".join(cmd)})
        t0 = time.time()
        ok, out = run_cmd(cmd, str(root))
        report["steps"][-1]["success"] = ok
        report["steps"][-1]["elapsed_s"] = round(time.time() - t0, 1)
        report["steps"][-1]["output_tail"] = out[-2000:] if out else ""
        if not ok:
            (exp_dir / "train_log.txt").write_text(out or "")
            print("Train failed. See", exp_dir / "train_log.txt")
            json.dump(report, open(exp_dir / "report.json", "w"), indent=2)
            sys.exit(1)
        report["checkpoint_dir"] = str(ckpt_dir)
    else:
        report["checkpoint_dir"] = str(root / "checkpoints" / args.agent)

    # 2. Compare
    comparison_path = exp_dir / "comparison.json"
    if not args.skip_compare:
        ckpt = str(Path(report["checkpoint_dir"]) / ("ppo_final" if args.agent == "ppo" else "dqn_final"))
        cmd = [
            sys.executable,
            "analysis/compare.py",
            "--episodes", str(args.compare_episodes),
            "--output", str(comparison_path),
        ]
        if Path(ckpt + ".zip").exists():
            cmd.extend(["--ppo-checkpoint", ckpt])
        report["steps"].append({"name": "compare", "cmd": " ".join(cmd)})
        t0 = time.time()
        ok, out = run_cmd(cmd, str(root))
        report["steps"][-1]["success"] = ok
        report["steps"][-1]["elapsed_s"] = round(time.time() - t0, 1)
        if ok and comparison_path.exists():
            report["results"]["comparison"] = json.loads(comparison_path.read_text())

    # 3. Visualize
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_visualize:
        metrics_path = root / "logs" / args.agent
        # Try to find a metrics file; visualize.py can use comparison only
        cmd = [
            sys.executable,
            "analysis/visualize.py",
            "--metrics", str(metrics_path / "metrics.json"),
            "--comparison", str(comparison_path),
            "--output-dir", str(plots_dir),
        ]
        report["steps"].append({"name": "visualize", "cmd": " ".join(cmd)})
        t0 = time.time()
        ok, out = run_cmd(cmd, str(root))
        report["steps"][-1]["success"] = ok
        report["steps"][-1]["elapsed_s"] = round(time.time() - t0, 1)

    # 4. Write report
    report_path = exp_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\nExperiment report:", report_path)
    print("Output dir:", exp_dir)

    # 5. HTML summary
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Experiment {stamp}</title>
<style>body{{font-family:system-ui;background:#0c0f18;color:#e2e8f4;padding:24px;}}
h1{{color:#00d4ff;}} table{{border-collapse:collapse;}} th,td{{border:1px solid #2a3a55;padding:8px 12px;text-align:left;}}
.success{{color:#00ff88;}} .fail{{color:#ff4757;}}</style></head><body>
<h1>Experiment: {args.agent} ({stamp})</h1>
<p>Train steps: {args.train_steps} | Compare episodes: {args.compare_episodes}</p>
<table><tr><th>Step</th><th>Success</th><th>Time (s)</th></tr>"""
    for s in report["steps"]:
        cls = "success" if s.get("success", False) else "fail"
        html += f"<tr><td>{s['name']}</td><td class='{cls}'>{s.get('success', False)}</td><td>{s.get('elapsed_s', '-')}</td></tr>"
    html += "</table>"
    if report.get("results", {}).get("comparison"):
        html += "<h2>Comparison</h2><pre>" + json.dumps(report["results"]["comparison"], indent=2) + "</pre>"
    html += "</body></html>"
    (exp_dir / "index.html").write_text(html)
    print("Summary HTML:", exp_dir / "index.html")


if __name__ == "__main__":
    main()
