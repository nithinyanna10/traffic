#!/usr/bin/env python3
"""
Example: fetch /api/metrics from a running dashboard and save to JSON.
Useful for feeding visualize.py or archiving a run.

Usage:
    python examples/export_metrics_from_dashboard.py [--url http://localhost:8000] [--output metrics.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--output", type=str, default="dashboard_metrics.json")
    parser.add_argument("--last", type=int, default=1000)
    args = parser.parse_args()

    u = f"{args.url.rstrip('/')}/api/metrics?last={args.last}"
    try:
        with urllib.request.urlopen(u, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        print("Error fetching metrics:", e)
        sys.exit(1)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved", len(data.get("metrics", [])), "episodes to", args.output)


if __name__ == "__main__":
    main()
