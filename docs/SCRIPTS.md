# Scripts Reference

All runnable scripts in the project and their arguments.

---

## CLI (cli.py)

Entry point: `python cli.py <command> [args]` or `traffic <command> [args]` after `pip install -e .`.

### train

Train an agent (PPO or DQN).

```bash
python cli.py train ppo [--steps 100000] [--lr 3e-4] [--checkpoint-dir ./checkpoints/ppo] [--log-dir ./logs/ppo] [--device auto] [--dashboard]
python cli.py train dqn [--steps 50000] [--lr 1e-4] [--checkpoint-dir ./checkpoints/dqn] [--log-dir ./logs/dqn] [--device auto]
```

- **ppo** — Uses `training/train_ppo.py`; `--dashboard` streams to dashboard if server is running.
- **dqn** — Uses `training/train_dqn.py`. Checkpoint and log dirs default to dqn subdirs when agent is dqn.

### compare

Run agent comparison (baselines + optional PPO checkpoint).

```bash
python cli.py compare [--episodes 50] [--ppo-checkpoint path] [--output ./logs/comparison.json]
```

### visualize

Plot training curves and comparison bar charts.

```bash
python cli.py visualize [--metrics ./logs/metrics/metrics.json] [--comparison ./logs/comparison.json] [--output-dir ./logs/plots]
```

### serve

Start the dashboard server.

```bash
python cli.py serve [--host 0.0.0.0] [--port 8000]
```

---

## Training scripts

### training/train_ppo.py

```bash
python training/train_ppo.py [--steps 100000] [--lr 3e-4] [--checkpoint-dir ./checkpoints/ppo] [--log-dir ./logs/ppo] [--dashboard] [--device auto]
```

- Uses CurriculumCallback to ramp traffic_rate, burst, and failure_injection.
- Saves checkpoint to `checkpoint_dir/ppo_final`.
- TensorBoard logs to `log_dir`.

### training/train_dqn.py

```bash
python training/train_dqn.py [--steps 50000] [--lr 1e-4] [--buffer-size 100000] [--checkpoint-dir ./checkpoints/dqn] [--log-dir ./logs/dqn] [--device auto]
```

- Saves checkpoint to `checkpoint_dir/dqn_final`.

---

## Analysis scripts

### analysis/compare.py

```bash
python analysis/compare.py [--episodes 50] [--ppo-checkpoint path] [--output ./logs/comparison.json]
```

- Evaluates each baseline and, if `--ppo-checkpoint` points to an existing `.zip`, PPO.
- Writes JSON and an HTML report (same path with `.json` replaced by `.html`).

### analysis/visualize.py

```bash
python analysis/visualize.py [--metrics ./logs/metrics/metrics.json] [--comparison ./logs/comparison.json] [--output-dir ./logs/plots]
```

- Reads metrics (list or `{ "metrics": [...] }`) and comparison JSON.
- Writes `training_curves.png` and `comparison.png` into `output-dir`.

---

## Utility scripts

### scripts/sweep_ppo.py

Hyperparameter sweep over learning rate and steps.

```bash
python scripts/sweep_ppo.py [--lrs 1e-4 3e-4 1e-3] [--steps 20000 50000] [--base-dir ./logs] [--output ./logs/sweep_results.json]
```

- Runs `train_ppo.py` for each (steps, lr) combination; records success and elapsed time.
- Writes results to `output` and prints a summary table.

### scripts/run_experiment.py

Full pipeline: train → compare → visualize → report.

```bash
python scripts/run_experiment.py [--agent ppo] [--train-steps 30000] [--compare-episodes 30] [--output-dir ./logs/experiments] [--skip-train] [--skip-compare] [--skip-visualize]
```

- Creates a timestamped directory under `output-dir` with report.json, comparison.json, plots, and index.html.

### scripts/benchmark.py

Benchmark one agent over multiple seeds.

```bash
python scripts/benchmark.py [--seeds 0 1 2 3 4] [--episodes 20] [--agent round_robin|least_connections|latency_heuristic|ppo] [--checkpoint path]
```

- For PPO, use `--checkpoint ./checkpoints/ppo/ppo_final` (without .zip).

### scripts/eval_all_agents.py

Evaluate all baselines and optional PPO/DQN over multiple seeds; print table.

```bash
python scripts/eval_all_agents.py [--episodes 20] [--seeds 0 1 2] [--ppo path] [--dqn path]
```

---

## Examples

### examples/run_baseline.py

```bash
python examples/run_baseline.py [--agent round_robin] [--steps 500] [--seed 0]
```

### examples/run_with_curriculum.py

Runs env with curriculum (random actions); no RL. No arguments.

### examples/export_metrics_from_dashboard.py

```bash
python examples/export_metrics_from_dashboard.py [--url http://localhost:8000] [--output dashboard_metrics.json] [--last 1000]
```

- Requires dashboard server running.

### examples/full_pipeline.py

```bash
python examples/full_pipeline.py [--train-steps 5000] [--eval-episodes 10] [--eval-seeds 3] [--out-dir ./logs/pipeline_demo]
```

- In-process train (short demo loop) + evaluate PPO and baselines; prints table. For full training use `train_ppo.py`.
