# RL-Traffic-Control Runbook

Quick reference for running training, evaluation, and the dashboard.

## One-time setup

```bash
cd traffic
pip install -e .
# Optional: pip install -e ".[dev]"  # for pytest
```

## Training

### PPO (recommended)

```bash
# Default: 100k steps, lr 3e-4
python training/train_ppo.py

# Custom steps and learning rate
python training/train_ppo.py --steps 50000 --lr 1e-4

# With TensorBoard
tensorboard --logdir ./logs/ppo
# Then run training in another terminal
```

### DQN

```bash
python training/train_dqn.py --steps 50000 --lr 1e-4
tensorboard --logdir ./logs/dqn
```

### Using the CLI

```bash
python cli.py train ppo --steps 100000
python cli.py train dqn --steps 50000 --checkpoint-dir ./checkpoints/dqn
```

## Dashboard

```bash
# Start server (port 8000)
python dashboard/server.py
# Or: python cli.py serve --port 8000

# Open in browser: http://localhost:8000
# - Set Steps, Curriculum, optional Checkpoint
# - Click "Train PPO" or run a baseline
# - Fullscreen cluster map, Export PNG for charts
# - Metrics page: http://localhost:8000/metrics
```

## Comparison and visualization

```bash
# Compare baselines + PPO (if checkpoint exists)
python analysis/compare.py --episodes 50 --ppo-checkpoint ./checkpoints/ppo/ppo_final --output ./logs/comparison.json
# Produces comparison.json and comparison.html

# Plot training curves (requires metrics JSON from a run)
python analysis/visualize.py --metrics ./logs/ppo/metrics.json --comparison ./logs/comparison.json --output-dir ./logs/plots
```

## Docker

```bash
docker-compose up --build
# Dashboard: http://localhost:8000
# TensorBoard: http://localhost:6006 (if you mount logs)
```

## Sweep and experiments

```bash
# Hyperparameter sweep (PPO lr and steps)
python scripts/sweep_ppo.py --lrs 1e-4 3e-4 --steps 20000 50000

# Full experiment: train → compare → visualize → report
python scripts/run_experiment.py --agent ppo --train-steps 30000 --compare-episodes 30
# Output in ./logs/experiments/ppo_<timestamp>/
```

## Benchmark (multiple seeds)

```bash
python scripts/benchmark.py --agent round_robin --seeds 0 1 2 --episodes 20
python scripts/benchmark.py --agent ppo --checkpoint ./checkpoints/ppo/ppo_final --episodes 10
```

## Troubleshooting

- **Dashboard "Disconnected"**: Ensure `python dashboard/server.py` is running and no firewall blocks port 8000.
- **PPO/DQN slow on CPU**: Use `--device cuda` if you have a GPU and PyTorch CUDA build.
- **Out of memory**: Reduce `--steps` or batch size in the agent (edit agent file).
- **compare.py skips PPO**: Ensure the checkpoint path points to the folder/name without `.zip` and that `ppo_final.zip` (or `dqn_final.zip`) exists there.
