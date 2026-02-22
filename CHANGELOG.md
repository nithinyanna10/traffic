# Changelog

All notable changes to RL-Traffic-Control are documented here.

## [Unreleased]

### Added

- **DQN agent** (`agents/dqn_agent.py`) and `training/train_dqn.py` for off-policy training.
- **Hard env variant** `MicroserviceRoutingHard-v0` (2000 steps, default traffic 1.5).
- **Curriculum:** burst and failure injection wired in env and request generator; `set_burst_config`, `set_failure_injection`; curriculum callback in train_ppo and dashboard training loop.
- **Dashboard:** configurable steps, curriculum checkbox, optional checkpoint load; fullscreen cluster map; Export PNG for time series and heatmap; SLA gauge and sparklines; event log; replay buffer and `/api/replay`; stress test `/api/stress`; metrics page at `/metrics` and replay page at `/replay`; checkpoint auto-save on stop.
- **CLI** (`cli.py`): `train ppo|dqn`, `compare`, `visualize`, `serve`; installable as `traffic` via pyproject.scripts.
- **Docker:** Dockerfile and docker-compose for dashboard and TensorBoard.
- **Scripts:** `sweep_ppo.py` (hyperparameter sweep), `run_experiment.py` (train → compare → visualize → report), `benchmark.py` (multi-seed benchmark), `eval_all_agents.py` (eval all agents and print table).
- **Env config:** `env/config.py` EnvConfig dataclass; `env/wrappers.py` NormalizeObservation, TimeLimitOverride.
- **Tests:** pytest suite in `tests/` (env, agents, curriculum, dashboard API, config, wrappers, integration).
- **Docs:** CONTRIBUTING.md, RUNBOOK.md, IMPROVEMENTS.md, docs/API.md, docs/ENVIRONMENT.md, docs/TROUBLESHOOTING.md.
- **Examples:** run_baseline.py, run_with_curriculum.py, export_metrics_from_dashboard.py, full_pipeline.py.

### Changed

- Dashboard UI: steps input, curriculum and checkpoint options; metrics normalized for visualize.py compatibility.
- compare.py: writes HTML report alongside JSON.
- visualize.py: accepts metrics as list or dict with "metrics" key; tolerant of "reward" vs "total_reward".

## [0.1.0] — Initial

- MicroserviceRoutingEnv (34-dim obs, 9 actions).
- PPO agent and baselines (round-robin, least-connections, latency heuristic).
- Dashboard with WebSocket state stream, cluster map, queue bars, policy heatmap, time series.
- Curriculum scheduler (traffic_rate only in callback).
- Analysis: compare.py, visualize.py.
- Simulator runner.
