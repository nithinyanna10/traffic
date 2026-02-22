# Contributing to RL-Traffic-Control

Thanks for your interest in contributing. This document covers development setup, running tests, and project structure.

## Development setup

```bash
git clone https://github.com/nithinyanna10/traffic.git
cd traffic
pip install -e ".[dev]"
```

Optional dev dependencies include `pytest` and `pytest-asyncio` for tests. For dashboard API tests you may also need `httpx` (or use `TestClient` from FastAPI which is included).

## Running tests

```bash
# All tests
pytest tests/ -v

# By module
pytest tests/test_env.py -v
pytest tests/test_agents.py -v
pytest tests/test_curriculum.py -v
pytest tests/test_dashboard_api.py -v
```

Tests assume the project root is on `PYTHONPATH`; `conftest.py` adds the project root and registers the Gymnasium env.

## Project structure

- **env/** — Gymnasium environment: `MicroserviceRoutingEnv`, services, request generator. Curriculum hooks: `set_traffic_rate`, `set_burst_config`, `set_failure_injection`.
- **agents/** — RL agents (PPO, DQN) and rule-based baselines. All expose `predict(obs) -> action`.
- **training/** — Training scripts and curriculum. `train_ppo.py` and `train_dqn.py`; curriculum callback updates env difficulty over time.
- **dashboard/** — FastAPI server and static UI. WebSocket streams state; API supports start/stop, metrics, replay, stress.
- **analysis/** — `compare.py` (agent comparison + HTML report), `visualize.py` (training curves from metrics JSON).
- **simulator/** — Unified runner for headless or dashboard-backed runs.
- **scripts/** — `sweep_ppo.py` (hyperparameter sweep), `run_experiment.py` (train → compare → visualize → report).

## Code style

- Python 3.9+.
- Use `from __future__ import annotations` in modules that use type hints.
- Prefer dataclasses or typed dicts for config; keep env and agent interfaces stable for compatibility with the dashboard and compare/visualize pipeline.

## Adding a new agent

1. Implement an agent that supports `predict(obs, deterministic=...)` and optional `save(path)` / `load(path)`.
2. Add it to the dashboard server’s `_run_training` branch (or a generic loader) and to `analysis/compare.py` if it should appear in comparison reports.
3. Add a short test in `tests/test_agents.py` that checks `predict` returns an action in `[0, 8]`.

## Adding a new env variant

1. Register a new Gymnasium id (e.g. `MicroserviceRoutingHard-v0`) in `env/__init__.py` with a different `entry_point` or kwargs (e.g. `max_steps`, `traffic_rate`).
2. If the observation/action space shape changes, update the dashboard and agents accordingly.

## Dashboard changes

- **Backend:** Endpoints and state live in `dashboard/server.py`. Training runs in a background thread and pushes state via `update_state()`; WebSocket clients receive it at 10 Hz.
- **Frontend:** Single-page app in `dashboard/static/index.html`; metrics chart page in `metrics.html`. Both use vanilla JS and canvas.

## Release / versioning

Version is in `pyproject.toml`. Bump version and tag for releases.
