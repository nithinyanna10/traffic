# Troubleshooting

Common issues and fixes for RL-Traffic-Control.

---

## Dashboard

### "Disconnected" in the UI

- Ensure the dashboard server is running: `python dashboard/server.py` or `python cli.py serve`.
- Check that nothing else is using port 8000.
- If using Docker, ensure port 8000 is published and the container is running.

### Training does not start when clicking "Train PPO"

- Check the browser console (F12) for errors.
- Ensure `/api/start` returns 200; if it returns "already_running", click Stop first.
- If the server is behind a proxy, ensure WebSocket upgrade is allowed for `/ws/state`.

### Metrics page or Replay page is empty

- Start a training run from the main dashboard first. Metrics and replay are populated only while training is running or after at least one episode has finished (metrics persist in server memory until restart).

---

## Training

### Out of memory (OOM) during PPO/DQN training

- Reduce `n_steps` (PPO) or `buffer_size` (DQN) in the agent constructor.
- Use `--device cpu` if GPU memory is limited.
- Reduce batch_size.

### Training is very slow on CPU

- Use fewer steps for quick experiments (e.g. 20_000).
- Use a GPU: install PyTorch with CUDA and pass `device="cuda"` (or `--device cuda` in scripts).

### Curriculum not changing difficulty

- In the dashboard, ensure "Curriculum" is checked when starting.
- For CLI training, curriculum is applied by default in `train_ppo.py` via CurriculumCallback; ensure you are not overwriting env config elsewhere.

---

## Comparison and visualization

### compare.py skips PPO

- Ensure the path passed to `--ppo-checkpoint` is the directory + base name without `.zip` (e.g. `./checkpoints/ppo/ppo_final`), and that `ppo_final.zip` exists there.
- Run training first: `python training/train_ppo.py --steps 50000`.

### visualize.py says "No metrics file found"

- The default path is `./logs/metrics/metrics.json`. Either run a training that writes that file, or export metrics from the dashboard (e.g. save the response of `GET /api/metrics` to a JSON file) and pass `--metrics path/to/file.json`.
- visualize.py accepts a JSON file that is either a list of metric dicts or a dict with a "metrics" key (e.g. the dashboard API response).

### comparison.json not found for visualize

- Run `python analysis/compare.py` first and pass the same path to `--comparison` that compare.py wrote (e.g. `./logs/comparison.json`).

---

## Environment and agents

### Agent predicts invalid action

- Observation must be shape (34,) and float32. If you wrapped the env, ensure the wrapper does not change obs shape.
- Baselines expect obs[0:5] = queue, obs[5:10] = health; ensure the env is the standard MicroserviceRouting-v0 (or Hard) and not a heavily modified version.

### Episode never terminates

- Standard env terminates when ≥3 services are dead or after MAX_STEPS (1000 or 2000). If you use a custom wrapper that overrides step, ensure termination/truncation is still set correctly.
- If you disabled failure injection and use only stable baselines, episodes will often run to MAX_STEPS.

### ImportError for env or agents

- Run from project root or ensure the project is installed: `pip install -e .`
- When running scripts, add the project root to PYTHONPATH: `export PYTHONPATH=/path/to/traffic:$PYTHONPATH` or `sys.path.insert(0, "/path/to/traffic")`.

---

## Docker

### Container exits immediately

- Check logs: `docker-compose logs dashboard`. Common cause: missing dependency or wrong working directory; ensure Dockerfile COPY paths match your repo layout.
- If you mount volumes for checkpoints/logs, ensure the app has write permissions.

### TensorBoard shows no runs

- Ensure the logs directory is mounted and that training has been run (e.g. from the host or another container) writing to that directory. The default compose mounts `./logs`; run `python training/train_ppo.py --log-dir ./logs/ppo` from the host so that `./logs/ppo` contains events.

---

## Tests

### pytest cannot find env or agents

- Run pytest from the project root: `cd traffic && pytest tests/`.
- conftest.py adds the project root to sys.path; do not run tests from a different cwd without PYTHONPATH.

### test_dashboard_api fails with "already_running"

- The test that starts training (round_robin, 100 steps) may leave the server in a running state if the test process is killed. Restart the server or avoid running that test in the same process as a live server. Using TestClient against the app in process does not start a real server, so this should be rare.
