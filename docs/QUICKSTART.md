# Quick Start

Get from zero to a running dashboard and first training run in a few minutes.

---

## 1. Install

```bash
cd traffic
pip install -e .
```

Optional: `pip install -e ".[dev]"` for pytest.

---

## 2. Start the dashboard

```bash
python dashboard/server.py
```

Open **http://localhost:8000** in your browser. You should see the RL-Traffic-Control dashboard (cluster map, queue bars, controls).

---

## 3. Run a quick training

In the dashboard:

1. Leave **Steps** at 50000 (or set to 10000 for a very short run).
2. Check **Curriculum** (optional; ramps difficulty over time).
3. Click **▶ Train PPO**.

Training runs in the background. You’ll see the cluster map and metrics update in real time. Click **■ Stop** to stop early. When training finishes (or you stop), the PPO model is saved to `checkpoints/ppo/dashboard_latest.zip`.

---

## 4. Compare with baselines

In a **new terminal** (dashboard can stay running):

```bash
cd traffic
python analysis/compare.py --episodes 30 --ppo-checkpoint ./checkpoints/ppo/dashboard_latest --output ./logs/comparison.json
```

This evaluates round-robin, least-connections, latency-heuristic, and PPO over 30 episodes each, then writes `./logs/comparison.json` and `./logs/comparison.html`. Open the HTML file to see the comparison table.

---

## 5. View full-session metrics

While training (or after), open **http://localhost:8000/metrics** to see reward, success rate, latency, and cost over all episodes. You can export the data as JSON from that page.

---

## 6. (Optional) Train from the command line

```bash
python cli.py train ppo --steps 100000
# or
python training/train_ppo.py --steps 100000
```

Checkpoints go to `./checkpoints/ppo/ppo_final.zip`. View logs with:

```bash
tensorboard --logdir ./logs/ppo
```

---

## 7. (Optional) Docker

```bash
docker-compose up --build
```

Dashboard at **http://localhost:8000**. Mount `./logs` and `./checkpoints` if you want to persist data.

---

## Next steps

- **RUNBOOK.md** — Commands and workflows.
- **docs/API.md** — Environment and agent API.
- **IMPROVEMENTS.md** — Ideas for extending the project.

---

## Troubleshooting

- **Dashboard shows "Disconnected"** — Make sure `python dashboard/server.py` is running and you are opening http://localhost:8000.
- **"already_running" when clicking Train** — Click Stop first, then Train again.
- **compare.py says no checkpoint** — Use the path without `.zip`, e.g. `./checkpoints/ppo/dashboard_latest`, and ensure the file `dashboard_latest.zip` exists in that directory.
- **Out of memory** — Reduce steps or use `--device cpu`; for PPO you can edit the agent to use a smaller batch_size or n_steps.
