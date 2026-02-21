# RL-Traffic-Control

**Learning to Route Internet Requests Under Failure**

A research-grade reinforcement learning project that trains an agent to dynamically route API requests across simulated backend microservices — optimizing latency, cost, and reliability simultaneously.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)
![Gymnasium](https://img.shields.io/badge/env-Gymnasium-green)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)

---

## Quick Start

### 1. Install

```bash
cd traffic
pip install -e .
```

### 2. Train PPO Agent

```bash
python training/train_ppo.py --steps 100000
```

### 3. Launch Dashboard & Watch Live

```bash
# Terminal 1: Start dashboard server
python dashboard/server.py

# Terminal 2: Open browser to http://localhost:8000
# Click "Train PPO" in the UI to start training with live visualization
```

### 4. Compare RL vs Baselines

```bash
python analysis/compare.py --episodes 50 --ppo-checkpoint ./checkpoints/ppo/ppo_final
```

---

## Architecture

```
traffic/
├── env/                        # Custom Gymnasium environment
│   ├── microservice_env.py     # MicroserviceRoutingEnv (34-dim obs, 9 actions)
│   ├── services.py             # 5 service simulators with stochastic behaviors
│   └── request_generator.py    # Traffic generation with curriculum support
│
├── agents/                     # RL + baseline agents
│   ├── ppo_agent.py            # SB3 PPO with 256-256 MLP
│   ├── impala_agent.py         # IMPALA with V-trace + distributed workers
│   └── baselines.py            # Round-robin, least-connections, latency heuristic
│
├── training/                   # Training infrastructure
│   ├── train_ppo.py            # PPO training script with curriculum
│   ├── train_impala.py         # IMPALA distributed training script
│   ├── curriculum.py           # Difficulty scheduler
│   ├── replay_buffer.py        # Rare event prioritized replay
│   └── metrics.py              # Metric tracking + TensorBoard
│
├── dashboard/                  # Live visualization
│   ├── server.py               # FastAPI + WebSocket server
│   └── static/index.html       # Canvas dashboard (cluster map, queues, charts)
│
├── simulator/                  # Unified runner
│   └── runner.py               # Run any agent with optional dashboard streaming
│
├── analysis/                   # Evaluation tools
│   ├── compare.py              # Automated RL vs baselines comparison
│   └── visualize.py            # Matplotlib training curve plots
│
├── pyproject.toml
└── README.md
```

---

## Environment Details

### Simulated Services

| Service | Latency | Cost | Behavior |
|---------|---------|------|----------|
| **Fast GPU** | 20–40ms | High | Limited concurrency (5 slots), queue overflow |
| **Cheap CPU** | 120–250ms | Very low | Extremely stable |
| **Flaky Spot** | 60–120ms | Medium | ~5% crash rate, 10-step cooldown |
| **Cold Start Serverless** | 1–3s cold, 30–50ms warm | Low | Resets after 20 idle steps |
| **Overloaded Queue** | Non-linear growth | Medium | Cascading failure at queue > 20 |

### Observation Space (34-dim)

`[queue_sizes(5), health(5), latency(5), failure_rate(5), request_meta(4), history(10)]`

### Action Space (Discrete 9)

| Action | Description |
|--------|-------------|
| 0–4 | Route to service 0–4 |
| 5 | Delay request |
| 6 | Retry previous |
| 7 | Drop request |
| 8 | Degrade quality |

### Reward Signal

Multi-objective: SLA success (+2), latency penalty, cost penalty, stability bonus, cascade penalty (−5), dropped high-priority (−3), retry storm penalty.

**Episode terminates** when ≥3 services fail simultaneously (system collapse) or after 1000 steps.

---

## Dashboard

The live dashboard (`http://localhost:8000`) shows:

- **Cluster Map** — 5 service nodes with health-colored borders, animated request particles, cascade pulse effects
- **Queue Bars** — Real-time queue sizes per service
- **Policy Heatmap** — Action distribution showing learned routing preferences
- **Time Series** — Rolling plots of success rate, latency, and cost

---



---

## License

MIT
