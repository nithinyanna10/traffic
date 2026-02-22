# RL-Traffic-Control API Reference

## Environment

### MicroserviceRoutingEnv

- **Id:** `MicroserviceRouting-v0`
- **Observation:** Box(34,) in [0, 1]. Layout: queue_sizes(5), health(5), rolling_latency(5), failure_rates(5), request_meta(4), history(10).
- **Action:** Discrete(9). 0–4: route to service 0–4; 5: delay; 6: retry; 7: drop; 8: degrade.
- **Episode:** Terminates when ≥3 services failed or after 1000 steps.

**Constructor:** `gym.make("MicroserviceRouting-v0")` or  
`MicroserviceRoutingEnv(traffic_rate=1.0, seed=None, render_mode=None)`.

**Methods:**
- `reset(seed=None, options=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `set_traffic_rate(rate: float)` — curriculum
- `set_burst_config(probability: float, size: int)` — curriculum
- `set_failure_injection(probability: float)` — curriculum

**info keys:** `step`, `success_rate`, `avg_latency`, `avg_cost`, `alive_services`, `services` (list of {name, queue, health, latency, failure_rate, alive}), `delayed_queue_size`, optional `response` after a route.

### MicroserviceRoutingHardEnv

- **Id:** `MicroserviceRoutingHard-v0`
- Same as above with `MAX_STEPS=2000` and default `traffic_rate=1.5`.

---

## Agents

### PPOAgent

```python
from agents.ppo_agent import PPOAgent
agent = PPOAgent(device="cpu")
agent.train(total_timesteps=100_000)
agent.save("./checkpoints/ppo/model")
agent.load("./checkpoints/ppo/model")
action = agent.predict(obs, deterministic=True)
```

**Constructor kwargs:** `env_id`, `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `policy_kwargs`, `tensorboard_log`, `device`.

### DQNAgent

```python
from agents.dqn_agent import DQNAgent
agent = DQNAgent(device="cpu")
agent.train(total_timesteps=50_000)
agent.save("./checkpoints/dqn/model")
action = agent.predict(obs, deterministic=True)
```

**Constructor kwargs:** `env_id`, `learning_rate`, `buffer_size`, `learning_starts`, `batch_size`, `gamma`, `exploration_fraction`, `exploration_final_eps`, `policy_kwargs`, `tensorboard_log`, `device`.

### Baselines

- `RoundRobinAgent()` — cycles 0,1,2,3,4.
- `LeastConnectionsAgent()` — routes to service with smallest queue among alive.
- `LatencyHeuristicAgent()` — prefers services with lower rolling latency.

All: `agent.predict(obs) -> int`, optional `agent.reset()`.

---

## Training

### Curriculum

```python
from training.curriculum import CurriculumScheduler, CurriculumConfig
scheduler = CurriculumScheduler()
config = scheduler.get_config(progress=0.5)  # progress in [0,1]
env.unwrapped.set_traffic_rate(config.traffic_rate)
env.unwrapped.set_burst_config(config.burst_probability, config.burst_size)
env.unwrapped.set_failure_injection(config.failure_injection_prob)
```

### Scripts

- `training/train_ppo.py` — PPO with optional curriculum callback.
- `training/train_dqn.py` — DQN training.

---

## Dashboard HTTP API

Base URL: `http://localhost:8000` when server is running.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Dashboard HTML |
| GET | /metrics | Metrics chart page |
| GET | /replay | Replay viewer page |
| WS | /ws/state | Live state stream (JSON) |
| POST | /api/start | Start training. Query: agent_type, steps, checkpoint, curriculum |
| POST | /api/stop | Stop training |
| GET | /api/status | training_active, state, recent_metrics |
| GET | /api/metrics | { metrics: [...] }. Query: last=N |
| GET | /api/replay | { replay: [...] }. Query: last=N |
| POST | /api/stress | Enable stress (high traffic). Body/query: duration_steps |

---

## Analysis

### compare.py

```bash
python analysis/compare.py --episodes 50 --ppo-checkpoint ./checkpoints/ppo/ppo_final --output ./logs/comparison.json
```

Produces `comparison.json` and `comparison.html`.

### visualize.py

```bash
python analysis/visualize.py --metrics ./logs/metrics.json --comparison ./logs/comparison.json --output-dir ./logs/plots
```

Expects metrics as list of { episode, total_reward, success_rate, avg_latency, avg_cost } or dict with "metrics" key.

---

## CLI

```bash
python cli.py train ppo --steps 100000
python cli.py train dqn --steps 50000
python cli.py compare --episodes 50 --ppo-checkpoint ./checkpoints/ppo/ppo_final
python cli.py visualize --metrics ./logs/ppo/metrics.json --output-dir ./logs/plots
python cli.py serve --port 8000
```

After `pip install -e .`: `traffic train ppo --steps 100000`, etc.

---

## Config and Wrappers

### EnvConfig (env/config.py)

Dataclass: traffic_rate, max_steps, seed, burst_probability, burst_size, failure_injection_prob. `to_dict()`, `from_dict()`.

### Wrappers (env/wrappers.py)

- `NormalizeObservation(env, mode="minmax")` — re-normalize obs.
- `RecordEpisodeStatistics(env)` — episode reward/length in info.
- `TimeLimitOverride(env, max_episode_steps)` — override step limit.

---

## Examples (code snippets)

### Minimal training loop (PPO)

```python
import gymnasium as gym
import env as _env_reg
from agents.ppo_agent import PPOAgent

env = gym.make("MicroserviceRouting-v0")
agent = PPOAgent(device="cpu")
agent.train(total_timesteps=10_000)
agent.save("./ckpt")
env.close()
```

### Evaluate an agent over N episodes

```python
import gymnasium as gym
import env as _env_reg
from agents.ppo_agent import PPOAgent

env = gym.make("MicroserviceRouting-v0")
agent = PPOAgent(device="cpu")
agent.load("./checkpoints/ppo/ppo_final")

rewards = []
for ep in range(50):
    obs, _ = env.reset(seed=ep)
    ep_rew = 0
    while True:
        action = agent.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        ep_rew += r
        if term or trunc:
            break
    rewards.append(ep_rew)
print("Mean reward:", sum(rewards) / len(rewards))
env.close()
```

### Curriculum in a custom loop

```python
import gymnasium as gym
import env as _env_reg
from training.curriculum import CurriculumScheduler

env = gym.make("MicroserviceRouting-v0")
unwrapped = env.unwrapped
scheduler = CurriculumScheduler()
obs, _ = env.reset(seed=0)

for step in range(5000):
    progress = step / 5000
    cfg = scheduler.get_config(progress)
    unwrapped.set_traffic_rate(cfg.traffic_rate)
    unwrapped.set_burst_config(cfg.burst_probability, cfg.burst_size)
    unwrapped.set_failure_injection(cfg.failure_injection_prob)
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    if term or trunc:
        obs, _ = env.reset()
env.close()
```

### Using the Hard env

```python
import gymnasium as gym
import env as _env_reg  # registers MicroserviceRoutingHard-v0

env = gym.make("MicroserviceRoutingHard-v0")  # 2000 steps, traffic 1.5
obs, info = env.reset()
# ... same interface as MicroserviceRouting-v0
env.close()
```

---

## Reference (signatures)

### env.microservice_env

- `MicroserviceRoutingEnv(traffic_rate=1.0, seed=None, render_mode=None)`
- `MicroserviceRoutingEnv.reset(seed=None, options=None) -> (np.ndarray, dict)`
- `MicroserviceRoutingEnv.step(action: int) -> (np.ndarray, float, bool, bool, dict)`
- `MicroserviceRoutingEnv.set_traffic_rate(rate: float) -> None`
- `MicroserviceRoutingEnv.set_burst_config(probability: float, size: int) -> None`
- `MicroserviceRoutingEnv.set_failure_injection(probability: float) -> None`
- `MicroserviceRoutingEnv.get_state_snapshot() -> dict`

### env.microservice_env_hard

- `MicroserviceRoutingHardEnv(traffic_rate=1.5, seed=None, render_mode=None)` (subclass of MicroserviceRoutingEnv, MAX_STEPS=2000)

### env.request_generator

- `RequestGenerator(traffic_rate=1.0, seed=None, burst_probability=0.0, burst_size=3)`
- `RequestGenerator.set_traffic_rate(rate: float) -> None`
- `RequestGenerator.set_burst_config(probability: float, size: int) -> None`
- `RequestGenerator.generate() -> Optional[Request]`
- `RequestGenerator.maybe_get_burst() -> Optional[List[Request]]`
- `RequestGenerator.generate_burst(count: int) -> list[Request]`

### env.services

- `BaseService.force_fail() -> None`
- `BaseService.process(request: Request, step: int) -> ServiceResponse`
- `BaseService.reset() -> None`
- `BaseService.tick() -> None`

### env.config

- `EnvConfig(traffic_rate=1.0, max_steps=1000, seed=None, burst_probability=0.0, burst_size=3, failure_injection_prob=0.0)`
- `EnvConfig.to_dict() -> dict`
- `EnvConfig.from_dict(d: dict) -> EnvConfig`
- `DEFAULT_ENV_CONFIG: EnvConfig`

### env.wrappers

- `NormalizeObservation(env, mode="minmax")` — observation(obs) -> obs
- `RecordEpisodeStatistics(env)` — from gymnasium
- `TimeLimitOverride(env, max_episode_steps: int)` — overrides step limit

### agents.ppo_agent

- `PPOAgent(env_id="MicroserviceRouting-v0", learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, policy_kwargs=None, tensorboard_log="./logs/ppo", device="auto")`
- `PPOAgent.train(total_timesteps, callback=None) -> None`
- `PPOAgent.predict(obs, deterministic=True) -> int`
- `PPOAgent.save(path: str) -> None`
- `PPOAgent.load(path: str) -> None`
- `MetricsCallback` (BaseCallback for TensorBoard)

### agents.dqn_agent

- `DQNAgent(env_id="MicroserviceRouting-v0", learning_rate=1e-4, buffer_size=100_000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=1000, exploration_fraction=0.2, exploration_initial_eps=1.0, exploration_final_eps=0.05, policy_kwargs=None, tensorboard_log="./logs/dqn", device="auto")`
- `DQNAgent.train(total_timesteps, callback=None) -> None`
- `DQNAgent.predict(obs, deterministic=True) -> int`
- `DQNAgent.save(path: str) -> None`
- `DQNAgent.load(path: str) -> None`

### agents.baselines

- `BASELINES: dict[str, type]` — "round_robin", "least_connections", "latency_heuristic"
- `RoundRobinAgent()`, `LeastConnectionsAgent()`, `LatencyHeuristicAgent()`
- `BaselineAgent.predict(obs) -> int`, `BaselineAgent.reset() -> None`

### training.curriculum

- `CurriculumConfig(traffic_rate=1.0, failure_injection_prob=0.0, burst_probability=0.0, burst_size=3)`
- `CurriculumScheduler(start=None, end=None, warmup_fraction=0.1)`
- `CurriculumScheduler.get_config(progress: float) -> CurriculumConfig`

### dashboard.server (FastAPI app)

- `update_state(state: dict) -> None` (called by training loop)
- `get_state() -> dict`
- `add_metrics(metrics: dict) -> None`
- Routes: GET /, /metrics, /replay; WS /ws/state; POST /api/start, /api/stop, /api/stress; GET /api/status, /api/metrics, /api/replay

### analysis.compare

- `evaluate_agent(agent, env, num_episodes=50) -> dict` — returns mean_reward, std_reward, mean_p95_latency, mean_cost, mean_success_rate, collapse_rate, mean_steps
- `write_html_report(results, path, num_episodes, json_output_path) -> None`
- CLI: `python analysis/compare.py --episodes 50 --ppo-checkpoint path --output path`

### analysis.visualize

- `plot_training_curves(metrics: list[dict], output_dir: str) -> None` — writes training_curves.png
- `plot_comparison(comparison_path: str, output_dir: str) -> None` — writes comparison.png
- CLI: `python analysis/visualize.py --metrics path --comparison path --output-dir path`

### cli

- `main()` — entry point for `traffic` command. Subcommands: train (ppo|dqn), compare, visualize, serve
