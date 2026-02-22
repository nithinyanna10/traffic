# Traffic Project — What More Can Be Done

A prioritized list of changes you can make next, from quick wins to larger features.

---

## 1. Dashboard & UX

| Change | Effort | Impact |
|--------|--------|--------|
| **Configurable training from UI** — Input fields for steps (e.g. 50k / 100k), learning rate, and a "Stress test" toggle that temporarily raises traffic rate so you can watch the agent under load. | Small | High |
| **Episode replay** — After an episode ends, show a "Replay last episode" button that replays the last N steps (using stored state snapshots or re-running with same seed) so you can review collapses. | Medium | High |
| **Fullscreen cluster map** — Button to expand the cluster map to fullscreen for demos or monitoring. | Small | Medium |
| **Dark/light theme toggle** — CSS variable swap; store preference in `localStorage`. | Small | Low |
| **Export charts** — "Download PNG" for time series and policy heatmap. | Small | Medium |
| **Sound alerts (optional, muted by default)** — Short beep on cascade failure or when success rate drops below threshold. | Small | Low |
| **Comparison view in dashboard** — Run two agents (e.g. PPO vs round-robin) in parallel and show side-by-side stats (would require server to support multiple runners or A/B state). | Large | High |

---

## 2. Backend / Server

| Change | Effort | Impact |
|--------|--------|--------|
| **Curriculum in dashboard training** — Use `CurriculumScheduler` in `_run_training` and call `set_traffic_rate(config.traffic_rate)` (and later burst/failure) so difficulty ramps up as steps increase. | Small | High |
| **Traffic burst API** — `POST /api/burst?rate=2.5&steps=100` to temporarily increase traffic for the running env (requires env to expose something like `set_traffic_rate` and a "burst end" step). | Small | Medium |
| **Checkpoint save from dashboard** — After N episodes or on "Stop", optionally save PPO checkpoint to `./checkpoints/ppo/dashboard_latest.zip` and show "Model saved" in UI. | Small | Medium |
| **Run from checkpoint** — `POST /api/start?agent_type=ppo&checkpoint=./checkpoints/ppo/ppo_final` so you can continue training or evaluate a saved model in the UI. | Small | Medium |
| **Metrics history API** — `GET /api/metrics?last=100` returning `_training_metrics` so the dashboard can draw full-session reward/success curves, not just rolling. | Small | Medium |

---

## 3. Environment & Simulation

| Change | Effort | Impact |
|--------|--------|--------|
| **Wire curriculum burst/failure** — `CurriculumConfig` has `burst_probability` and `failure_injection_prob` but the env doesn’t use them yet. Add to env: e.g. each step with probability `burst_probability` call `generate_burst(burst_size)` and merge into a queue; with probability `failure_injection_prob` force a random service to fail. Then have `CurriculumCallback` pass these from scheduler to env (env needs `set_burst_config(...)` and `set_failure_injection(p)`). | Medium | High |
| **More services or action space** — Add a 6th service (e.g. "Reserved capacity") or extra actions (e.g. "replicate to two services") for richer policies. | Medium | Medium |
| **Configurable episode length** — `MAX_STEPS` or traffic rate as env init args so you can do short episodes for debugging and long for evaluation. | Small | Low |
| **Structured info for analytics** — Add to `info`: e.g. `actions_this_episode: [0,1,0,2,...]`, `rewards_per_step: [...]` so compare/visualize scripts can compute action distribution and reward variance without re-running. | Small | Medium |

---

## 4. Training & Evaluation

| Change | Effort | Impact |
|--------|--------|--------|
| **Use curriculum in `train_ppo.py`** — Ensure `CurriculumCallback` is actually registered and that the env supports all curriculum knobs (traffic + burst + failure) so training gets harder over time. | Small | High |
| **Periodic evaluation during training** — Every K steps, run 5–10 eval episodes with `deterministic=True` and log `eval/mean_reward`, `eval/success_rate` to TensorBoard so you see generalization, not just train reward. | Medium | High |
| **Hyperparameter sweep script** — `scripts/sweep_ppo.py` that runs `train_ppo` with different lr, batch size, or curriculum end config and writes results to a table or JSON for comparison. | Medium | Medium |
| **Compare script → HTML report** — Extend `analysis/compare.py` to generate an `comparison.html` with tables and small plots (or embed matplotlib base64) so you can share results. | Small | Medium |
| **Align visualize.py with saved metrics** — Ensure whatever `train_ppo` (or dashboard) saves matches the keys expected by `visualize.py` (e.g. `total_reward`, `success_rate`, `avg_latency`, `avg_cost`); add a small export in dashboard or train script that writes that format. | Small | Medium |

---

## 5. DevOps & DX

| Change | Effort | Impact |
|--------|--------|--------|
| **Docker Compose** — Single `docker-compose.yml` that runs dashboard on port 8000 and optionally a TensorBoard on 6006, with volume mounts for checkpoints and logs. | Small | Medium |
| **One-command demo** — `make demo` or `scripts/demo.sh` that starts dashboard, opens browser, and runs a short PPO training so a new user can see the full loop in one go. | Small | Low |
| **Pre-commit or CI** — Run `pytest` (if you add tests) and a linter (ruff/black) on push. | Small | Low |
| **README badges and "Quick start with Docker"** — Link to TensorBoard and dashboard screenshots; add a "Run with Docker" section. | Small | Low |

---

## Suggested order

1. **Curriculum in dashboard** — Use `CurriculumScheduler` in `_run_training` and ramp traffic (and optionally burst/failure once env supports it). High impact, small effort.
2. **Configurable steps + optional checkpoint save** — Let user pick steps in UI; optionally save model on stop.
3. **Wire burst/failure in env + curriculum** — Makes training and dashboard stress tests more interesting.
4. **Episode replay or fullscreen cluster** — Improves demos and debugging.
5. **Compare → HTML report + align visualize.py** — Better sharing and reproducibility of results.

If you tell me which area you care about most (dashboard vs env vs training vs DevOps), I can implement a concrete set of changes next.
