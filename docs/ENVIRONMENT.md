# MicroserviceRoutingEnv — Full Specification

This document describes the observation space, action space, reward signal, and service behaviors in detail.

---

## Observation Space

**Type:** `gymnasium.spaces.Box(low=0.0, high=1.0, shape=(34,), dtype=np.float32)`

### Layout (34 dimensions)

| Index | Slice | Description | Normalization |
|-------|--------|-------------|----------------|
| 0–4 | queue_sizes | Current queue length per service | min(q/25, 1) |
| 5–9 | health_scores | Health per service (0=dead, 1=healthy) | raw [0,1] |
| 10–14 | rolling_latency | EMA latency per service | min(lat/5000, 1) |
| 15–19 | failure_rates | EMA failure rate per service | raw [0,1] |
| 20 | request: sla_deadline | Current request’s SLA deadline (ms) | min(deadline/3000, 1) |
| 21 | request: complexity | Current request complexity | [0,1] |
| 22 | request: priority | Current request priority | [0,1] |
| 23 | request: cost_sensitivity | Current request cost sensitivity | [0,1] |
| 24–33 | routing_history | Last 10 actions taken | action_index / 8 |

All values are in [0, 1] except that health and failure_rates are already in that range; queue and latency are clipped.

---

## Action Space

**Type:** `gymnasium.spaces.Discrete(9)`

| Action | Name | Effect |
|--------|------|--------|
| 0 | Route to service 0 | Send current request to Fast GPU |
| 1 | Route to service 1 | Send current request to Cheap CPU |
| 2 | Route to service 2 | Send current request to Flaky Spot |
| 3 | Route to service 3 | Send current request to Serverless |
| 4 | Route to service 4 | Send current request to Queue Cluster |
| 5 | Delay | Put request back in delayed queue; small penalty |
| 6 | Retry | Re-send to previous service (or penalty if none) |
| 7 | Drop | Drop request; large penalty if high priority |
| 8 | Degrade | Route to cheapest alive service with degraded quality |

---

## Reward Formula

Per-step reward is a weighted sum of:

- **SLA success:** +2.0 if request succeeded and latency ≤ deadline; partial if succeeded but over deadline; −1 if failed.
- **Latency penalty:** −0.002 × latency_ms (normalized).
- **Cost penalty:** −1.0 × cost × request.cost_sensitivity.
- **Stability bonus:** +0.5 × mean(health over all services).
- **Cascade penalty:** −5.0 if the response had cascade_failure.
- **Drop high-priority:** −3.0 if action was drop and request priority > 0.7.
- **Retry storm:** −1.0 × (consecutive_retries − 3) when retries > 3.
- **Delay:** −0.1 for action 5.

Episode terminates when ≥3 services are dead (health < 0.1 or is_alive=False) or after MAX_STEPS (1000 for v0, 2000 for Hard).

---

## Services (summary)

1. **Fast GPU (0):** 20–40 ms, expensive, max concurrency 5; queue overflow returns failure.
2. **Cheap CPU (1):** 120–250 ms, very cheap, stable (health always 1).
3. **Flaky Spot (2):** 60–120 ms, ~5% crash rate, 10-step cooldown after crash.
4. **Serverless (3):** Cold 1–3 s, warm 30–50 ms; resets after 20 idle steps.
5. **Queue Cluster (4):** Non-linear queue growth; cascading failure when queue > 20.

Each service exposes: name, queue_size, health, rolling_latency, failure_rate, is_alive. After a route, `info["response"]` contains success, latency_ms, cost, service_name, queue_overflow, cascade_failure.

---

## Curriculum Hooks

- **set_traffic_rate(rate):** Request generator probability per step (1.0 ≈ one request per step on average).
- **set_burst_config(prob, size):** Each step, with probability `prob`, add `size` extra requests to the delayed queue.
- **set_failure_injection(prob):** Each step, with probability `prob`, force a random alive service to fail (force_fail).

These are used by CurriculumScheduler to ramp difficulty over training (e.g. traffic_rate 0.6 → 3.0, burst_prob 0 → 0.15, failure_injection 0 → 0.05).

---

## Info Dict (after step)

- `step`: Current step in episode.
- `success_rate`: Fraction of requests that succeeded so far this episode.
- `avg_latency`: Mean latency (ms) of completed requests this episode.
- `avg_cost`: Mean cost of completed requests this episode.
- `alive_services`: Count of services with is_alive=True.
- `services`: List of 5 dicts with name, queue, health, latency, failure_rate, alive.
- `delayed_queue_size`: Length of the delayed queue (action 5).
- `response` (if a route was taken): success, latency_ms, cost, service, cascade, queue_overflow.

---

## MicroserviceRoutingHard-v0

Same as above with:

- **MAX_STEPS = 2000**
- **Default traffic_rate = 1.5**

Registration: `gym.register("MicroserviceRoutingHard-v0", entry_point="env.microservice_env_hard:MicroserviceRoutingHardEnv", max_episode_steps=2000)`.

---

## Reward Constants (env constants)

Defined in `MicroserviceRoutingEnv`:

- `W_SLA = 2.0`
- `W_LATENCY = -0.002`
- `W_COST = -1.0`
- `W_STABILITY = 0.5`
- `W_CASCADE = -5.0`
- `W_DROP_HIGH_PRIORITY = -3.0`
- `W_RETRY_STORM = -1.0`

SLA component: if success and latency ≤ deadline, add W_SLA; if success but over deadline, add W_SLA * max(0, 1 - overshoot_ratio); if not success, add −1.0.

---

## Service Implementation Notes

- **Fast GPU:** `_active_requests` incremented on each request; drained by 2 per tick. Health = max(0, 1 - active/(MAX_CONCURRENCY+3)). Queue overflow when active > 5.
- **Cheap CPU:** Single queue drained by 1 per tick. No failure; latency 120–250 ms * complexity factor.
- **Flaky Spot:** CRASH_PROB = 0.05; on crash, is_alive=False for COOLDOWN_STEPS = 10, then health recovers from 0.5.
- **Serverless:** Tracks idle steps; after 20 idle steps, next request pays cold latency (1–3 s), else warm (30–50 ms).
- **Queue Cluster:** Queue grows with requests; when queue > 20, cascade_failure can be set and health drops.

All services update `state.rolling_latency` and `state.failure_rate` via exponential moving average after each response. `force_fail()` (curriculum) sets is_alive=False and health=0; FlakySpot sets cooldown.
