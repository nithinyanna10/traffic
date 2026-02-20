"""
MicroserviceRoutingEnv — Custom Gymnasium environment for RL-based
dynamic API request routing across simulated backend services.

Observation (34-dim float vector):
  - queue_sizes (5)
  - health_scores (5)
  - rolling_latency (5), normalized
  - failure_rates (5)
  - request_metadata (4): deadline, complexity, priority, cost_sensitivity
  - recent_routing_history (10): last 10 action decisions

Action space (Discrete(9)):
  0-4: route to service index 0-4
  5: delay request (re-queue)
  6: retry previous request
  7: drop request
  8: degrade quality (route to cheapest available with lower fidelity)

Reward: multi-objective weighted sum
Episode ends on system collapse (≥3 services failed) or 1000 steps.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Optional

from env.services import (
    ALL_SERVICES,
    BaseService,
    Request,
    ServiceResponse,
)
from env.request_generator import RequestGenerator


class MicroserviceRoutingEnv(gym.Env):
    """Microservice request routing environment."""

    metadata = {"render_modes": ["human"]}

    NUM_SERVICES = 5
    HISTORY_LEN = 10
    MAX_STEPS = 1000

    # Reward weights
    W_SLA = 2.0
    W_LATENCY = -0.002
    W_COST = -1.0
    W_STABILITY = 0.5
    W_CASCADE = -5.0
    W_DROP_HIGH_PRIORITY = -3.0
    W_RETRY_STORM = -1.0

    def __init__(
        self,
        traffic_rate: float = 1.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._traffic_rate = traffic_rate

        # --- Spaces ---
        # 5 queue + 5 health + 5 latency + 5 failure + 4 request + 10 history = 34
        obs_dim = self.NUM_SERVICES * 4 + 4 + self.HISTORY_LEN
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        # --- Internal state ---
        self._services: list[BaseService] = []
        self._req_gen = RequestGenerator(traffic_rate=traffic_rate, seed=seed)
        self._step_count = 0
        self._routing_history: list[int] = []
        self._current_request: Optional[Request] = None
        self._prev_action: Optional[int] = None
        self._consecutive_retries = 0
        self._total_requests = 0
        self._successful_requests = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
        self._delayed_queue: list[Request] = []
        self._last_response: Optional[ServiceResponse] = None

        # Snapshot for dashboard streaming
        self.last_step_info: dict = {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Re-init services
        self._services = [cls() for cls in ALL_SERVICES]
        for s in self._services:
            s.reset()

        self._req_gen = RequestGenerator(
            traffic_rate=self._traffic_rate,
            seed=seed,
        )
        self._step_count = 0
        self._routing_history = [0] * self.HISTORY_LEN
        self._current_request = None
        self._prev_action = None
        self._consecutive_retries = 0
        self._total_requests = 0
        self._successful_requests = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
        self._delayed_queue = []
        self._last_response = None
        self.last_step_info = {}

        # Generate first request
        self._current_request = self._req_gen.generate()
        if self._current_request is None:
            self._current_request = self._req_gen.generate_burst(1)[0]

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        reward = 0.0
        response: Optional[ServiceResponse] = None
        request = self._current_request

        # Tick all services (drains queues, updates cooldowns)
        for svc in self._services:
            svc.tick()

        # --- Process action ---
        if action <= 4:
            # Route to specific service
            self._consecutive_retries = 0
            svc = self._services[action]
            response = svc.process(request, self._step_count)
            reward += self._compute_routing_reward(request, response)

        elif action == 5:
            # Delay: put request back
            self._delayed_queue.append(request)
            reward += -0.1  # small penalty for delaying
            self._consecutive_retries = 0

        elif action == 6:
            # Retry previous action
            self._consecutive_retries += 1
            if self._prev_action is not None and self._prev_action <= 4:
                retry_req = Request(
                    sla_deadline_ms=request.sla_deadline_ms,
                    complexity=request.complexity,
                    priority=request.priority,
                    cost_sensitivity=request.cost_sensitivity,
                    is_retry=True,
                )
                svc = self._services[self._prev_action]
                response = svc.process(retry_req, self._step_count)
                reward += self._compute_routing_reward(retry_req, response)
                # Retry storm penalty
                if self._consecutive_retries > 3:
                    reward += self.W_RETRY_STORM * (self._consecutive_retries - 3)
            else:
                # No previous action to retry, penalty
                reward += -0.5
                self._consecutive_retries = 0

        elif action == 7:
            # Drop request
            self._consecutive_retries = 0
            if request.priority > 0.7:
                reward += self.W_DROP_HIGH_PRIORITY
            else:
                reward += -0.3  # mild penalty for low priority drops

        elif action == 8:
            # Degrade quality — route to cheapest alive service
            self._consecutive_retries = 0
            degraded_req = Request(
                sla_deadline_ms=request.sla_deadline_ms,
                complexity=request.complexity,
                priority=request.priority,
                cost_sensitivity=request.cost_sensitivity,
                degraded=True,
            )
            # Pick alive service with lowest cost (cheap CPU = index 1)
            alive = [i for i, s in enumerate(self._services) if s.state.is_alive]
            if alive:
                # prefer cheapest (index 1), then by health
                target = min(alive, key=lambda i: (
                    0 if i == 1 else 1,
                    -self._services[i].state.health
                ))
                response = self._services[target].process(degraded_req, self._step_count)
                reward += self._compute_routing_reward(degraded_req, response) * 0.8
            else:
                reward += -2.0

        # Track history
        self._routing_history.append(action)
        if len(self._routing_history) > self.HISTORY_LEN:
            self._routing_history = self._routing_history[-self.HISTORY_LEN:]
        self._prev_action = action

        # Track stats
        if response is not None:
            self._total_requests += 1
            self._total_latency += response.latency_ms
            self._total_cost += response.cost
            if response.success:
                self._successful_requests += 1
            if response.cascade_failure:
                reward += self.W_CASCADE
            self._last_response = response

        # Stability reward
        alive_count = sum(1 for s in self._services if s.state.is_alive)
        avg_health = np.mean([s.state.health for s in self._services])
        reward += self.W_STABILITY * avg_health

        # Check termination
        failed_count = sum(1 for s in self._services if not s.state.is_alive or s.state.health < 0.1)
        system_collapsed = failed_count >= 3
        truncated = self._step_count >= self.MAX_STEPS

        # Generate next request
        if self._delayed_queue:
            self._current_request = self._delayed_queue.pop(0)
        else:
            self._current_request = self._req_gen.generate()
            if self._current_request is None:
                self._current_request = self._req_gen.generate_burst(1)[0]

        obs = self._get_obs()
        info = self._get_info()
        if response:
            info["response"] = {
                "success": response.success,
                "latency_ms": response.latency_ms,
                "cost": response.cost,
                "service": response.service_name,
                "cascade": response.cascade_failure,
                "queue_overflow": response.queue_overflow,
            }

        # Save for dashboard
        self.last_step_info = info

        return obs, reward, system_collapsed, truncated, info

    def _compute_routing_reward(self, request: Request, response: ServiceResponse) -> float:
        """Compute reward from a single routing decision."""
        r = 0.0
        # SLA check
        if response.success and response.latency_ms <= request.sla_deadline_ms:
            r += self.W_SLA
        elif response.success:
            # Successful but missed SLA
            overshoot = (response.latency_ms - request.sla_deadline_ms) / request.sla_deadline_ms
            r += self.W_SLA * max(0, 1.0 - overshoot)
        else:
            r += -1.0  # failed request

        # Latency penalty (normalized)
        r += self.W_LATENCY * response.latency_ms

        # Cost penalty (weighted by request sensitivity)
        r += self.W_COST * response.cost * request.cost_sensitivity

        return r

    def _get_obs(self) -> np.ndarray:
        """Build the 34-dim observation vector."""
        obs = []

        # Queue sizes (normalized to [0, 1])
        for s in self._services:
            obs.append(min(s.state.queue_size / 25.0, 1.0))

        # Health scores
        for s in self._services:
            obs.append(s.state.health)

        # Rolling latency (normalized by 5 seconds max)
        for s in self._services:
            obs.append(min(s.state.rolling_latency / 5000.0, 1.0))

        # Failure rates
        for s in self._services:
            obs.append(s.state.failure_rate)

        # Request metadata (normalized)
        req = self._current_request
        if req:
            obs.append(min(req.sla_deadline_ms / 3000.0, 1.0))
            obs.append(req.complexity)
            obs.append(req.priority)
            obs.append(req.cost_sensitivity)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        # Routing history (normalized action indices)
        for a in self._routing_history[-self.HISTORY_LEN:]:
            obs.append(a / 8.0)

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Build info dict for logging and dashboard."""
        sr = (self._successful_requests / max(self._total_requests, 1))
        avg_lat = (self._total_latency / max(self._total_requests, 1))
        avg_cost = (self._total_cost / max(self._total_requests, 1))
        alive = sum(1 for s in self._services if s.state.is_alive)
        return {
            "step": self._step_count,
            "success_rate": sr,
            "avg_latency": avg_lat,
            "avg_cost": avg_cost,
            "alive_services": alive,
            "total_requests": self._total_requests,
            "services": [
                {
                    "name": s.name,
                    "queue": s.state.queue_size,
                    "health": s.state.health,
                    "latency": s.state.rolling_latency,
                    "failure_rate": s.state.failure_rate,
                    "alive": s.state.is_alive,
                }
                for s in self._services
            ],
            "delayed_queue_size": len(self._delayed_queue),
        }

    def set_traffic_rate(self, rate: float) -> None:
        """Used by curriculum scheduler to increase difficulty."""
        self._traffic_rate = rate
        self._req_gen.set_traffic_rate(rate)

    def get_state_snapshot(self) -> dict:
        """Full state snapshot for dashboard streaming."""
        return {
            **self._get_info(),
            "routing_history": list(self._routing_history),
            "current_action": self._prev_action,
        }
