"""
Simulated backend services with realistic stochastic behaviors.

Each service implements process(request, current_step) -> ServiceResponse
"""

from __future__ import annotations

import dataclasses
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Request:
    """An incoming API request with SLA metadata."""
    sla_deadline_ms: float      # max acceptable latency in ms
    complexity: float            # 0.0 - 1.0, affects processing time
    priority: float              # 0.0 - 1.0, higher = more important
    cost_sensitivity: float      # 0.0 - 1.0, higher = prefers cheaper
    degraded: bool = False       # if True, simulate cheaper/faster processing
    is_retry: bool = False       # if True, this is a retried request


@dataclass
class ServiceResponse:
    """Result of processing a request on a service."""
    success: bool
    latency_ms: float
    cost: float
    service_name: str
    queue_overflow: bool = False
    cascade_failure: bool = False


@dataclass
class ServiceState:
    """Observable state of a service."""
    queue_size: int = 0
    health: float = 1.0          # 0.0 = dead, 1.0 = healthy
    rolling_latency: float = 0.0 # exponential moving average
    failure_rate: float = 0.0    # exponential moving average of failures
    is_alive: bool = True


class BaseService(ABC):
    """Abstract base for all service simulators."""

    def __init__(self, name: str):
        self.name = name
        self.state = ServiceState()
        self._latency_ema_alpha = 0.1
        self._failure_ema_alpha = 0.05

    def reset(self) -> None:
        """Reset service to initial state."""
        self.state = ServiceState()
        self._on_reset()

    def _on_reset(self) -> None:
        """Subclass hook for reset."""
        pass

    def tick(self) -> None:
        """Called every environment step to update internal state."""
        self._on_tick()

    def _on_tick(self) -> None:
        """Subclass hook for per-step updates."""
        pass

    def force_fail(self) -> None:
        """Force this service to fail (e.g. for curriculum failure injection)."""
        self.state.is_alive = False
        self.state.health = 0.0
        if hasattr(self, "_cooldown_remaining"):
            self._cooldown_remaining = getattr(self, "COOLDOWN_STEPS", 10)

    def process(self, request: Request, step: int) -> ServiceResponse:
        """Process a request. Returns ServiceResponse."""
        response = self._handle_request(request, step)
        # Update rolling stats
        self.state.rolling_latency = (
            (1 - self._latency_ema_alpha) * self.state.rolling_latency
            + self._latency_ema_alpha * response.latency_ms
        )
        fail_val = 0.0 if response.success else 1.0
        self.state.failure_rate = (
            (1 - self._failure_ema_alpha) * self.state.failure_rate
            + self._failure_ema_alpha * fail_val
        )
        return response

    @abstractmethod
    def _handle_request(self, request: Request, step: int) -> ServiceResponse:
        ...


class FastGPUService(BaseService):
    """Very fast but expensive; limited concurrency with queue overflow."""

    MAX_CONCURRENCY = 5

    def __init__(self):
        super().__init__("fast_gpu_service")
        self._active_requests = 0

    def _on_reset(self) -> None:
        self._active_requests = 0

    def _on_tick(self) -> None:
        # Drain one active slot per tick (simplified)
        self._active_requests = max(0, self._active_requests - 2)
        self.state.queue_size = self._active_requests
        self.state.health = max(0.0, 1.0 - self._active_requests / (self.MAX_CONCURRENCY + 3))

    def _handle_request(self, request: Request, step: int) -> ServiceResponse:
        self._active_requests += 1
        if self._active_requests > self.MAX_CONCURRENCY:
            self.state.queue_size = self._active_requests
            return ServiceResponse(
                success=False,
                latency_ms=5.0,
                cost=0.01,
                service_name=self.name,
                queue_overflow=True,
            )
        complexity_factor = 1.0 + request.complexity * 0.5
        if request.degraded:
            complexity_factor *= 0.5
        latency = random.uniform(20, 40) * complexity_factor
        cost = 0.10 * complexity_factor
        self.state.queue_size = self._active_requests
        return ServiceResponse(
            success=True,
            latency_ms=latency,
            cost=cost,
            service_name=self.name,
        )


class CheapCPUService(BaseService):
    """Slow but cheap and very stable."""

    def __init__(self):
        super().__init__("cheap_cpu_service")
        self._queue = 0

    def _on_reset(self) -> None:
        self._queue = 0

    def _on_tick(self) -> None:
        self._queue = max(0, self._queue - 1)
        self.state.queue_size = self._queue
        self.state.health = 1.0  # always stable

    def _handle_request(self, request: Request, step: int) -> ServiceResponse:
        self._queue += 1
        complexity_factor = 1.0 + request.complexity
        if request.degraded:
            complexity_factor *= 0.6
        latency = random.uniform(120, 250) * complexity_factor
        cost = 0.01 * complexity_factor
        self.state.queue_size = self._queue
        return ServiceResponse(
            success=True,
            latency_ms=latency,
            cost=cost,
            service_name=self.name,
        )


class FlakySpotInstance(BaseService):
    """Medium latency, randomly crashes, restarts after cooldown."""

    CRASH_PROB = 0.05
    COOLDOWN_STEPS = 10

    def __init__(self):
        super().__init__("flaky_spot_instance")
        self._cooldown_remaining = 0
        self._queue = 0

    def _on_reset(self) -> None:
        self._cooldown_remaining = 0
        self._queue = 0
        self.state.is_alive = True

    def _on_tick(self) -> None:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self.state.is_alive = True
                self.state.health = 0.5  # recovering
            else:
                self.state.health = 0.0
        elif self.state.is_alive:
            self.state.health = min(1.0, self.state.health + 0.1)
        self._queue = max(0, self._queue - 1)
        self.state.queue_size = self._queue

    def _handle_request(self, request: Request, step: int) -> ServiceResponse:
        if not self.state.is_alive:
            return ServiceResponse(
                success=False,
                latency_ms=1.0,
                cost=0.0,
                service_name=self.name,
            )
        # Random crash
        if random.random() < self.CRASH_PROB:
            self.state.is_alive = False
            self._cooldown_remaining = self.COOLDOWN_STEPS
            self.state.health = 0.0
            return ServiceResponse(
                success=False,
                latency_ms=5.0,
                cost=0.02,
                service_name=self.name,
            )
        self._queue += 1
        complexity_factor = 1.0 + request.complexity * 0.3
        if request.degraded:
            complexity_factor *= 0.7
        latency = random.uniform(60, 120) * complexity_factor
        cost = 0.04 * complexity_factor
        self.state.queue_size = self._queue
        return ServiceResponse(
            success=True,
            latency_ms=latency,
            cost=cost,
            service_name=self.name,
        )


class ColdStartServerless(BaseService):
    """First request very slow, then fast. Resets after idle period."""

    COLD_LATENCY_MIN = 1000  # 1 sec
    COLD_LATENCY_MAX = 3000  # 3 sec
    WARM_LATENCY_MIN = 30
    WARM_LATENCY_MAX = 50
    IDLE_RESET_STEPS = 20

    def __init__(self):
        super().__init__("cold_start_serverless")
        self._is_warm = False
        self._idle_counter = 0
        self._queue = 0

    def _on_reset(self) -> None:
        self._is_warm = False
        self._idle_counter = 0
        self._queue = 0

    def _on_tick(self) -> None:
        self._idle_counter += 1
        if self._is_warm and self._idle_counter >= self.IDLE_RESET_STEPS:
            self._is_warm = False
        self._queue = max(0, self._queue - 1)
        self.state.queue_size = self._queue
        self.state.health = 1.0 if self._is_warm else 0.6

    def _handle_request(self, request: Request, step: int) -> ServiceResponse:
        self._idle_counter = 0
        self._queue += 1
        complexity_factor = 1.0 + request.complexity * 0.2
        if request.degraded:
            complexity_factor *= 0.5

        if not self._is_warm:
            latency = random.uniform(self.COLD_LATENCY_MIN, self.COLD_LATENCY_MAX) * complexity_factor
            self._is_warm = True
        else:
            latency = random.uniform(self.WARM_LATENCY_MIN, self.WARM_LATENCY_MAX) * complexity_factor

        cost = 0.03 * complexity_factor
        self.state.queue_size = self._queue
        return ServiceResponse(
            success=True,
            latency_ms=latency,
            cost=cost,
            service_name=self.name,
        )


class OverloadedQueueCluster(BaseService):
    """Latency increases non-linearly with load. Cascading failure if abused."""

    CASCADE_THRESHOLD = 20
    BASE_LATENCY = 50

    def __init__(self):
        super().__init__("overloaded_queue_cluster")
        self._queue = 0
        self._cascade_active = False
        self._cascade_cooldown = 0

    def _on_reset(self) -> None:
        self._queue = 0
        self._cascade_active = False
        self._cascade_cooldown = 0

    def _on_tick(self) -> None:
        if self._cascade_active:
            self._cascade_cooldown -= 1
            if self._cascade_cooldown <= 0:
                self._cascade_active = False
                self._queue = 0
                self.state.health = 0.3
            else:
                self.state.health = 0.0
                self._queue = max(0, self._queue - 1)
        else:
            self._queue = max(0, self._queue - 3)
            load_factor = self._queue / max(self.CASCADE_THRESHOLD, 1)
            self.state.health = max(0.0, 1.0 - load_factor)
        self.state.queue_size = self._queue

    def _handle_request(self, request: Request, step: int) -> ServiceResponse:
        if self._cascade_active:
            return ServiceResponse(
                success=False,
                latency_ms=2.0,
                cost=0.0,
                service_name=self.name,
                cascade_failure=True,
            )
        self._queue += 1
        # Check for cascade
        if self._queue >= self.CASCADE_THRESHOLD:
            self._cascade_active = True
            self._cascade_cooldown = 15
            self.state.health = 0.0
            return ServiceResponse(
                success=False,
                latency_ms=5000.0,
                cost=0.0,
                service_name=self.name,
                cascade_failure=True,
            )
        complexity_factor = 1.0 + request.complexity * 0.4
        if request.degraded:
            complexity_factor *= 0.6
        # Non-linear latency growth
        latency = self.BASE_LATENCY * (1 + (self._queue ** 2) / 100) * complexity_factor
        cost = 0.05 * complexity_factor
        self.state.queue_size = self._queue
        return ServiceResponse(
            success=True,
            latency_ms=latency,
            cost=cost,
            service_name=self.name,
        )


# Ordered list matching action indices 0-4
ALL_SERVICES = [
    FastGPUService,
    CheapCPUService,
    FlakySpotInstance,
    ColdStartServerless,
    OverloadedQueueCluster,
]
