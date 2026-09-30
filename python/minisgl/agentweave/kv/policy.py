from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Literal

KVActionName = Literal["none", "retain", "release", "offload", "prefetch"]


@dataclass(frozen=True)
class KVCacheDescriptor:
    lifecycle_id: str
    agent_id: str
    tool_name: str
    num_pages: int
    bytes_per_page: int

    @property
    def size_bytes(self) -> int:
        return self.num_pages * self.bytes_per_page


@dataclass(frozen=True)
class KVPressure:
    gpu_usage: float
    free_pages: int
    waiting_pages: int
    host_free_pages: int

    def __post_init__(self) -> None:
        if not 0 <= self.gpu_usage <= 1:
            raise ValueError("gpu_usage must be in [0, 1]")
        if min(self.free_pages, self.waiting_pages, self.host_free_pages) < 0:
            raise ValueError("page counts must be non-negative")


@dataclass(frozen=True)
class KVAction:
    action: KVActionName
    lifecycle_id: str
    reason: str
    execute_at: float | None = None
    estimated_duration_s: float | None = None


class ToolDurationHistory:
    def __init__(self, *, max_samples: int = 128, ewma_alpha: float = 0.3) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if not 0 < ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        self.max_samples = max_samples
        self.ewma_alpha = ewma_alpha
        self._samples: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.max_samples))
        self._ewma: Dict[str, float] = {}

    def observe(self, tool_name: str, duration_s: float) -> None:
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        self._samples[tool_name].append(duration_s)
        previous = self._ewma.get(tool_name)
        self._ewma[tool_name] = (
            duration_s
            if previous is None
            else self.ewma_alpha * duration_s + (1 - self.ewma_alpha) * previous
        )

    def estimate(self, tool_name: str, *, quantile: float = 0.5) -> float | None:
        if not 0 <= quantile <= 1:
            raise ValueError("quantile must be in [0, 1]")
        samples = self._samples.get(tool_name)
        if not samples:
            return None
        ordered = sorted(samples)
        index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
        return ordered[index]

    def ewma(self, tool_name: str) -> float | None:
        return self._ewma.get(tool_name)


@dataclass
class TransferCostModel:
    d2h_bandwidth_gbps: float = 12.0
    h2d_bandwidth_gbps: float = 12.0
    d2h_base_s: float = 0.0001
    h2d_base_s: float = 0.0001
    safety_margin_s: float = 0.002
    calibration_alpha: float = 0.2

    def __post_init__(self) -> None:
        if self.d2h_bandwidth_gbps <= 0 or self.h2d_bandwidth_gbps <= 0:
            raise ValueError("transfer bandwidth must be positive")
        if min(self.d2h_base_s, self.h2d_base_s, self.safety_margin_s) < 0:
            raise ValueError("transfer latency values must be non-negative")
        if not 0 <= self.calibration_alpha <= 1:
            raise ValueError("calibration_alpha must be in [0, 1]")

    def estimate(self, size_bytes: int, direction: Literal["d2h", "h2d"]) -> float:
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if direction == "d2h":
            return self.d2h_base_s + size_bytes / (self.d2h_bandwidth_gbps * 1e9)
        return self.h2d_base_s + size_bytes / (self.h2d_bandwidth_gbps * 1e9)

    def round_trip(self, size_bytes: int) -> float:
        return (
            self.estimate(size_bytes, "d2h")
            + self.estimate(size_bytes, "h2d")
            + self.safety_margin_s
        )

    def observe(
        self,
        size_bytes: int,
        direction: Literal["d2h", "h2d"],
        elapsed_s: float,
    ) -> float | None:
        if size_bytes < 0 or elapsed_s < 0:
            raise ValueError("transfer samples must be non-negative")
        if size_bytes == 0 or self.calibration_alpha == 0:
            return None
        base_s = self.d2h_base_s if direction == "d2h" else self.h2d_base_s
        payload_s = elapsed_s - base_s
        if payload_s <= 0:
            return None
        observed_gbps = size_bytes / (payload_s * 1e9)
        if direction == "d2h":
            current = self.d2h_bandwidth_gbps
            self.d2h_bandwidth_gbps = (
                self.calibration_alpha * observed_gbps + (1 - self.calibration_alpha) * current
            )
        else:
            current = self.h2d_bandwidth_gbps
            self.h2d_bandwidth_gbps = (
                self.calibration_alpha * observed_gbps + (1 - self.calibration_alpha) * current
            )
        return observed_gbps


@dataclass
class _PolicyState:
    descriptor: KVCacheDescriptor
    started_at: float
    predicted_duration_s: float
    deadline: float | None = None
    prefetched: bool = False
    released: bool = False


class ToolKVPolicy(ABC):
    def __init__(
        self,
        history: ToolDurationHistory | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.history = history or ToolDurationHistory()
        self.clock = clock
        self.states: Dict[str, _PolicyState] = {}

    @abstractmethod
    def on_tool_started(
        self,
        descriptor: KVCacheDescriptor,
        pressure: KVPressure,
        *,
        estimated_duration_s: float | None = None,
    ) -> KVAction: ...

    def on_tick(self, now: float | None = None) -> list[KVAction]:
        _ = now
        return []

    def on_tool_finished(
        self, lifecycle_id: str, *, actual_duration_s: float | None = None
    ) -> KVAction:
        state = self.states.pop(lifecycle_id, None)
        if state is None:
            return KVAction("none", lifecycle_id, "unknown_lifecycle")
        duration = (
            actual_duration_s
            if actual_duration_s is not None
            else max(0.0, self.clock() - state.started_at)
        )
        self.history.observe(state.descriptor.tool_name, duration)
        return KVAction("none", lifecycle_id, "tool_finished")

    def on_abort(self, lifecycle_id: str) -> KVAction:
        state = self.states.pop(lifecycle_id, None)
        return KVAction(
            "release" if state is not None else "none",
            lifecycle_id,
            "aborted" if state is not None else "unknown_lifecycle",
        )

    def _duration(self, descriptor: KVCacheDescriptor, supplied: float | None) -> float:
        if supplied is not None:
            if supplied <= 0:
                raise ValueError("estimated_duration_s must be positive")
            return supplied
        return self.history.ewma(descriptor.tool_name) or 0.0


class NoControlPolicy(ToolKVPolicy):
    def on_tool_started(
        self,
        descriptor: KVCacheDescriptor,
        pressure: KVPressure,
        *,
        estimated_duration_s: float | None = None,
    ) -> KVAction:
        _ = pressure, estimated_duration_s
        return KVAction("none", descriptor.lifecycle_id, "engine_default")


class TokenCakePolicy(ToolKVPolicy):
    def __init__(
        self,
        transfer: TransferCostModel | None = None,
        history: ToolDurationHistory | None = None,
        *,
        min_gpu_usage: float = 0.7,
        prefetch_lead_factor: float = 1.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(history, clock=clock)
        self.transfer = transfer or TransferCostModel()
        self.min_gpu_usage = min_gpu_usage
        self.prefetch_lead_factor = prefetch_lead_factor

    def on_tool_started(
        self,
        descriptor: KVCacheDescriptor,
        pressure: KVPressure,
        *,
        estimated_duration_s: float | None = None,
    ) -> KVAction:
        duration = self._duration(descriptor, estimated_duration_s)
        d2h = self.transfer.estimate(descriptor.size_bytes, "d2h")
        h2d = self.transfer.estimate(descriptor.size_bytes, "h2d")
        needed = d2h + h2d + self.transfer.safety_margin_s
        lifecycle_id = descriptor.lifecycle_id
        if descriptor.num_pages <= 0:
            return KVAction("none", lifecycle_id, "empty_cache", estimated_duration_s=duration)
        if pressure.host_free_pages < descriptor.num_pages:
            return KVAction("none", lifecycle_id, "host_capacity", estimated_duration_s=duration)
        if (
            pressure.waiting_pages <= pressure.free_pages
            and pressure.gpu_usage < self.min_gpu_usage
        ):
            return KVAction("none", lifecycle_id, "low_pressure", estimated_duration_s=duration)
        if duration <= needed:
            return KVAction(
                "none", lifecycle_id, "transfer_not_amortized", estimated_duration_s=duration
            )

        now = self.clock()
        prefetch_at = (
            now
            + duration
            - max(
                self.transfer.safety_margin_s,
                self.prefetch_lead_factor * h2d,
            )
        )
        self.states[lifecycle_id] = _PolicyState(
            descriptor,
            now,
            duration,
            deadline=max(now, prefetch_at),
        )
        return KVAction(
            "offload",
            lifecycle_id,
            "profitable_idle_window",
            execute_at=now,
            estimated_duration_s=duration,
        )

    def on_tick(self, now: float | None = None) -> list[KVAction]:
        now = self.clock() if now is None else now
        actions = []
        for lifecycle_id, state in self.states.items():
            if not state.prefetched and state.deadline is not None and now >= state.deadline:
                state.prefetched = True
                actions.append(
                    KVAction(
                        "prefetch",
                        lifecycle_id,
                        "predicted_tool_completion",
                        execute_at=now,
                        estimated_duration_s=state.predicted_duration_s,
                    )
                )
        return actions

    def on_tool_finished(
        self, lifecycle_id: str, *, actual_duration_s: float | None = None
    ) -> KVAction:
        state = self.states.pop(lifecycle_id, None)
        if state is None:
            return KVAction("none", lifecycle_id, "not_offloaded")
        duration = (
            actual_duration_s
            if actual_duration_s is not None
            else max(0.0, self.clock() - state.started_at)
        )
        self.history.observe(state.descriptor.tool_name, duration)
        return KVAction(
            "none" if state.prefetched else "prefetch",
            lifecycle_id,
            "already_prefetched" if state.prefetched else "tool_returned_early",
        )


def create_kv_policy(
    name: str,
    *,
    clock: Callable[[], float] = time.monotonic,
    transfer: TransferCostModel | None = None,
) -> ToolKVPolicy:
    if name == "no-control":
        return NoControlPolicy(clock=clock)
    if name == "tokencake":
        return TokenCakePolicy(transfer=transfer, clock=clock)
    if name == "continuum":
        return ContinuumPolicy(clock=clock)
    raise ValueError(f"unknown AgentWeave KV policy {name!r}")


class ContinuumPolicy(ToolKVPolicy):
    def __init__(
        self,
        history: ToolDurationHistory | None = None,
        *,
        ttl_quantile: float = 0.75,
        default_ttl_s: float = 2.0,
        min_ttl_s: float = 0.05,
        max_ttl_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(history, clock=clock)
        if not 0 <= ttl_quantile <= 1:
            raise ValueError("ttl_quantile must be in [0, 1]")
        self.ttl_quantile = ttl_quantile
        self.default_ttl_s = default_ttl_s
        self.min_ttl_s = min_ttl_s
        self.max_ttl_s = max_ttl_s

    def on_tool_started(
        self,
        descriptor: KVCacheDescriptor,
        pressure: KVPressure,
        *,
        estimated_duration_s: float | None = None,
    ) -> KVAction:
        historical = self.history.estimate(
            descriptor.tool_name,
            quantile=self.ttl_quantile,
        )
        ttl = historical if historical is not None else self.default_ttl_s
        if estimated_duration_s is not None:
            if estimated_duration_s <= 0:
                raise ValueError("estimated_duration_s must be positive")
            ttl = min(ttl, estimated_duration_s)

        # Opportunity cost rises with both occupancy and unsatisfied demand.
        demand_ratio = pressure.waiting_pages / max(1, pressure.free_pages + pressure.waiting_pages)
        pressure_scale = max(0.0, 1.0 - 0.75 * max(pressure.gpu_usage, demand_ratio))
        ttl = min(self.max_ttl_s, max(0.0, ttl * pressure_scale))
        lifecycle_id = descriptor.lifecycle_id
        if descriptor.num_pages <= 0 or ttl < self.min_ttl_s:
            return KVAction("release", lifecycle_id, "ttl_too_short", execute_at=self.clock())

        now = self.clock()
        deadline = now + ttl
        self.states[lifecycle_id] = _PolicyState(
            descriptor,
            now,
            estimated_duration_s or historical or self.default_ttl_s,
            deadline=deadline,
        )
        return KVAction(
            "retain",
            lifecycle_id,
            "ttl_active",
            execute_at=deadline,
            estimated_duration_s=ttl,
        )

    def on_tick(self, now: float | None = None) -> list[KVAction]:
        now = self.clock() if now is None else now
        actions = []
        for lifecycle_id, state in self.states.items():
            if not state.released and state.deadline is not None and now >= state.deadline:
                state.released = True
                actions.append(KVAction("release", lifecycle_id, "ttl_expired", execute_at=now))
        return actions

    def on_tool_finished(
        self, lifecycle_id: str, *, actual_duration_s: float | None = None
    ) -> KVAction:
        state = self.states.pop(lifecycle_id, None)
        if state is None:
            return KVAction("none", lifecycle_id, "unknown_lifecycle")
        duration = (
            actual_duration_s
            if actual_duration_s is not None
            else max(0.0, self.clock() - state.started_at)
        )
        self.history.observe(state.descriptor.tool_name, duration)
        return KVAction(
            "none" if state.released else "release",
            lifecycle_id,
            "ttl_already_expired" if state.released else "continuation_arrived",
        )
