from __future__ import annotations

import asyncio
import time
import weakref
from dataclasses import dataclass
from typing import Dict, Literal

from .prewarm import (
    PrewarmController,
    PrewarmDecision,
    ResourceMonitor,
    ResourcePressure,
    ToolSchema,
)
from .sandbox import PreparedSandbox, SandboxBackend, SandboxKey, SandboxSpec, WarmPool


@dataclass(frozen=True)
class ToolDefinition:
    schema: ToolSchema
    sandbox: SandboxSpec

    def __post_init__(self) -> None:
        if self.schema.sandbox_type != self.sandbox.key.backend:
            raise ValueError("tool schema sandbox_type must match sandbox backend")


@dataclass(frozen=True)
class SandboxAcquisition:
    sandbox: PreparedSandbox
    tool_name: str
    predicted_tool: str | None
    correct_prediction: bool
    source: Literal["prewarm", "pool", "cold"]


class SandboxCoordinator:
    """Connects streaming prediction to asynchronous sandbox preparation."""

    def __init__(
        self,
        backends: Dict[str, SandboxBackend],
        warm_pool: WarmPool,
        prewarm: PrewarmController | None = None,
        *,
        resource_monitor: ResourceMonitor | None = None,
        cold_start_prior_ms: float = 100.0,
    ) -> None:
        self.backends = backends
        self.warm_pool = warm_pool
        self.prewarm = prewarm or PrewarmController()
        self._tasks: Dict[str, asyncio.Task[PreparedSandbox]] = {}
        self._predicted_tools: Dict[str, str] = {}
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.cold_start_prior_ms = cold_start_prior_ms
        self.cold_start_estimates: Dict[SandboxKey, float] = {}
        self.cold_start_samples: Dict[SandboxKey, int] = {}
        self._sandboxes: weakref.WeakSet[PreparedSandbox] = weakref.WeakSet()
        self._preparing = 0

    @property
    def live_sandboxes(self) -> int:
        return self._preparing + sum(not sandbox.closed for sandbox in self._sandboxes)

    def observe_delta(
        self,
        lifecycle_id: str,
        delta: str,
        tools: tuple[ToolDefinition, ...],
        *,
        previous_tool: str | None = None,
        token_probability: float | None = None,
        pressure: ResourcePressure | None = None,
        cold_start_ms: float | None = None,
    ) -> PrewarmDecision:
        schemas = tuple(tool.schema for tool in tools)
        if pressure is None:
            pressure = self.resource_monitor.sample(
                live_sandboxes=self.live_sandboxes,
                pool_size=self.warm_pool.size,
                pool_capacity=self.warm_pool.max_total,
            )
        costs = {
            tool.schema.name: self.cold_start_estimates.get(
                tool.sandbox.key, self.cold_start_prior_ms
            )
            for tool in tools
        }
        decision = self.prewarm.feed(
            lifecycle_id,
            delta,
            schemas,
            previous_tool=previous_tool,
            token_probability=token_probability,
            pressure=pressure,
            cold_start_ms=costs if cold_start_ms is None else cold_start_ms,
        )
        if decision.should_prewarm and lifecycle_id not in self._tasks:
            assert decision.tool_name is not None
            definition = self._definition(decision.tool_name, tools)
            self._predicted_tools[lifecycle_id] = decision.tool_name
            self._tasks[lifecycle_id] = asyncio.create_task(self._prepare(definition.sandbox))
        return decision

    async def claim(
        self,
        lifecycle_id: str,
        actual_tool: str,
        tools: tuple[ToolDefinition, ...],
        *,
        previous_tool: str | None = None,
    ) -> SandboxAcquisition:
        predicted = self._predicted_tools.pop(lifecycle_id, None)
        task = self._tasks.pop(lifecycle_id, None)
        self.prewarm.finish(
            lifecycle_id,
            actual_tool=actual_tool,
            previous_tool=previous_tool,
        )

        if task is not None:
            try:
                speculative = await task
            except Exception:
                speculative = None
            if speculative is not None and predicted == actual_tool:
                return SandboxAcquisition(
                    speculative,
                    actual_tool,
                    predicted,
                    True,
                    "prewarm",
                )
            if speculative is not None:
                await self.warm_pool.release(speculative)

        definition = self._definition(actual_tool, tools)
        pooled = await self.warm_pool.acquire(definition.sandbox.key)
        if pooled is not None:
            return SandboxAcquisition(pooled, actual_tool, predicted, False, "pool")
        sandbox = await self._cold_prepare(definition.sandbox)
        return SandboxAcquisition(sandbox, actual_tool, predicted, False, "cold")

    async def recycle(self, acquisition: SandboxAcquisition) -> bool:
        return await self.warm_pool.release(acquisition.sandbox)

    def reject_prediction(
        self,
        lifecycle_id: str,
        *,
        previous_tool: str | None = None,
    ) -> None:
        """Recycle a wrong prediction without delaying the final agent response."""

        self.prewarm.finish(
            lifecycle_id,
            actual_tool=None,
            previous_tool=previous_tool,
        )
        self._predicted_tools.pop(lifecycle_id, None)
        task = self._tasks.pop(lifecycle_id, None)
        if task is None:
            return
        cleanup = asyncio.create_task(self._recycle_when_ready(task))
        self._cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._cleanup_tasks.discard)

    async def abort(self, lifecycle_id: str) -> None:
        self.prewarm.abort(lifecycle_id)
        self._predicted_tools.pop(lifecycle_id, None)
        task = self._tasks.pop(lifecycle_id, None)
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            sandbox = await task
        except (asyncio.CancelledError, Exception):
            return
        await self.warm_pool.release(sandbox)

    async def close(self) -> None:
        for lifecycle_id in tuple(self._tasks):
            await self.abort(lifecycle_id)
        if self._cleanup_tasks:
            await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
        await self.warm_pool.close()

    async def _recycle_when_ready(self, task: asyncio.Task[PreparedSandbox]) -> None:
        try:
            sandbox = await task
        except (asyncio.CancelledError, Exception):
            return
        await self.warm_pool.release(sandbox)

    async def _prepare(self, spec: SandboxSpec) -> PreparedSandbox:
        pooled = await self.warm_pool.acquire(spec.key)
        if pooled is not None:
            return pooled
        return await self._cold_prepare(spec)

    async def _cold_prepare(self, spec: SandboxSpec) -> PreparedSandbox:
        self._preparing += 1
        started = time.monotonic()
        try:
            sandbox = await self._backend(spec).prepare(spec)
        finally:
            self._preparing -= 1
        elapsed_ms = (time.monotonic() - started) * 1000
        previous = self.cold_start_estimates.get(spec.key, elapsed_ms)
        self.cold_start_estimates[spec.key] = 0.3 * elapsed_ms + 0.7 * previous
        self.cold_start_samples[spec.key] = self.cold_start_samples.get(spec.key, 0) + 1
        self._sandboxes.add(sandbox)
        return sandbox

    def _backend(self, spec: SandboxSpec) -> SandboxBackend:
        try:
            return self.backends[spec.key.backend]
        except KeyError as exc:
            raise ValueError(f"no sandbox backend registered for {spec.key.backend!r}") from exc

    @staticmethod
    def _definition(name: str, tools: tuple[ToolDefinition, ...]) -> ToolDefinition:
        for tool in tools:
            if tool.schema.name == name:
                return tool
        raise ValueError(f"unknown tool {name!r}")
