from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping

from .base import ExecutionResult, PreparedSandbox, SandboxBackend, SandboxKey, SandboxSpec


class SimulatedSandbox(PreparedSandbox):
    def __init__(self, key: SandboxKey, execution_delay_s: float) -> None:
        self._key = key
        self.execution_delay_s = execution_delay_s
        self.closed = False
        self.reset_count = 0

    @property
    def key(self) -> SandboxKey:
        return self._key

    async def execute(
        self, payload: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> ExecutionResult:
        if self.closed:
            raise RuntimeError("sandbox is closed")
        start = time.monotonic()
        sleep = asyncio.sleep(self.execution_delay_s)
        if timeout_s is None:
            await sleep
        else:
            await asyncio.wait_for(sleep, timeout_s)
        return ExecutionResult(
            output=payload.get("result"),
            duration_s=time.monotonic() - start,
        )

    async def reset(self) -> None:
        if self.closed:
            raise RuntimeError("sandbox is closed")
        self.reset_count += 1

    async def close(self) -> None:
        self.closed = True


class SimulatedSandboxBackend(SandboxBackend):
    def __init__(self, *, preparation_delay_s: float = 0.0, execution_delay_s: float = 0.0) -> None:
        self.preparation_delay_s = preparation_delay_s
        self.execution_delay_s = execution_delay_s
        self.prepare_count = 0

    async def prepare(self, spec: SandboxSpec) -> PreparedSandbox:
        await asyncio.sleep(self.preparation_delay_s)
        self.prepare_count += 1
        return SimulatedSandbox(spec.key, self.execution_delay_s)
