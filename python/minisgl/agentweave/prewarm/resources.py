from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .confidence import ResourcePressure


def memory_usage() -> float:
    """Linux memory pressure, including a cgroup limit when one is configured."""
    info = {
        key: int(value.split()[0])
        for key, value in (
            line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    usage = 1 - info["MemAvailable"] / info["MemTotal"]
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            directory = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            limit_file = directory / "memory.max"
            if limit_file.exists():
                limit = limit_file.read_text().strip()
                if limit != "max":
                    usage = max(usage, int((directory / "memory.current").read_text()) / int(limit))
    return min(1.0, max(0.0, usage))


class ResourceMonitor:
    """Cache OS reads while keeping sandbox/pool counters current for each delta."""

    def __init__(
        self,
        *,
        max_sandboxes: int = 32,
        sample_interval_s: float = 0.25,
        memory_reader: Callable[[], float] = memory_usage,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_sandboxes = max_sandboxes
        self.sample_interval_s = sample_interval_s
        self.memory_reader = memory_reader
        self.clock = clock
        self._sampled_at = float("-inf")
        self._memory_usage = 0.0

    def sample(
        self, *, live_sandboxes: int, pool_size: int, pool_capacity: int
    ) -> ResourcePressure:
        now = self.clock()
        if now - self._sampled_at >= self.sample_interval_s:
            self._memory_usage = self.memory_reader()
            self._sampled_at = now
        return ResourcePressure(
            process_usage=live_sandboxes / self.max_sandboxes,
            memory_usage=self._memory_usage,
            pool_usage=pool_size / pool_capacity if pool_capacity else 0.0,
        )
