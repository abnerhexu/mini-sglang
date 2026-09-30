from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict

from .base import PreparedSandbox, SandboxKey


@dataclass
class PoolEntry:
    sandbox: PreparedSandbox
    released_at: float


@dataclass(frozen=True)
class WarmPoolStats:
    hits: int
    misses: int
    releases: int
    evictions: int
    reset_failures: int
    size: int


class WarmPool:
    def __init__(
        self,
        *,
        max_total: int = 8,
        max_per_key: int = 2,
        ttl_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_total < 0 or max_per_key < 0 or ttl_s < 0:
            raise ValueError("pool limits and ttl_s must be non-negative")
        self.max_total = max_total
        self.max_per_key = max_per_key
        self.ttl_s = ttl_s
        self.clock = clock
        self._entries: Dict[SandboxKey, Deque[PoolEntry]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._releases = 0
        self._evictions = 0
        self._reset_failures = 0

    @property
    def size(self) -> int:
        return sum(len(entries) for entries in self._entries.values())

    async def acquire(self, key: SandboxKey) -> PreparedSandbox | None:
        to_close: list[PreparedSandbox] = []
        async with self._lock:
            to_close.extend(self._remove_expired_locked())
            entries = self._entries.get(key)
            if entries:
                sandbox = entries.pop().sandbox
                if not entries:
                    del self._entries[key]
                self._hits += 1
            else:
                sandbox = None
                self._misses += 1
        await self._close_all(to_close)
        return sandbox

    async def release(self, sandbox: PreparedSandbox) -> bool:
        try:
            await sandbox.reset()
        except Exception:
            self._reset_failures += 1
            await sandbox.close()
            return False

        to_close: list[PreparedSandbox] = []
        accepted = True
        async with self._lock:
            to_close.extend(self._remove_expired_locked())
            if self.max_total == 0 or self.max_per_key == 0:
                accepted = False
                to_close.append(sandbox)
            else:
                while self.size >= self.max_total:
                    evicted = self._pop_oldest_locked()
                    if evicted is None:
                        break
                    to_close.append(evicted)
                    self._evictions += 1
                entries = self._entries[sandbox.key]
                while len(entries) >= self.max_per_key:
                    to_close.append(entries.popleft().sandbox)
                    self._evictions += 1
                entries.append(PoolEntry(sandbox, self.clock()))
                self._releases += 1
        await self._close_all(to_close)
        return accepted

    async def discard_expired(self) -> int:
        async with self._lock:
            to_close = self._remove_expired_locked()
        await self._close_all(to_close)
        return len(to_close)

    async def close(self) -> None:
        async with self._lock:
            to_close = [entry.sandbox for entries in self._entries.values() for entry in entries]
            self._entries.clear()
        await self._close_all(to_close)

    def stats(self) -> WarmPoolStats:
        return WarmPoolStats(
            hits=self._hits,
            misses=self._misses,
            releases=self._releases,
            evictions=self._evictions,
            reset_failures=self._reset_failures,
            size=self.size,
        )

    def _remove_expired_locked(self) -> list[PreparedSandbox]:
        now = self.clock()
        expired: list[PreparedSandbox] = []
        for key, entries in list(self._entries.items()):
            while entries and now - entries[0].released_at >= self.ttl_s:
                expired.append(entries.popleft().sandbox)
                self._evictions += 1
            if not entries:
                del self._entries[key]
        return expired

    def _pop_oldest_locked(self) -> PreparedSandbox | None:
        oldest_key: SandboxKey | None = None
        oldest_time = float("inf")
        for key, entries in self._entries.items():
            if entries and entries[0].released_at < oldest_time:
                oldest_key = key
                oldest_time = entries[0].released_at
        if oldest_key is None:
            return None
        entries = self._entries[oldest_key]
        sandbox = entries.popleft().sandbox
        if not entries:
            del self._entries[oldest_key]
        return sandbox

    @staticmethod
    async def _close_all(sandboxes: list[PreparedSandbox]) -> None:
        if sandboxes:
            await asyncio.gather(*(sandbox.close() for sandbox in sandboxes))
