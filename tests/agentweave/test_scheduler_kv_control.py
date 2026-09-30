from __future__ import annotations

from types import SimpleNamespace

import torch
from minisgl.agentweave import (
    AgentRequestMetadata,
    EventLogger,
    LifecycleRegistry,
    ToolLifecycleEvent,
)
from minisgl.agentweave.kv import ContinuumPolicy, NoControlPolicy
from minisgl.kvcache import BaseCacheHandle, MatchResult
from minisgl.scheduler.scheduler import Scheduler


class FakeHandle(BaseCacheHandle):
    def get_matched_indices(self) -> torch.Tensor:
        return torch.arange(self.cached_len, dtype=torch.int32)


class FakeCacheManager:
    page_size = 1
    num_pages = 10

    def __init__(self, handle: FakeHandle) -> None:
        self.free_slots = torch.arange(10)
        self.handle = handle
        self.locked = []
        self.unlocked = []

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        _ = input_ids
        return MatchResult(self.handle)

    def lock(self, handle: BaseCacheHandle) -> None:
        self.locked.append(handle)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.unlocked.append(handle)


def _scheduler(policy, policy_name: str, handle: FakeHandle) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.agent_lifecycles = LifecycleRegistry()
    scheduler.agent_kv_policy_name = policy_name
    scheduler.agent_kv_policy = policy
    scheduler.agent_cache_handles = {}
    scheduler.agent_cache_tokens = {}
    scheduler.agent_cache_locks = set()
    scheduler.agent_kv_started = set()
    scheduler.agent_events = EventLogger(None, component="scheduler")
    scheduler.cache_manager = FakeCacheManager(handle)
    scheduler.prefill_manager = SimpleNamespace(pending_list=[])
    scheduler.engine = SimpleNamespace(kv_cache=SimpleNamespace(bytes_per_page=128))
    return scheduler


def test_scheduler_continuum_lease_locks_then_releases_prefix() -> None:
    now = [0.0]
    handle = FakeHandle(cached_len=4)
    scheduler = _scheduler(
        ContinuumPolicy(default_ttl_s=2.0, clock=lambda: now[0]),
        "continuum",
        handle,
    )
    metadata = AgentRequestMetadata("agent-1", 0, "life-1")
    scheduler.agent_lifecycles.request_received(1, metadata)
    scheduler.agent_lifecycles.apply(
        ToolLifecycleEvent(
            "life-1",
            "tool_started",
            tool_name="python",
            emitted_at_ns=1,
        )
    )
    request = SimpleNamespace(
        agent_metadata=metadata,
        input_ids=torch.arange(5, dtype=torch.int32),
    )

    scheduler._register_agent_cache(request, handle)
    assert scheduler.cache_manager.locked == [handle]
    assert "life-1" in scheduler.agent_cache_locks

    now[0] = 3.0
    scheduler._process_kv_policy_tick()
    assert scheduler.cache_manager.unlocked == [handle]
    assert "life-1" not in scheduler.agent_cache_locks
    assert "life-1" not in scheduler.agent_cache_tokens


def test_scheduler_no_control_does_not_retain_request_tokens() -> None:
    handle = FakeHandle(cached_len=4)
    scheduler = _scheduler(NoControlPolicy(), "no-control", handle)
    request = SimpleNamespace(
        agent_metadata=AgentRequestMetadata("agent-1", 0, "life-1"),
        input_ids=torch.arange(5, dtype=torch.int32),
    )
    scheduler._register_agent_cache(request, handle)
    assert scheduler.agent_cache_tokens == {}
    assert scheduler.cache_manager.locked == []
