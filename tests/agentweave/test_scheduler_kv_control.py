from __future__ import annotations

from types import SimpleNamespace

import torch
from minisgl.agentweave import (
    AgentRequestMetadata,
    EventLogger,
    LifecycleRegistry,
    ToolLifecycleEvent,
)
from minisgl.agentweave.kv import (
    ContinuumPolicy,
    KVAction,
    NoControlPolicy,
    OffloadResult,
    PrefetchResult,
    TransferCostModel,
)
from minisgl.kvcache import BaseCacheHandle, MatchResult
from minisgl.scheduler.scheduler import DeferredContinuation, Scheduler


class FakeHandle(BaseCacheHandle):
    def get_matched_indices(self) -> torch.Tensor:
        return torch.arange(self.cached_len, dtype=torch.int32)


class FakeTransfer:
    def __init__(self, prefetched_handle: FakeHandle) -> None:
        self.entries = {}
        self.gpu_handles = {}
        self.prefetched_handle = prefetched_handle
        self.offloaded = []
        self.released = []

    def offload(self, lifecycle_id, tokens, handle) -> OffloadResult:
        self.offloaded.append((lifecycle_id, tokens.clone(), handle))
        self.entries[lifecycle_id] = object()
        return OffloadResult(4, 3, ())

    def prefetch(self, lifecycle_id) -> PrefetchResult:
        assert lifecycle_id in self.entries
        self.gpu_handles[lifecycle_id] = self.prefetched_handle
        return PrefetchResult(3, self.prefetched_handle.cached_len, self.prefetched_handle)

    def release(self, lifecycle_id) -> None:
        self.entries.pop(lifecycle_id, None)
        self.gpu_handles.pop(lifecycle_id, None)
        self.released.append(lifecycle_id)


class FailingTransfer(FakeTransfer):
    def __init__(self, prefetched_handle: FakeHandle, failures: int) -> None:
        super().__init__(prefetched_handle)
        self.failures = failures

    def prefetch(self, lifecycle_id) -> PrefetchResult:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("GPU pages unavailable")
        return super().prefetch(lifecycle_id)


class FakePrefillManager:
    def __init__(self) -> None:
        self.pending_list = []
        self.admitted = []
        self.runnable = True

    def add_one_req(self, msg, *, priority=False) -> None:
        self.admitted.append(msg)


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
    scheduler.agent_transfer_cost = TransferCostModel()
    scheduler.agent_cache_handles = {}
    scheduler.agent_cache_tokens = {}
    scheduler.agent_cache_locks = set()
    scheduler.agent_kv_started = set()
    scheduler.agent_deferred_continuations = {}
    scheduler.agent_continuation_restore_timeout_s = 5.0
    scheduler.agent_clock = lambda: 0.0
    scheduler.agent_host_pool = None
    scheduler.agent_kv_transfer = None
    scheduler.agent_events = EventLogger(None, component="scheduler")
    scheduler.cache_manager = FakeCacheManager(handle)
    scheduler.prefill_manager = FakePrefillManager()
    scheduler.decode_manager = SimpleNamespace(runnable=False)
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


def test_scheduler_executes_tokencake_transfer_actions() -> None:
    source_handle = FakeHandle(cached_len=4)
    restored_handle = FakeHandle(cached_len=4)
    scheduler = _scheduler(NoControlPolicy(), "tokencake", source_handle)
    transfer = FakeTransfer(restored_handle)
    scheduler.agent_kv_transfer = transfer
    scheduler.agent_cache_handles["life-1"] = source_handle
    scheduler.agent_cache_tokens["life-1"] = torch.arange(4, dtype=torch.int32)
    scheduler.agent_kv_started.add("life-1")

    scheduler._apply_agent_kv_action(KVAction("offload", "life-1", "test"))
    assert len(transfer.offloaded) == 1
    assert "life-1" not in scheduler.agent_cache_handles

    scheduler._apply_agent_kv_action(KVAction("prefetch", "life-1", "test"))
    assert scheduler.agent_cache_handles["life-1"] is restored_handle
    assert "life-1" in scheduler.agent_cache_locks

    scheduler._apply_agent_kv_action(KVAction("release", "life-1", "test"))
    assert transfer.released == ["life-1"]
    assert "life-1" not in scheduler.agent_cache_locks
    assert "life-1" not in scheduler.agent_kv_started


def test_scheduler_demand_restores_before_continuation_then_releases_lease() -> None:
    source_handle = FakeHandle(cached_len=4)
    restored_handle = FakeHandle(cached_len=4)
    scheduler = _scheduler(NoControlPolicy(), "tokencake", source_handle)
    transfer = FakeTransfer(restored_handle)
    scheduler.agent_kv_transfer = transfer
    scheduler.agent_cache_handles["life-1"] = source_handle
    scheduler.agent_cache_tokens["life-1"] = torch.arange(4, dtype=torch.int32)
    scheduler.agent_kv_started.add("life-1")
    scheduler._apply_agent_kv_action(KVAction("offload", "life-1", "test"))

    scheduler._prepare_agent_continuation("life-1")
    assert transfer.released == ["life-1"]
    assert "life-1" not in transfer.entries
    assert "life-1" not in scheduler.agent_cache_locks
    assert scheduler.agent_cache_handles == {}


def test_scheduler_retries_deferred_continuation_restore() -> None:
    source_handle = FakeHandle(cached_len=4)
    restored_handle = FakeHandle(cached_len=4)
    scheduler = _scheduler(NoControlPolicy(), "tokencake", source_handle)
    transfer = FailingTransfer(restored_handle, failures=1)
    transfer.entries["life-1"] = object()
    scheduler.agent_kv_transfer = transfer
    msg = SimpleNamespace(uid=2, agent_metadata=None)
    scheduler.agent_deferred_continuations["life-1"] = DeferredContinuation("life-1", msg, 5.0)

    scheduler._process_kv_policy_tick()
    assert "life-1" in scheduler.agent_deferred_continuations
    assert scheduler.prefill_manager.admitted == []

    scheduler._process_kv_policy_tick()
    assert scheduler.agent_deferred_continuations == {}
    assert scheduler.prefill_manager.admitted == [msg]
    assert transfer.released == ["life-1"]


def test_scheduler_deferred_continuation_falls_back_after_timeout() -> None:
    source_handle = FakeHandle(cached_len=4)
    scheduler = _scheduler(NoControlPolicy(), "tokencake", source_handle)
    transfer = FailingTransfer(source_handle, failures=10)
    transfer.entries["life-1"] = object()
    scheduler.agent_kv_transfer = transfer
    scheduler.agent_clock = lambda: 6.0
    msg = SimpleNamespace(uid=2, agent_metadata=None)
    scheduler.agent_deferred_continuations["life-1"] = DeferredContinuation("life-1", msg, 5.0)

    scheduler._process_kv_policy_tick()
    assert scheduler.agent_deferred_continuations == {}
    assert scheduler.prefill_manager.admitted == [msg]
    assert transfer.released == ["life-1"]


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


def test_continuum_prioritizes_only_continuations_admitted_within_ttl() -> None:
    from minisgl.agentweave.kv import KVCacheDescriptor, KVPressure
    from minisgl.core import SamplingParams
    from minisgl.scheduler.prefill import PrefillManager

    now = [0.0]
    policy = ContinuumPolicy(clock=lambda: now[0])
    scheduler = _scheduler(policy, "continuum", FakeHandle(cached_len=4))
    scheduler.prefill_manager = PrefillManager(None, None, None)

    def request(uid, parent=None):
        metadata = AgentRequestMetadata(
            "agent",
            1,
            f"child-{uid}",
            parent_lifecycle_id=parent,
            is_continuation=parent is not None,
        )
        return SimpleNamespace(
            uid=uid,
            input_ids=torch.tensor([1]),
            sampling_params=SamplingParams(),
            agent_metadata=metadata,
        )

    for name in ("early", "late"):
        policy.on_tool_started(
            KVCacheDescriptor(name, "agent", "python", 4, 128), KVPressure(0, 100, 0, 0)
        )
    now[0] = 0.5
    policy.on_tool_finished("early")
    policy.on_tool_finished("late")
    scheduler._admit_agent_request(request(1))
    scheduler._admit_agent_request(request(2, "early"))
    now[0] = 3.0
    scheduler._admit_agent_request(request(3, "late"))
    assert [req.uid for req in scheduler.prefill_manager.pending_list] == [2, 1, 3]
    assert [req.priority for req in scheduler.prefill_manager.pending_list] == [True, False, False]
    assert not policy.continuation_deadlines
