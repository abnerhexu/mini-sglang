from __future__ import annotations

from minisgl.agentweave.kv import (
    ContinuumPolicy,
    KVCacheDescriptor,
    KVPressure,
    NoControlPolicy,
    TokenCakePolicy,
    ToolDurationHistory,
    TransferCostModel,
)


def _descriptor(lifecycle_id: str = "life-1") -> KVCacheDescriptor:
    return KVCacheDescriptor(
        lifecycle_id=lifecycle_id,
        agent_id="agent-1",
        tool_name="python",
        num_pages=100,
        bytes_per_page=1_000_000,
    )


def _pressure(*, usage: float = 0.9) -> KVPressure:
    return KVPressure(
        gpu_usage=usage,
        free_pages=10,
        waiting_pages=200,
        host_free_pages=1000,
    )


def test_no_control_never_changes_cache_state() -> None:
    policy = NoControlPolicy()
    action = policy.on_tool_started(_descriptor(), _pressure(), estimated_duration_s=10)
    assert action.action == "none"
    assert not policy.states


def test_tokencake_offloads_and_prefetches_near_predicted_completion() -> None:
    now = [10.0]
    transfer = TransferCostModel(
        d2h_bandwidth_gbps=1,
        h2d_bandwidth_gbps=1,
        d2h_base_s=0,
        h2d_base_s=0,
        safety_margin_s=0.1,
    )
    policy = TokenCakePolicy(transfer, clock=lambda: now[0])
    action = policy.on_tool_started(
        _descriptor(),
        _pressure(),
        estimated_duration_s=2.0,
    )
    assert action.action == "offload"
    assert policy.on_tick(11.0) == []

    actions = policy.on_tick(11.9)
    assert len(actions) == 1
    assert actions[0].action == "prefetch"
    finished = policy.on_tool_finished("life-1", actual_duration_s=2.1)
    assert finished.action == "none"
    assert policy.history.ewma("python") == 2.1


def test_tokencake_rejects_unprofitable_or_capacity_limited_transfer() -> None:
    policy = TokenCakePolicy(
        TransferCostModel(
            d2h_bandwidth_gbps=1,
            h2d_bandwidth_gbps=1,
            safety_margin_s=0.1,
        )
    )
    short = policy.on_tool_started(
        _descriptor(),
        _pressure(),
        estimated_duration_s=0.1,
    )
    assert short.reason == "transfer_not_amortized"

    capacity = policy.on_tool_started(
        _descriptor("life-2"),
        KVPressure(0.9, 0, 100, 10),
        estimated_duration_s=2,
    )
    assert capacity.reason == "host_capacity"


def test_continuum_uses_historical_quantile_and_learns_after_ttl_expiry() -> None:
    now = [5.0]
    history = ToolDurationHistory()
    for sample in (1.0, 2.0, 4.0):
        history.observe("python", sample)
    policy = ContinuumPolicy(
        history,
        ttl_quantile=0.75,
        default_ttl_s=2,
        max_ttl_s=10,
        clock=lambda: now[0],
    )
    action = policy.on_tool_started(
        _descriptor(),
        KVPressure(0.0, 1000, 0, 0),
    )
    assert action.action == "retain"
    assert action.estimated_duration_s == 4.0
    assert action.execute_at == 9.0

    assert policy.on_tick(8.9) == []
    expired = policy.on_tick(9.0)
    assert expired[0].action == "release"
    finished = policy.on_tool_finished("life-1", actual_duration_s=5.0)
    assert finished.action == "none"
    assert finished.reason == "ttl_already_expired"
    assert policy.history.estimate("python", quantile=1.0) == 5.0


def test_continuum_shortens_ttl_under_pressure() -> None:
    policy = ContinuumPolicy(default_ttl_s=2.0, min_ttl_s=0.05, clock=lambda: 0.0)
    low = policy.on_tool_started(
        _descriptor("low"),
        KVPressure(0.0, 100, 0, 0),
    )
    high = policy.on_tool_started(
        _descriptor("high"),
        KVPressure(1.0, 0, 100, 0),
    )
    assert low.estimated_duration_s == 2.0
    assert high.estimated_duration_s == 0.5
