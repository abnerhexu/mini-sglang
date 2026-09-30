from __future__ import annotations

from minisgl.agentweave import AgentRequestMetadata, LifecycleRegistry, ToolLifecycleEvent


def test_lifecycle_accepts_out_of_order_and_duplicate_events() -> None:
    registry = LifecycleRegistry(clock_ns=iter([100, 200, 300]).__next__)

    record, created = registry.apply(
        ToolLifecycleEvent(
            lifecycle_id="life-1",
            event="tool_started",
            agent_id="agent-1",
            turn_id=0,
            tool_name="python",
            estimated_duration_s=2.5,
            emitted_at_ns=20,
        )
    )
    assert created
    assert record.state == "tool_running"

    record, created = registry.request_received(
        7,
        AgentRequestMetadata(
            agent_id="agent-1",
            turn_id=0,
            lifecycle_id="life-1",
        ),
    )
    assert created
    assert record.request_received_at_ns == 100

    _, created = registry.apply(
        ToolLifecycleEvent(
            lifecycle_id="life-1",
            event="tool_started",
            emitted_at_ns=30,
        )
    )
    assert not created
    assert record.tool_started_at_ns == 20
    assert record.event_counts["tool_started"] == 2

    first, created = registry.first_token(7)
    assert first is record and created
    finished, created = registry.generation_finished(7)
    assert finished is record and created
    assert record.first_token_at_ns == 200
    assert record.generation_finished_at_ns == 300


def test_lifecycle_rejects_uid_reuse_for_different_request() -> None:
    registry = LifecycleRegistry()
    metadata = AgentRequestMetadata("agent-1", 0, "life-1")
    registry.request_received(1, metadata)

    try:
        registry.request_received(2, metadata)
    except ValueError as exc:
        assert "already associated" in str(exc)
    else:
        raise AssertionError("expected duplicate lifecycle association to fail")


def test_protocol_validation() -> None:
    for metadata in (
        lambda: AgentRequestMetadata("", 0, "life"),
        lambda: AgentRequestMetadata("agent", -1, "life"),
        lambda: AgentRequestMetadata("agent", 0, ""),
    ):
        try:
            metadata()
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid metadata to fail")

    try:
        ToolLifecycleEvent("life", "tool_started", estimated_duration_s=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid duration to fail")
