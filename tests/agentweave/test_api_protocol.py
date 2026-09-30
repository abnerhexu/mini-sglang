from __future__ import annotations

import asyncio

from fastapi import HTTPException
from minisgl.message import LifecycleControlMsg
from minisgl.server import api_server


def test_agent_metadata_request_conversion() -> None:
    request = api_server.AgentMetadataRequest(
        agent_id="agent-1",
        turn_id=2,
        lifecycle_id="life-2",
        parent_lifecycle_id="life-1",
        is_continuation=True,
    )
    metadata = request.to_protocol()
    assert metadata.agent_id == "agent-1"
    assert metadata.parent_lifecycle_id == "life-1"
    assert metadata.is_continuation


def test_agent_tool_request_conversion() -> None:
    request = api_server.AgentToolRequest(
        name="python",
        required_keys=["code"],
        sandbox_backend="python-process",
        runtime="python-3.12",
        dependencies=["numpy"],
        environment={"MODE": "test"},
    )
    definition = request.to_definition()
    assert definition.schema.required_keys == ("code",)
    assert definition.sandbox.key.backend == "python-process"
    assert definition.sandbox.key.dependencies == ("numpy",)
    assert definition.sandbox.environment == (("MODE", "test"),)


def test_lifecycle_endpoint_forwards_control_message() -> None:
    class FakeState:
        def __init__(self) -> None:
            self.messages = []

        async def send_one(self, msg) -> None:
            self.messages.append(msg)

    async def scenario() -> None:
        old_state = api_server._GLOBAL_STATE
        state = FakeState()
        api_server._GLOBAL_STATE = state
        try:
            response = await api_server.agentweave_event(
                api_server.LifecycleEventRequest(
                    lifecycle_id="life-1",
                    event="tool_started",
                    agent_id="agent-1",
                    turn_id=0,
                    tool_name="python",
                    estimated_duration_s=1.5,
                )
            )
        finally:
            api_server._GLOBAL_STATE = old_state

        assert response == {
            "status": "accepted",
            "lifecycle_id": "life-1",
            "event": "tool_started",
        }
        assert len(state.messages) == 1
        assert isinstance(state.messages[0], LifecycleControlMsg)
        assert state.messages[0].event.tool_name == "python"

    asyncio.run(scenario())


def test_agent_endpoint_requires_explicit_enable_flag() -> None:
    class FakeState:
        sandbox_coordinator = None

    async def scenario() -> None:
        old_state = api_server._GLOBAL_STATE
        api_server._GLOBAL_STATE = FakeState()
        try:
            try:
                await api_server.agent_run(
                    api_server.AgentRunRequest(
                        model="test",
                        agent_id="agent-1",
                        messages=[api_server.Message(role="user", content="hello")],
                        tools=[
                            api_server.AgentToolRequest(
                                name="python",
                                required_keys=["code"],
                            )
                        ],
                    )
                )
            except HTTPException as exc:
                assert exc.status_code == 403
            else:
                raise AssertionError("expected disabled runtime to reject agent run")
        finally:
            api_server._GLOBAL_STATE = old_state

    asyncio.run(scenario())
