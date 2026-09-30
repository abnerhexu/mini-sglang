from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from minisgl.agentweave import SandboxCoordinator, ToolDefinition
from minisgl.agentweave.prewarm import ConfidencePolicy, PrewarmController, ToolSchema
from minisgl.agentweave.protocol import AgentRequestMetadata, ToolLifecycleEvent
from minisgl.agentweave.runtime import AgentMessage, AgentRuntime
from minisgl.agentweave.sandbox import SandboxKey, SandboxSpec, SimulatedSandboxBackend, WarmPool


class FakeGenerator:
    def __init__(self, outputs: list[list[str]]) -> None:
        self.outputs = outputs
        self.metadata: list[AgentRequestMetadata] = []

    async def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
        metadata: AgentRequestMetadata,
        *,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        _ = messages, tools, max_tokens
        self.metadata.append(metadata)
        for chunk in self.outputs.pop(0):
            yield chunk


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[ToolLifecycleEvent] = []

    async def emit(self, event: ToolLifecycleEvent) -> None:
        self.events.append(event)


def _tool(name: str = "lookup") -> ToolDefinition:
    return ToolDefinition(
        ToolSchema(name, sandbox_type="simulated"),
        SandboxSpec(SandboxKey("simulated", runtime=name)),
    )


def test_agent_runtime_runs_tool_and_continuation() -> None:
    async def scenario() -> None:
        generator = FakeGenerator(
            [
                ['{"name":"lookup",', '"arguments":{"result":4}}'],
                ["The answer is 4."],
            ]
        )
        sink = RecordingSink()
        coordinator = SandboxCoordinator(
            {"simulated": SimulatedSandboxBackend()},
            WarmPool(),
            PrewarmController(ConfidencePolicy()),
        )
        runtime = AgentRuntime(generator, coordinator, sink)
        result = await runtime.run(
            [AgentMessage("user", "Find the answer")],
            [_tool()],
            agent_id="agent-1",
            run_id="run-1",
        )

        assert result.status == "completed"
        assert result.content == "The answer is 4."
        assert result.steps == 2
        assert result.tool_calls == 1
        assert generator.metadata[0].lifecycle_id == "run-1:0"
        assert generator.metadata[1].parent_lifecycle_id == "run-1:0"
        assert generator.metadata[1].is_continuation
        assert [event.event for event in sink.events] == [
            "tool_predicted",
            "tool_started",
            "tool_finished",
            "continuation_submitted",
            "generation_released",
        ]
        assert any("Tool 'lookup' returned:\n4" in message.content for message in result.messages)
        await coordinator.close()

    asyncio.run(scenario())


def test_agent_runtime_allows_model_to_repair_unknown_tool() -> None:
    async def scenario() -> None:
        generator = FakeGenerator(
            [
                ['{"name":"missing","arguments":{}}'],
                ["Recovered without a tool."],
            ]
        )
        coordinator = SandboxCoordinator(
            {"simulated": SimulatedSandboxBackend()},
            WarmPool(),
        )
        result = await AgentRuntime(generator, coordinator).run(
            [AgentMessage("user", "Do work")],
            [_tool()],
            agent_id="agent-1",
            run_id="repair",
        )

        assert result.status == "completed"
        assert result.steps == 2
        assert result.tool_calls == 0
        assert any("unknown tool" in message.content for message in result.messages)
        await coordinator.close()

    asyncio.run(scenario())
