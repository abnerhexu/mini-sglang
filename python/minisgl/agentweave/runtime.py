from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .coordinator import SandboxCoordinator, ToolDefinition
from .prewarm import ResourcePressure, parse_complete_tool_call
from .protocol import AgentRequestMetadata, ToolLifecycleEvent


@dataclass(frozen=True)
class AgentMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    status: Literal["completed", "max_steps"]
    content: str
    messages: tuple[AgentMessage, ...]
    steps: int
    tool_calls: int


class StreamingGenerator(Protocol):
    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
        metadata: AgentRequestMetadata,
        *,
        max_tokens: int,
    ) -> AsyncIterator[str]: ...


class LifecycleEventSink(Protocol):
    async def emit(self, event: ToolLifecycleEvent) -> None: ...


class NullLifecycleEventSink:
    async def emit(self, event: ToolLifecycleEvent) -> None:
        _ = event


class AgentRuntime:
    def __init__(
        self,
        generator: StreamingGenerator,
        sandbox: SandboxCoordinator,
        event_sink: LifecycleEventSink | None = None,
    ) -> None:
        self.generator = generator
        self.sandbox = sandbox
        self.event_sink = event_sink or NullLifecycleEventSink()

    async def run(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
        *,
        agent_id: str,
        run_id: str | None = None,
        max_steps: int = 8,
        max_tokens: int = 1024,
        previous_tool: str | None = None,
        pressure: ResourcePressure = ResourcePressure(),
    ) -> AgentRunResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not tools:
            raise ValueError("at least one tool is required")

        run_id = run_id or uuid.uuid4().hex
        conversation = list(messages)
        prompt = _tool_instruction(tools)
        if not conversation or conversation[0].role != "system":
            conversation.insert(0, AgentMessage("system", prompt))
        else:
            first = conversation[0]
            conversation[0] = AgentMessage("system", f"{first.content}\n\n{prompt}")

        parent_lifecycle_id: str | None = None
        tool_calls = 0
        final_content = ""
        definitions = tuple(tools)

        for turn_id in range(max_steps):
            lifecycle_id = f"{run_id}:{turn_id}"
            if parent_lifecycle_id is not None:
                await self.event_sink.emit(
                    ToolLifecycleEvent(
                        lifecycle_id=parent_lifecycle_id,
                        event="continuation_submitted",
                        agent_id=agent_id,
                        turn_id=turn_id - 1,
                    )
                )
            metadata = AgentRequestMetadata(
                agent_id=agent_id,
                turn_id=turn_id,
                lifecycle_id=lifecycle_id,
                parent_lifecycle_id=parent_lifecycle_id,
                is_continuation=parent_lifecycle_id is not None,
                tool_name_hint=previous_tool,
            )
            chunks: list[str] = []
            prediction_emitted = False
            try:
                async for delta in self.generator.generate(
                    conversation,
                    definitions,
                    metadata,
                    max_tokens=max_tokens,
                ):
                    chunks.append(delta)
                    decision = self.sandbox.observe_delta(
                        lifecycle_id,
                        delta,
                        definitions,
                        previous_tool=previous_tool,
                        pressure=pressure,
                    )
                    if decision.should_prewarm and not prediction_emitted:
                        prediction_emitted = True
                        await self.event_sink.emit(
                            ToolLifecycleEvent(
                                lifecycle_id=lifecycle_id,
                                event="tool_predicted",
                                agent_id=agent_id,
                                turn_id=turn_id,
                                tool_name=decision.tool_name,
                                detail={
                                    "confidence": decision.confidence,
                                    "threshold": decision.threshold,
                                    "expected_value_ms": decision.expected_value_ms,
                                },
                            )
                        )
            except BaseException:
                await self.sandbox.abort(lifecycle_id)
                await self.event_sink.emit(
                    ToolLifecycleEvent(
                        lifecycle_id=lifecycle_id,
                        event="abort",
                        agent_id=agent_id,
                        turn_id=turn_id,
                    )
                )
                raise

            output = "".join(chunks)
            final_content = output
            call = parse_complete_tool_call(output)
            if call is None:
                self.sandbox.reject_prediction(lifecycle_id, previous_tool=previous_tool)
                await self.event_sink.emit(
                    ToolLifecycleEvent(
                        lifecycle_id=lifecycle_id,
                        event="generation_released",
                        agent_id=agent_id,
                        turn_id=turn_id,
                    )
                )
                conversation.append(AgentMessage("assistant", output))
                return AgentRunResult(
                    run_id,
                    "completed",
                    output,
                    tuple(conversation),
                    turn_id + 1,
                    tool_calls,
                )

            definition = _find_tool(call.name, definitions)
            if definition is None:
                self.sandbox.reject_prediction(lifecycle_id, previous_tool=previous_tool)
                conversation.extend(
                    (
                        AgentMessage("assistant", output),
                        AgentMessage(
                            "user",
                            f"Tool error: unknown tool {call.name!r}. Use one of the declared tools.",
                        ),
                    )
                )
                parent_lifecycle_id = lifecycle_id
                continue

            missing = set(definition.schema.required_keys) - set(call.arguments)
            if missing:
                self.sandbox.reject_prediction(lifecycle_id, previous_tool=previous_tool)
                conversation.extend(
                    (
                        AgentMessage("assistant", output),
                        AgentMessage(
                            "user",
                            f"Tool error: missing required arguments {sorted(missing)!r}.",
                        ),
                    )
                )
                parent_lifecycle_id = lifecycle_id
                continue

            await self.event_sink.emit(
                ToolLifecycleEvent(
                    lifecycle_id=lifecycle_id,
                    event="tool_started",
                    agent_id=agent_id,
                    turn_id=turn_id,
                    tool_name=call.name,
                )
            )
            acquisition = await self.sandbox.claim(
                lifecycle_id,
                call.name,
                definitions,
                previous_tool=previous_tool,
            )
            await self.event_sink.emit(
                ToolLifecycleEvent(
                    lifecycle_id=lifecycle_id,
                    event="sandbox_acquired",
                    agent_id=agent_id,
                    turn_id=turn_id,
                    tool_name=call.name,
                    detail={
                        "source": acquisition.source,
                        "predicted_tool": acquisition.predicted_tool,
                        "correct_prediction": acquisition.correct_prediction,
                    },
                )
            )
            success = False
            try:
                result = await acquisition.sandbox.execute(call.arguments)
                success = True
                result_text = _format_tool_result(result.output, result.stdout, result.stderr)
            except Exception as exc:
                result_text = f"Tool execution failed: {type(exc).__name__}: {exc}"
            finally:
                await self.sandbox.recycle(acquisition)

            await self.event_sink.emit(
                ToolLifecycleEvent(
                    lifecycle_id=lifecycle_id,
                    event="tool_finished",
                    agent_id=agent_id,
                    turn_id=turn_id,
                    tool_name=call.name,
                    detail={"success": success, "result_chars": len(result_text)},
                )
            )
            tool_calls += 1
            conversation.extend(
                (
                    AgentMessage("assistant", output),
                    AgentMessage("user", f"Tool {call.name!r} returned:\n{result_text}"),
                )
            )
            parent_lifecycle_id = lifecycle_id
            previous_tool = call.name

        if parent_lifecycle_id is not None:
            await self.event_sink.emit(
                ToolLifecycleEvent(
                    lifecycle_id=parent_lifecycle_id,
                    event="generation_released",
                    agent_id=agent_id,
                    turn_id=max_steps - 1,
                )
            )
        return AgentRunResult(
            run_id,
            "max_steps",
            final_content,
            tuple(conversation),
            max_steps,
            tool_calls,
        )


def _find_tool(name: str, tools: Sequence[ToolDefinition]) -> ToolDefinition | None:
    return next((tool for tool in tools if tool.schema.name == name), None)


def _tool_instruction(tools: Sequence[ToolDefinition]) -> str:
    declarations = [
        {
            "name": tool.schema.name,
            "required_arguments": list(tool.schema.required_keys),
        }
        for tool in tools
    ]
    return (
        "Available tools:\n"
        f"{json.dumps(declarations, ensure_ascii=False)}\n"
        "When a tool is needed, respond with exactly one JSON object of the form "
        '{"name":"tool_name","arguments":{...}}. '
        "After receiving a tool result, continue solving the task. "
        "When no tool is needed, answer normally without a tool-call JSON object."
    )


def _format_tool_result(output: object, stdout: str, stderr: str) -> str:
    parts = []
    if output is not None:
        parts.append(json.dumps(output, ensure_ascii=False, default=str))
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts) if parts else "(no output)"
