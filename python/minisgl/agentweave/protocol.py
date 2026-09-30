from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

LifecycleEventName = Literal[
    "tool_predicted",
    "tool_started",
    "tool_finished",
    "continuation_submitted",
    "generation_released",
    "abort",
]


@dataclass(frozen=True)
class AgentRequestMetadata:
    """Identity propagated with one LLM generation in an agent workflow."""

    agent_id: str
    turn_id: int
    lifecycle_id: str
    parent_lifecycle_id: str | None = None
    is_continuation: bool = False
    tool_name_hint: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        if self.turn_id < 0:
            raise ValueError("turn_id must be non-negative")
        if not self.lifecycle_id:
            raise ValueError("lifecycle_id must not be empty")


@dataclass(frozen=True)
class ToolLifecycleEvent:
    """A coordinator-to-scheduler event at the LLM/tool boundary."""

    lifecycle_id: str
    event: LifecycleEventName
    agent_id: str | None = None
    turn_id: int | None = None
    tool_name: str | None = None
    estimated_duration_s: float | None = None
    detail: Dict[str, Any] = field(default_factory=dict)
    emitted_at_ns: int = field(default_factory=time.monotonic_ns)

    def __post_init__(self) -> None:
        if not self.lifecycle_id:
            raise ValueError("lifecycle_id must not be empty")
        if self.turn_id is not None and self.turn_id < 0:
            raise ValueError("turn_id must be non-negative")
        if self.estimated_duration_s is not None and self.estimated_duration_s <= 0:
            raise ValueError("estimated_duration_s must be positive")
