from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .confidence import (
    ConfidencePolicy,
    PrewarmDecision,
    ResourcePressure,
    ToolTransitionHistory,
)
from .parser import StreamingToolCallParser, ToolSchema


@dataclass
class PredictionState:
    parser: StreamingToolCallParser = field(default_factory=StreamingToolCallParser)
    selected_tool: str | None = None
    latest_decision: PrewarmDecision | None = None


class PrewarmController:
    """Per-lifecycle streaming prediction state.

    Resource creation is deliberately left to the coordinator. Returning a
    decision keeps parser/policy tests deterministic and prevents blocking work
    from entering the detokenization path.
    """

    def __init__(self, policy: ConfidencePolicy | None = None) -> None:
        self.policy = policy or ConfidencePolicy()
        self.history = ToolTransitionHistory()
        self.states: Dict[str, PredictionState] = {}

    def feed(
        self,
        lifecycle_id: str,
        delta: str,
        schemas: tuple[ToolSchema, ...],
        *,
        previous_tool: str | None = None,
        token_probability: float | None = None,
        pressure: ResourcePressure = ResourcePressure(),
        cold_start_ms: float = 100.0,
    ) -> PrewarmDecision:
        state = self.states.setdefault(lifecycle_id, PredictionState())
        evidence = state.parser.feed(delta, schemas)
        decision = self.policy.decide(
            evidence,
            schemas,
            previous_tool=previous_tool,
            transition_history=self.history,
            token_probability=token_probability,
            pressure=pressure,
            cold_start_ms=cold_start_ms,
        )
        if decision.should_prewarm and state.selected_tool is None:
            state.selected_tool = decision.tool_name
        state.latest_decision = decision
        return decision

    def finish(
        self,
        lifecycle_id: str,
        *,
        actual_tool: str | None,
        previous_tool: str | None = None,
    ) -> str | None:
        state = self.states.pop(lifecycle_id, None)
        if actual_tool is not None:
            self.history.observe(previous_tool, actual_tool)
        return state.selected_tool if state is not None else None

    def abort(self, lifecycle_id: str) -> None:
        self.states.pop(lifecycle_id, None)
