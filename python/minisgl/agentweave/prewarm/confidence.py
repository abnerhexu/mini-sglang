from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict

from .parser import ToolCallEvidence, ToolSchema


@dataclass(frozen=True)
class ResourcePressure:
    process_usage: float = 0.0
    memory_usage: float = 0.0
    pool_usage: float = 0.0

    @property
    def normalized(self) -> float:
        return min(1.0, max(0.0, max(self.process_usage, self.memory_usage, self.pool_usage)))


@dataclass(frozen=True)
class PrewarmDecision:
    tool_name: str | None
    confidence: float
    threshold: float
    expected_value_ms: float
    should_prewarm: bool
    reason: str


class ToolTransitionHistory:
    def __init__(self) -> None:
        self._transitions: DefaultDict[str, Counter[str]] = defaultdict(Counter)

    def observe(self, previous_tool: str | None, current_tool: str) -> None:
        self._transitions[previous_tool or "<start>"][current_tool] += 1

    def probability(self, previous_tool: str | None, current_tool: str) -> float | None:
        counts = self._transitions.get(previous_tool or "<start>")
        if not counts:
            return None
        # Laplace smoothing avoids making unseen transitions impossible.
        return (counts[current_tool] + 1) / (sum(counts.values()) + len(counts) + 1)


class ConfidencePolicy:
    def __init__(
        self,
        *,
        base_threshold: float = 0.62,
        max_pressure_penalty: float = 0.25,
        wrong_prediction_cost_ms: float = 10.0,
    ) -> None:
        if not 0 <= base_threshold <= 1:
            raise ValueError("base_threshold must be in [0, 1]")
        self.base_threshold = base_threshold
        self.max_pressure_penalty = max_pressure_penalty
        self.wrong_prediction_cost_ms = wrong_prediction_cost_ms

    def decide(
        self,
        evidence: ToolCallEvidence,
        schemas: tuple[ToolSchema, ...],
        *,
        previous_tool: str | None = None,
        transition_history: ToolTransitionHistory | None = None,
        token_probability: float | None = None,
        pressure: ResourcePressure = ResourcePressure(),
        cold_start_ms: float = 100.0,
    ) -> PrewarmDecision:
        schema_map = {schema.name: schema for schema in schemas}
        candidate = self._select_candidate(evidence, schemas, previous_tool, transition_history)
        threshold = min(
            0.95,
            self.base_threshold + pressure.normalized * self.max_pressure_penalty,
        )
        if candidate is None:
            return PrewarmDecision(
                None, 0.0, threshold, -self.wrong_prediction_cost_ms, False, "ambiguous"
            )

        confidence = 0.0
        if evidence.exact_name == candidate:
            confidence += 0.65
        elif evidence.name_prefix and candidate.startswith(evidence.name_prefix):
            progress = len(evidence.name_prefix) / max(1, len(candidate))
            confidence += 0.25 + 0.25 * progress

        schema = schema_map[candidate]
        if schema.required_keys:
            matched = len(set(schema.required_keys) & set(evidence.observed_keys))
            confidence += 0.15 * matched / len(schema.required_keys)
        elif evidence.exact_name == candidate:
            confidence += 0.05

        if transition_history is not None:
            history_probability = transition_history.probability(previous_tool, candidate)
            if history_probability is not None:
                confidence += 0.15 * history_probability

        if token_probability is not None:
            confidence += 0.05 * min(1.0, max(0.0, token_probability))

        confidence = min(1.0, confidence)
        expected_value = (
            confidence * cold_start_ms
            - (1 - confidence) * self.wrong_prediction_cost_ms
            - pressure.normalized * self.wrong_prediction_cost_ms
        )
        should_prewarm = confidence >= threshold and expected_value > 0
        reason = (
            "selected"
            if should_prewarm
            else (
                "resource_pressure"
                if confidence >= self.base_threshold and threshold > confidence
                else "low_confidence"
            )
        )
        return PrewarmDecision(
            candidate,
            confidence,
            threshold,
            expected_value,
            should_prewarm,
            reason,
        )

    @staticmethod
    def _select_candidate(
        evidence: ToolCallEvidence,
        schemas: tuple[ToolSchema, ...],
        previous_tool: str | None,
        history: ToolTransitionHistory | None,
    ) -> str | None:
        if evidence.exact_name is not None:
            return evidence.exact_name
        if len(evidence.candidate_names) == 1:
            return evidence.candidate_names[0]
        if not evidence.candidate_names or history is None:
            return None
        probabilities = [
            (history.probability(previous_tool, name), name) for name in evidence.candidate_names
        ]
        known = [
            (probability, name) for probability, name in probabilities if probability is not None
        ]
        if not known:
            return None
        known.sort(reverse=True)
        if len(known) > 1 and known[0][0] == known[1][0]:
            return None
        return known[0][1]
