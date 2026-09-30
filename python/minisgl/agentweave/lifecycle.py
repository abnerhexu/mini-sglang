from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict

from .protocol import AgentRequestMetadata, ToolLifecycleEvent


@dataclass
class LifecycleRecord:
    lifecycle_id: str
    agent_id: str | None = None
    turn_id: int | None = None
    parent_lifecycle_id: str | None = None
    uid: int | None = None
    request_received_at_ns: int | None = None
    first_token_at_ns: int | None = None
    generation_finished_at_ns: int | None = None
    tool_predicted_at_ns: int | None = None
    tool_started_at_ns: int | None = None
    tool_finished_at_ns: int | None = None
    continuation_submitted_at_ns: int | None = None
    generation_released_at_ns: int | None = None
    aborted_at_ns: int | None = None
    tool_name: str | None = None
    estimated_duration_s: float | None = None
    event_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def state(self) -> str:
        if self.aborted_at_ns is not None:
            return "aborted"
        if self.continuation_submitted_at_ns is not None:
            return "continuation_submitted"
        if self.tool_finished_at_ns is not None:
            return "tool_finished"
        if self.tool_started_at_ns is not None:
            return "tool_running"
        if self.generation_finished_at_ns is not None:
            return "generation_finished"
        if self.first_token_at_ns is not None:
            return "generating"
        return "pending"


class LifecycleRegistry:
    """Idempotent lifecycle bookkeeping.

    External tool events and generation callbacks can arrive in either order. The
    registry records the first timestamp for each transition and counts repeats.
    It intentionally does not enforce a strict transition order.
    """

    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self.records: Dict[str, LifecycleRecord] = {}
        self.uid_to_lifecycle: Dict[int, str] = {}

    def _get_or_create(
        self,
        lifecycle_id: str,
        *,
        agent_id: str | None = None,
        turn_id: int | None = None,
    ) -> LifecycleRecord:
        record = self.records.get(lifecycle_id)
        if record is None:
            record = LifecycleRecord(
                lifecycle_id=lifecycle_id,
                agent_id=agent_id,
                turn_id=turn_id,
            )
            self.records[lifecycle_id] = record
        else:
            if record.agent_id is None:
                record.agent_id = agent_id
            if record.turn_id is None:
                record.turn_id = turn_id
        return record

    @staticmethod
    def _set_once(record: LifecycleRecord, field_name: str, timestamp_ns: int) -> bool:
        if getattr(record, field_name) is not None:
            return False
        setattr(record, field_name, timestamp_ns)
        return True

    @staticmethod
    def _count(record: LifecycleRecord, event: str) -> None:
        record.event_counts[event] = record.event_counts.get(event, 0) + 1

    def request_received(
        self,
        uid: int,
        metadata: AgentRequestMetadata,
        timestamp_ns: int | None = None,
    ) -> tuple[LifecycleRecord, bool]:
        record = self._get_or_create(
            metadata.lifecycle_id,
            agent_id=metadata.agent_id,
            turn_id=metadata.turn_id,
        )
        if record.uid is not None and record.uid != uid:
            raise ValueError(
                f"lifecycle {metadata.lifecycle_id!r} is already associated with uid {record.uid}"
            )
        record.uid = uid
        record.parent_lifecycle_id = metadata.parent_lifecycle_id
        if metadata.tool_name_hint is not None and record.tool_name is None:
            record.tool_name = metadata.tool_name_hint
        self.uid_to_lifecycle[uid] = metadata.lifecycle_id
        self._count(record, "request_received")
        created = self._set_once(
            record,
            "request_received_at_ns",
            timestamp_ns if timestamp_ns is not None else self._clock_ns(),
        )
        return record, created

    def first_token(
        self, uid: int, timestamp_ns: int | None = None
    ) -> tuple[LifecycleRecord | None, bool]:
        lifecycle_id = self.uid_to_lifecycle.get(uid)
        if lifecycle_id is None:
            return None, False
        record = self.records[lifecycle_id]
        self._count(record, "generation_first_token")
        created = self._set_once(
            record,
            "first_token_at_ns",
            timestamp_ns if timestamp_ns is not None else self._clock_ns(),
        )
        return record, created

    def generation_finished(
        self, uid: int, timestamp_ns: int | None = None
    ) -> tuple[LifecycleRecord | None, bool]:
        lifecycle_id = self.uid_to_lifecycle.get(uid)
        if lifecycle_id is None:
            return None, False
        record = self.records[lifecycle_id]
        self._count(record, "generation_finished")
        created = self._set_once(
            record,
            "generation_finished_at_ns",
            timestamp_ns if timestamp_ns is not None else self._clock_ns(),
        )
        return record, created

    def apply(self, event: ToolLifecycleEvent) -> tuple[LifecycleRecord, bool]:
        record = self._get_or_create(
            event.lifecycle_id,
            agent_id=event.agent_id,
            turn_id=event.turn_id,
        )
        self._count(record, event.event)
        if event.tool_name is not None:
            record.tool_name = event.tool_name
        if event.estimated_duration_s is not None:
            record.estimated_duration_s = event.estimated_duration_s

        field_name = {
            "tool_predicted": "tool_predicted_at_ns",
            "tool_started": "tool_started_at_ns",
            "tool_finished": "tool_finished_at_ns",
            "continuation_submitted": "continuation_submitted_at_ns",
            "generation_released": "generation_released_at_ns",
            "abort": "aborted_at_ns",
        }[event.event]
        return record, self._set_once(record, field_name, event.emitted_at_ns)

    def forget_uid(self, uid: int) -> None:
        self.uid_to_lifecycle.pop(uid, None)
