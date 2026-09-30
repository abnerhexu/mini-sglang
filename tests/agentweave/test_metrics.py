from __future__ import annotations

import json

from minisgl.agentweave import EventLogger


def test_event_logger_writes_jsonl_with_placeholders(tmp_path) -> None:
    path = tmp_path / "{component}.{rank}.jsonl"
    logger = EventLogger(str(path), component="scheduler", rank=0)
    logger.emit(
        "tool_started",
        lifecycle_id="life-1",
        agent_id="agent-1",
        turn_id=2,
        uid=4,
        timestamp_ns=99,
        tool_name="python",
    )
    logger.close()

    output = tmp_path / "scheduler.0.jsonl"
    record = json.loads(output.read_text())
    assert record["schema_version"] == 1
    assert record["timestamp_ns"] == 99
    assert record["lifecycle_id"] == "life-1"
    assert record["tool_name"] == "python"


def test_disabled_event_logger_is_noop() -> None:
    logger = EventLogger(None, component="scheduler")
    assert not logger.enabled
    logger.emit("ignored")
    logger.close()
