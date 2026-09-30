from __future__ import annotations

import json

from minisgl.benchmark import agent


def _request(run_id: str = "run-1") -> dict:
    return {
        "model": "test-model",
        "agent_id": "agent-1",
        "run_id": run_id,
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"name": "lookup", "sandbox_backend": "simulated"}],
    }


def test_load_agent_workload_repeats_with_unique_ids(tmp_path) -> None:
    path = tmp_path / "workload.jsonl"
    record = {**_request(), "_delay_s": 0.25}
    path.write_text(json.dumps(record) + "\n")

    items = agent.load_workload(path, repeats=2)
    assert [item.index for item in items] == [0, 1]
    assert [item.delay_s for item in items] == [0.25, 0.25]
    assert items[0].request["run_id"] == "run-1-repeat-0"
    assert items[1].request["agent_id"] == "agent-1-repeat-1"
    assert "_delay_s" not in items[0].request


def test_run_agent_workload_records_success_and_http_failure(monkeypatch) -> None:
    def fake_post(endpoint, payload, timeout_s):
        _ = endpoint, timeout_s
        if payload["run_id"] == "ok":
            return 200, {"id": "ok", "status": "completed", "steps": 2, "tool_calls": 1}
        return 503, {"detail": "busy"}

    monkeypatch.setattr(agent, "_post_json", fake_post)
    items = [
        agent.WorkloadItem(0, _request("ok")),
        agent.WorkloadItem(1, _request("failed")),
    ]
    results = agent.run_workload("http://server", items, concurrency=2, timeout_s=1)

    assert results[0]["success"] is True
    assert results[0]["tool_calls"] == 1
    assert results[1]["success"] is False
    assert results[1]["http_status"] == 503
    assert results[1]["error"].startswith("HTTP 503")
