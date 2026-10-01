from __future__ import annotations

import json

from minisgl.agentweave.analysis import load_jsonl, main, summarize_events


def _event(lifecycle_id: str | None, event: str, timestamp_ns: int, **fields):
    return {
        "lifecycle_id": lifecycle_id,
        "event": event,
        "timestamp_ns": timestamp_ns,
        **fields,
    }


def test_event_summary_reports_latency_sandbox_and_transfer_metrics() -> None:
    records = [
        _event("life-1", "request_received", 0),
        _event("life-1", "generation_first_token", 10_000_000),
        _event("life-1", "generation_finished", 20_000_000),
        _event("life-1", "tool_started", 30_000_000),
        _event(
            "life-1",
            "sandbox_acquired",
            35_000_000,
            detail={
                "source": "prewarm",
                "predicted_tool": "python",
                "correct_prediction": True,
            },
        ),
        _event("life-1", "tool_finished", 50_000_000),
        _event("life-1", "continuation_submitted", 52_000_000),
        _event("life-1", "continuation_restore_deferred", 52_100_000),
        _event("life-1", "continuation_restore_ready", 52_200_000),
        _event(
            "life-1",
            "kv_policy_decision",
            31_000_000,
            action="offload",
            implemented=True,
            copied_pages=2,
            reclaimed_pages=1,
            transfer_s=0.1,
        ),
        _event(
            "life-1",
            "kv_policy_decision",
            49_000_000,
            action="prefetch",
            implemented=False,
            restored_pages=2,
            transfer_s=0.2,
        ),
        _event("life-2", "request_received", 0),
        _event("life-2", "generation_first_token", 30_000_000),
        _event(None, "unscoped", 1),
    ]

    summary = summarize_events(records, include_lifecycles=True)
    assert summary["num_lifecycles"] == 2
    assert summary["ignored_records"] == 1
    assert summary["durations"]["ttft_ms"] == {
        "count": 2,
        "mean": 20.0,
        "p50": 10.0,
        "p95": 30.0,
        "p99": 30.0,
        "min": 10.0,
        "max": 30.0,
    }
    assert summary["lifecycles"]["life-1"]["sandbox_wait_ms"] == 5.0
    assert summary["lifecycles"]["life-1"]["tool_execution_ms"] == 15.0
    assert summary["lifecycles"]["life-1"]["continuation_gap_ms"] == 2.0
    assert summary["sandbox_sources"] == {"prewarm": 1}
    assert summary["predictions"] == {
        "correct": 1,
        "incorrect": 0,
        "accuracy": 1.0,
    }
    assert summary["kv_actions"] == {"offload": 1, "prefetch": 1}
    assert summary["continuation_restore"] == {"deferred": 1, "ready": 1}
    assert summary["transfers"] == {
        "d2h_pages": 2,
        "h2d_pages": 2,
        "reclaimed_pages": 1,
        "transfer_s": 0.30000000000000004,
        "failed_actions": 1,
    }


def test_analysis_cli_reads_jsonl_and_writes_summary(tmp_path) -> None:
    log_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "summary.json"
    records = [
        _event("life-1", "request_received", 1),
        _event("life-1", "generation_first_token", 1_000_001),
    ]
    log_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert main([str(log_path), "--output", str(output_path)]) == 0
    assert load_jsonl([log_path]) == records
    output = json.loads(output_path.read_text())
    assert output["durations"]["ttft_ms"]["mean"] == 1.0
