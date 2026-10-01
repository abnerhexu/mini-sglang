from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

_DURATION_PAIRS = {
    "ttft_ms": ("request_received", "generation_first_token"),
    "generation_ms": ("request_received", "generation_finished"),
    "sandbox_wait_ms": ("tool_started", "sandbox_acquired"),
    "tool_execution_ms": ("sandbox_acquired", "tool_finished"),
    "tool_window_ms": ("tool_started", "tool_finished"),
    "continuation_gap_ms": ("tool_finished", "continuation_submitted"),
}


def load_jsonl(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                records.append(record)
    return records


def summarize_events(
    records: Iterable[Mapping[str, Any]],
    *,
    include_lifecycles: bool = False,
) -> dict[str, Any]:
    by_lifecycle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ignored = 0
    for record in records:
        lifecycle_id = record.get("lifecycle_id")
        if not isinstance(lifecycle_id, str) or not lifecycle_id:
            ignored += 1
            continue
        by_lifecycle[lifecycle_id].append(record)

    lifecycle_metrics: dict[str, dict[str, Any]] = {}
    durations: dict[str, list[float]] = defaultdict(list)
    sandbox_sources: Counter[str] = Counter()
    predictions: Counter[str] = Counter()
    kv_actions: Counter[str] = Counter()
    continuation_restore: Counter[str] = Counter()
    transfer_totals = {
        "d2h_pages": 0,
        "h2d_pages": 0,
        "reclaimed_pages": 0,
        "transfer_s": 0.0,
        "failed_actions": 0,
    }

    for lifecycle_id, events in by_lifecycle.items():
        metrics = _summarize_lifecycle(events)
        lifecycle_metrics[lifecycle_id] = metrics
        for name in _DURATION_PAIRS:
            value = metrics.get(name)
            if isinstance(value, (int, float)):
                durations[name].append(float(value))
        source = metrics.get("sandbox_source")
        if isinstance(source, str):
            sandbox_sources[source] += 1
        prediction_correct = metrics.get("prediction_correct")
        if metrics.get("prediction_made") is True and isinstance(prediction_correct, bool):
            predictions["correct" if prediction_correct else "incorrect"] += 1

        for event in events:
            event_name = event.get("event")
            if isinstance(event_name, str) and event_name.startswith("continuation_restore_"):
                continuation_restore[event_name.removeprefix("continuation_restore_")] += 1
            if event_name != "kv_policy_decision":
                continue
            action = event.get("action")
            if isinstance(action, str):
                kv_actions[action] += 1
            if event.get("implemented") is False and action in ("offload", "prefetch"):
                transfer_totals["failed_actions"] += 1
            transfer_totals["d2h_pages"] += _nonnegative_int(event.get("copied_pages"))
            transfer_totals["h2d_pages"] += _nonnegative_int(event.get("restored_pages"))
            transfer_totals["reclaimed_pages"] += _nonnegative_int(event.get("reclaimed_pages"))
            transfer_s = event.get("transfer_s")
            if isinstance(transfer_s, (int, float)) and transfer_s >= 0:
                transfer_totals["transfer_s"] += float(transfer_s)

    result: dict[str, Any] = {
        "num_lifecycles": len(by_lifecycle),
        "ignored_records": ignored,
        "durations": {name: _distribution(durations[name]) for name in _DURATION_PAIRS},
        "sandbox_sources": dict(sorted(sandbox_sources.items())),
        "predictions": {
            "correct": predictions["correct"],
            "incorrect": predictions["incorrect"],
            "accuracy": (
                predictions["correct"] / predictions.total() if predictions.total() else None
            ),
        },
        "kv_actions": dict(sorted(kv_actions.items())),
        "continuation_restore": dict(sorted(continuation_restore.items())),
        "transfers": transfer_totals,
    }
    if include_lifecycles:
        result["lifecycles"] = lifecycle_metrics
    return result


def _summarize_lifecycle(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timestamps: dict[str, int] = {}
    sandbox_source = None
    prediction_correct = None
    prediction_made = False
    for event in events:
        name = event.get("event")
        timestamp = event.get("timestamp_ns")
        if isinstance(name, str) and isinstance(timestamp, int):
            previous = timestamps.get(name)
            timestamps[name] = timestamp if previous is None else min(previous, timestamp)
        if name == "sandbox_acquired":
            detail = event.get("detail")
            if isinstance(detail, Mapping):
                source = detail.get("source")
                if isinstance(source, str):
                    sandbox_source = source
                predicted_tool = detail.get("predicted_tool")
                prediction_made = isinstance(predicted_tool, str) and bool(predicted_tool)
                correct = detail.get("correct_prediction")
                if isinstance(correct, bool):
                    prediction_correct = correct

    result: dict[str, Any] = {}
    for metric, (start_event, end_event) in _DURATION_PAIRS.items():
        start = timestamps.get(start_event)
        end = timestamps.get(end_event)
        if start is not None and end is not None and end >= start:
            result[metric] = (end - start) / 1_000_000
    if sandbox_source is not None:
        result["sandbox_source"] = sandbox_source
    result["prediction_made"] = prediction_made
    if prediction_correct is not None:
        result["prediction_correct"] = prediction_correct
    return result


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "mean": fmean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _nonnegative_int(value: object) -> int:
    return int(value) if isinstance(value, int) and value >= 0 else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize AgentWeave JSONL event logs")
    parser.add_argument("logs", nargs="+", help="JSONL event log paths")
    parser.add_argument(
        "--include-lifecycles",
        action="store_true",
        help="Include per-lifecycle timing records in the JSON output.",
    )
    parser.add_argument("--output", help="Write JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    summary = summarize_events(
        load_jsonl(args.logs),
        include_lifecycles=args.include_lifecycles,
    )
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
