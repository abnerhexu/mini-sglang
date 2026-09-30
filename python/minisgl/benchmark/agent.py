from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class WorkloadItem:
    index: int
    request: dict[str, Any]
    delay_s: float = 0.0


def load_workload(path: str | Path, *, repeats: int = 1) -> list[WorkloadItem]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    templates: list[tuple[dict[str, Any], float]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            delay_s = value.pop("_delay_s", 0.0)
            if not isinstance(delay_s, (int, float)) or delay_s < 0:
                raise ValueError(f"line {line_number}: _delay_s must be non-negative")
            _validate_request(value, line_number)
            templates.append((value, float(delay_s)))
    if not templates:
        raise ValueError("workload is empty")

    items: list[WorkloadItem] = []
    for repeat in range(repeats):
        for template, delay_s in templates:
            request = json.loads(json.dumps(template))
            if repeats > 1:
                suffix = f"-repeat-{repeat}"
                request["agent_id"] = str(request["agent_id"]) + suffix
                if request.get("run_id"):
                    request["run_id"] = str(request["run_id"]) + suffix
            items.append(WorkloadItem(len(items), request, delay_s))
    return items


def run_workload(
    endpoint: str,
    items: Sequence[WorkloadItem],
    *,
    concurrency: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for item in items:
            if item.delay_s:
                time.sleep(item.delay_s)
            futures.append(executor.submit(_execute_one, endpoint, item, timeout_s))
        results = [future.result() for future in as_completed(futures)]
    return sorted(results, key=lambda result: result["index"])


def _execute_one(
    endpoint: str,
    item: WorkloadItem,
    timeout_s: float,
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    wall_started_ns = time.time_ns()
    status_code = None
    response: dict[str, Any] | None = None
    error = None
    try:
        status_code, response = _post_json(endpoint, item.request, timeout_s)
        if not 200 <= status_code < 300:
            error = f"HTTP {status_code}: {response}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished_ns = time.monotonic_ns()
    success = (
        error is None
        and status_code is not None
        and 200 <= status_code < 300
        and response is not None
    )
    return {
        "index": item.index,
        "agent_id": item.request.get("agent_id"),
        "requested_run_id": item.request.get("run_id"),
        "run_id": response.get("id") if response else None,
        "http_status": status_code,
        "success": success,
        "agent_status": response.get("status") if response else None,
        "steps": response.get("steps") if response else None,
        "tool_calls": response.get("tool_calls") if response else None,
        "latency_ms": (finished_ns - started_ns) / 1_000_000,
        "wall_started_ns": wall_started_ns,
        "error": error,
    }


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            content = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        content = exc.read()
    decoded = json.loads(content) if content else {}
    if not isinstance(decoded, dict):
        raise ValueError("server response must be a JSON object")
    return status, decoded


def _validate_request(request: dict[str, Any], line_number: int) -> None:
    required = ("model", "agent_id", "messages", "tools")
    missing = [key for key in required if key not in request]
    if missing:
        raise ValueError(f"line {line_number}: missing fields {missing}")
    if not isinstance(request["messages"], list) or not request["messages"]:
        raise ValueError(f"line {line_number}: messages must be a non-empty list")
    if not isinstance(request["tools"], list) or not request["tools"]:
        raise ValueError(f"line {line_number}: tools must be a non-empty list")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an AgentWeave JSONL workload")
    parser.add_argument("workload", help="JSONL file containing AgentRunRequest objects")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:1919/v1/agent/runs",
        help="Agent run endpoint.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", required=True, help="Client result JSONL path")
    args = parser.parse_args(argv)

    items = load_workload(args.workload, repeats=args.repeats)
    results = run_workload(
        args.endpoint,
        items,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for result in results:
            destination.write(json.dumps(result, sort_keys=True) + "\n")
    failures = sum(not result["success"] for result in results)
    print(json.dumps({"requests": len(results), "failures": failures}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
