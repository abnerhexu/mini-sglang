from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class EventLogger:
    """Append-only JSONL event logger used by AgentWeave experiments."""

    def __init__(self, path: str | None, *, component: str, rank: int | None = None) -> None:
        self.component = component
        self.rank = rank
        self._lock = threading.Lock()
        self._file = None
        self.path: Path | None = None
        if path is None:
            return

        resolved = path.format(rank=rank if rank is not None else 0, component=component)
        self.path = Path(resolved)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def emit(
        self,
        event: str,
        *,
        lifecycle_id: str | None = None,
        agent_id: str | None = None,
        turn_id: int | None = None,
        uid: int | None = None,
        timestamp_ns: int | None = None,
        **fields: Any,
    ) -> None:
        if self._file is None:
            return
        record = {
            "schema_version": 1,
            "event": event,
            "component": self.component,
            "rank": self.rank,
            "pid": os.getpid(),
            "timestamp_ns": timestamp_ns if timestamp_ns is not None else time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
            "lifecycle_id": lifecycle_id,
            "agent_id": agent_id,
            "turn_id": turn_id,
            "uid": uid,
            **fields,
        }
        with self._lock:
            assert self._file is not None
            self._file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
