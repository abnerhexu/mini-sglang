from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    agentweave_event_log: str | None = None
    agentweave_kv_policy: str = "no-control"
    agentweave_host_kv_pages: int = 0
    agentweave_d2h_bandwidth_gbps: float = 12.0
    agentweave_h2d_bandwidth_gbps: float = 12.0
    agentweave_d2h_base_ms: float = 0.1
    agentweave_h2d_base_ms: float = 0.1
    agentweave_transfer_safety_margin_ms: float = 2.0
    agentweave_transfer_calibration_alpha: float = 0.2

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
