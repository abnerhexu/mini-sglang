from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SandboxKey:
    backend: str
    runtime: str = "default"
    dependencies: tuple[str, ...] = ()
    preparation_level: str = "runtime"
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SandboxSpec:
    key: SandboxKey
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.key.environment != self.environment:
            raise ValueError("sandbox key and specification environments must match")


@dataclass(frozen=True)
class ExecutionResult:
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


class PreparedSandbox(ABC):
    @property
    @abstractmethod
    def key(self) -> SandboxKey: ...

    @abstractmethod
    async def execute(
        self, payload: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> ExecutionResult: ...

    @abstractmethod
    async def reset(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class SandboxBackend(ABC):
    @abstractmethod
    async def prepare(self, spec: SandboxSpec) -> PreparedSandbox: ...
