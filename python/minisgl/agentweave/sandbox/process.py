from __future__ import annotations

import asyncio
import contextlib
import io
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Mapping

from .base import ExecutionResult, PreparedSandbox, SandboxBackend, SandboxKey, SandboxSpec


def _python_worker(connection: Connection, workdir: str, environment: dict[str, str]) -> None:
    os.chdir(workdir)
    os.environ.update(environment)
    clean_environment = dict(os.environ)
    namespace: dict[str, Any] = {"__name__": "__agentweave_sandbox__"}
    connection.send({"ready": True})
    try:
        while True:
            command = connection.recv()
            operation = command.get("operation")
            if operation == "close":
                return
            if operation == "reset":
                namespace = {"__name__": "__agentweave_sandbox__"}
                os.environ.clear()
                os.environ.update(clean_environment)
                for entry in Path(workdir).iterdir():
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink(missing_ok=True)
                connection.send({"ok": True})
                continue
            if operation != "execute":
                connection.send({"ok": False, "error": f"unknown operation: {operation}"})
                continue

            stdout = io.StringIO()
            stderr = io.StringIO()
            started = time.monotonic()
            try:
                code = command["code"]
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(code, namespace, namespace)
                result_name = command.get("result_name")
                result = namespace.get(result_name) if result_name else None
                connection.send(
                    {
                        "ok": True,
                        "output": result,
                        "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue(),
                        "duration_s": time.monotonic() - started,
                    }
                )
            except BaseException:
                connection.send(
                    {
                        "ok": False,
                        "error": traceback.format_exc(),
                        "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue(),
                        "duration_s": time.monotonic() - started,
                    }
                )
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class PythonProcessSandbox(PreparedSandbox):
    """Persistent Python worker used by the research prototype.

    This isolates process state but is not a security boundary. Untrusted code
    requires a container, microVM, or Wasm backend with explicit restrictions.
    """

    def __init__(
        self,
        key: SandboxKey,
        process: mp.Process,
        connection: Connection,
        workdir: tempfile.TemporaryDirectory[str],
    ) -> None:
        self._key = key
        self._process = process
        self._connection = connection
        self._workdir = workdir
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def key(self) -> SandboxKey:
        return self._key

    @property
    def workdir(self) -> Path:
        return Path(self._workdir.name)

    async def execute(
        self, payload: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> ExecutionResult:
        code = payload.get("code")
        if not isinstance(code, str):
            raise ValueError("Python process sandbox requires a string 'code' field")
        command = {
            "operation": "execute",
            "code": code,
            "result_name": payload.get("result_name"),
        }
        response = await self._request(command, timeout_s)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "sandbox execution failed"))
        return ExecutionResult(
            output=response.get("output"),
            stdout=response.get("stdout", ""),
            stderr=response.get("stderr", ""),
            duration_s=response.get("duration_s", 0.0),
        )

    async def reset(self) -> None:
        response = await self._request({"operation": "reset"}, 5.0)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "sandbox reset failed"))

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.is_alive():
                try:
                    self._connection.send({"operation": "close"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                await asyncio.to_thread(self._process.join, 1.0)
            if self._process.is_alive():
                self._process.terminate()
                await asyncio.to_thread(self._process.join, 1.0)
            self._connection.close()
            self._workdir.cleanup()

    async def _request(self, command: dict[str, Any], timeout_s: float | None) -> dict[str, Any]:
        async with self._lock:
            if self._closed or not self._process.is_alive():
                raise RuntimeError("sandbox is closed")

            def round_trip() -> dict[str, Any]:
                self._connection.send(command)
                if timeout_s is not None and not self._connection.poll(timeout_s):
                    raise TimeoutError("sandbox operation timed out")
                response = self._connection.recv()
                if not isinstance(response, dict):
                    raise RuntimeError("invalid sandbox response")
                return response

            try:
                return await asyncio.to_thread(round_trip)
            except TimeoutError:
                self._process.terminate()
                await asyncio.to_thread(self._process.join, 1.0)
                self._connection.close()
                self._workdir.cleanup()
                self._closed = True
                raise


class PythonProcessBackend(SandboxBackend):
    def __init__(self, *, start_method: str = "spawn", startup_timeout_s: float = 10.0) -> None:
        self.context = mp.get_context(start_method)
        self.startup_timeout_s = startup_timeout_s

    async def prepare(self, spec: SandboxSpec) -> PreparedSandbox:
        if spec.key.backend != "python-process":
            raise ValueError(f"unsupported sandbox backend {spec.key.backend!r}")
        parent, child = self.context.Pipe()
        workdir = tempfile.TemporaryDirectory(prefix="agentweave-")
        process = self.context.Process(
            target=_python_worker,
            args=(child, workdir.name, dict(spec.environment)),
            daemon=True,
            name="agentweave-python-sandbox",
        )
        process.start()
        child.close()

        def wait_until_ready() -> None:
            if not parent.poll(self.startup_timeout_s):
                raise TimeoutError("sandbox startup timed out")
            message = parent.recv()
            if message != {"ready": True}:
                raise RuntimeError(f"invalid sandbox startup response: {message!r}")

        try:
            await asyncio.to_thread(wait_until_ready)
        except Exception:
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, 1.0)
            parent.close()
            workdir.cleanup()
            raise
        return PythonProcessSandbox(spec.key, process, parent, workdir)
