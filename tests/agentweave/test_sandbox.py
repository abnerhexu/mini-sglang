from __future__ import annotations

import asyncio

from minisgl.agentweave import SandboxCoordinator, ToolDefinition
from minisgl.agentweave.prewarm import ConfidencePolicy, PrewarmController, ToolSchema
from minisgl.agentweave.sandbox import (
    PythonProcessBackend,
    SandboxKey,
    SandboxSpec,
    SimulatedSandboxBackend,
    WarmPool,
)


def test_warm_pool_reuses_and_expires_sandboxes() -> None:
    async def scenario() -> None:
        now = [0.0]
        key = SandboxKey("simulated")
        backend = SimulatedSandboxBackend()
        sandbox = await backend.prepare(SandboxSpec(key))
        pool = WarmPool(max_total=2, max_per_key=1, ttl_s=5, clock=lambda: now[0])

        assert await pool.release(sandbox)
        assert await pool.acquire(key) is sandbox
        assert pool.stats().hits == 1
        assert sandbox.reset_count == 1  # type: ignore[attr-defined]

        assert await pool.release(sandbox)
        now[0] = 6.0
        assert await pool.discard_expired() == 1
        assert sandbox.closed  # type: ignore[attr-defined]
        assert pool.stats().size == 0

    asyncio.run(scenario())


def test_coordinator_prewarms_and_recycles_wrong_prediction() -> None:
    async def scenario() -> None:
        backend = SimulatedSandboxBackend(preparation_delay_s=0.01)
        pool = WarmPool(max_total=4, max_per_key=2)
        coordinator = SandboxCoordinator(
            {"simulated": backend},
            pool,
            PrewarmController(ConfidencePolicy(base_threshold=0.6)),
        )
        python = ToolDefinition(
            ToolSchema("python", ("code",), sandbox_type="simulated"),
            SandboxSpec(SandboxKey("simulated", runtime="python")),
        )
        bash = ToolDefinition(
            ToolSchema("bash", ("command",), sandbox_type="simulated"),
            SandboxSpec(SandboxKey("simulated", runtime="bash")),
        )
        tools = (python, bash)

        decision = coordinator.observe_delta(
            "life-1",
            '{"name":"python","arguments":{"code":"x=1"}}',
            tools,
        )
        assert decision.should_prewarm
        await asyncio.sleep(0.02)
        acquisition = await coordinator.claim("life-1", "python", tools)
        assert acquisition.source == "prewarm"
        assert acquisition.correct_prediction
        await coordinator.recycle(acquisition)

        coordinator.observe_delta(
            "life-2",
            '{"name":"python","arguments":{"code":"x=1"}}',
            tools,
        )
        await asyncio.sleep(0.02)
        wrong = await coordinator.claim("life-2", "bash", tools)
        assert wrong.predicted_tool == "python"
        assert not wrong.correct_prediction
        assert wrong.source == "cold"
        assert pool.stats().size == 1
        await coordinator.recycle(wrong)
        await coordinator.close()

    asyncio.run(scenario())


def test_python_process_sandbox_is_persistent_and_resettable() -> None:
    async def scenario() -> None:
        backend = PythonProcessBackend(startup_timeout_s=10)
        spec = SandboxSpec(SandboxKey("python-process", runtime="python"))
        sandbox = await backend.prepare(spec)
        first = await sandbox.execute(
            {
                "code": "x = 4\nopen('artifact.txt', 'w').write('data')\nresult = x * 2",
                "result_name": "result",
            },
            timeout_s=5,
        )
        assert first.output == 8

        await sandbox.reset()
        second = await sandbox.execute(
            {
                "code": (
                    "import os\n" "result = ('x' in globals(), os.path.exists('artifact.txt'))"
                ),
                "result_name": "result",
            },
            timeout_s=5,
        )
        assert second.output == (False, False)
        await sandbox.close()

    asyncio.run(scenario())
