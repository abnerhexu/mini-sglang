from __future__ import annotations

import asyncio

from minisgl.agentweave import SandboxCoordinator, ToolDefinition
from minisgl.agentweave.prewarm import ResourceMonitor, ToolSchema
from minisgl.agentweave.sandbox import SandboxKey, SandboxSpec, SimulatedSandboxBackend, WarmPool


def test_resource_monitor_caches_os_reads_but_refreshes_live_counts():
    reads = []
    now = [0.0]
    monitor = ResourceMonitor(
        max_sandboxes=4, clock=lambda: now[0], memory_reader=lambda: reads.append(1) or 0.4
    )
    first = monitor.sample(live_sandboxes=1, pool_size=0, pool_capacity=2)
    second = monitor.sample(live_sandboxes=4, pool_size=2, pool_capacity=2)
    assert first.memory_usage == second.memory_usage == 0.4
    assert second.process_usage == second.pool_usage == 1
    assert len(reads) == 1
    now[0] = 1.0
    monitor.sample(live_sandboxes=0, pool_size=0, pool_capacity=0)
    assert len(reads) == 2


def test_coordinator_learns_cold_cost_and_reads_pressure_without_runtime_hints():
    async def scenario():
        memory = [0.0]
        monitor = ResourceMonitor(memory_reader=lambda: memory[0], sample_interval_s=0)
        coordinator = SandboxCoordinator(
            {"simulated": SimulatedSandboxBackend(preparation_delay_s=0.02)},
            WarmPool(),
            resource_monitor=monitor,
        )
        spec = SandboxSpec(SandboxKey("simulated", runtime="python"))
        tools = (ToolDefinition(ToolSchema("python", ("code",), sandbox_type="simulated"), spec),)
        acquisition = await coordinator.claim("first", "python", tools)
        estimate = coordinator.cold_start_estimates[spec.key]
        assert estimate >= 20
        await coordinator.recycle(acquisition)
        assert coordinator.live_sandboxes == 1
        memory[0] = 0.99
        decision = coordinator.observe_delta(
            "next", '{"name":"python","arguments":{"code":"x"}}', tools
        )
        assert not decision.should_prewarm
        assert decision.reason == "resource_capacity"
        assert decision.resource_pressure.memory_usage == 0.99
        assert decision.cold_start_ms == estimate
        pooled = await coordinator.claim("next", "python", tools)
        assert pooled.source == "pool"
        assert coordinator.cold_start_samples[spec.key] == 1
        await coordinator.recycle(pooled)
        await coordinator.close()
        assert coordinator.live_sandboxes == 0

    asyncio.run(scenario())
