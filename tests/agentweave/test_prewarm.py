from __future__ import annotations

from minisgl.agentweave.prewarm import (
    ConfidencePolicy,
    PrewarmController,
    ResourcePressure,
    StreamingToolCallParser,
    ToolSchema,
    ToolTransitionHistory,
)


def test_streaming_parser_and_confidence_decision() -> None:
    schemas = (
        ToolSchema(
            name="python",
            required_keys=("code",),
            sandbox_type="python-process",
        ),
        ToolSchema(name="bash", required_keys=("command",), sandbox_type="python-process"),
    )
    parser = StreamingToolCallParser()
    first = parser.feed('{"name":"py', schemas)
    assert first.name_prefix == "py"
    assert first.candidate_names == ("python",)
    assert first.exact_name is None

    evidence = parser.feed('thon","arguments":{"code":"print(1)"}}', schemas)
    assert evidence.exact_name == "python"
    assert evidence.observed_keys == ("code",)
    assert evidence.complete

    decision = ConfidencePolicy().decide(evidence, schemas, cold_start_ms=100)
    assert decision.should_prewarm
    assert decision.tool_name == "python"
    assert decision.confidence >= 0.8


def test_resource_pressure_raises_adaptive_threshold() -> None:
    schema = (ToolSchema(name="python", required_keys=("code",)),)
    evidence = StreamingToolCallParser().feed('{"name":"python","arguments":{', schema)
    policy = ConfidencePolicy(base_threshold=0.62, max_pressure_penalty=0.3)

    normal = policy.decide(evidence, schema, pressure=ResourcePressure())
    pressured = policy.decide(
        evidence,
        schema,
        pressure=ResourcePressure(process_usage=1.0),
    )
    assert normal.should_prewarm
    assert not pressured.should_prewarm
    assert pressured.reason == "resource_pressure"


def test_transition_history_breaks_ambiguous_prefix_tie() -> None:
    schemas = (ToolSchema(name="git_status"), ToolSchema(name="git_diff"))
    history = ToolTransitionHistory()
    for _ in range(4):
        history.observe("bash", "git_diff")
    history.observe("bash", "git_status")
    evidence = StreamingToolCallParser().feed('{"name":"git_', schemas)

    decision = ConfidencePolicy(base_threshold=0.0).decide(
        evidence,
        schemas,
        previous_tool="bash",
        transition_history=history,
    )
    assert decision.tool_name == "git_diff"


def test_controller_learns_actual_transition_and_clears_state() -> None:
    controller = PrewarmController(ConfidencePolicy(base_threshold=0.0))
    schemas = (ToolSchema(name="python"),)
    controller.feed("life-1", '{"name":"python"', schemas, previous_tool="bash")
    selected = controller.finish("life-1", actual_tool="python", previous_tool="bash")
    assert selected == "python"
    assert "life-1" not in controller.states
    assert controller.history.probability("bash", "python") is not None
