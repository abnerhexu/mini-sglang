from __future__ import annotations

import torch
from minisgl.agentweave import AgentRequestMetadata, ToolLifecycleEvent
from minisgl.core import SamplingParams
from minisgl.message import (
    BaseBackendMsg,
    BaseTokenizerMsg,
    LifecycleBackendMsg,
    LifecycleControlMsg,
    TokenizeMsg,
    UserMsg,
)


def test_request_metadata_round_trip_across_tokenizer_and_backend() -> None:
    metadata = AgentRequestMetadata(
        agent_id="agent-1",
        turn_id=3,
        lifecycle_id="life-3",
        parent_lifecycle_id="life-2",
        is_continuation=True,
        tool_name_hint="python",
    )
    tokenize = TokenizeMsg(
        uid=9,
        text="hello",
        sampling_params=SamplingParams(max_tokens=4),
        agent_metadata=metadata,
    )
    decoded_tokenize = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(tokenize))
    assert decoded_tokenize == tokenize

    user = UserMsg(
        uid=9,
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
        agent_metadata=metadata,
    )
    decoded_user = BaseBackendMsg.decoder(user.encoder())
    assert decoded_user.uid == user.uid
    assert torch.equal(decoded_user.input_ids, user.input_ids)
    assert decoded_user.agent_metadata == metadata


def test_lifecycle_event_round_trip() -> None:
    event = ToolLifecycleEvent(
        lifecycle_id="life-1",
        event="tool_predicted",
        agent_id="agent-1",
        turn_id=1,
        tool_name="bash",
        estimated_duration_s=1.25,
        detail={"confidence": 0.9, "keys": ["cmd"]},
        emitted_at_ns=123,
    )

    control = LifecycleControlMsg(event)
    decoded_control = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(control))
    assert decoded_control == control

    backend = LifecycleBackendMsg(event)
    decoded_backend = BaseBackendMsg.decoder(backend.encoder())
    assert decoded_backend == backend
