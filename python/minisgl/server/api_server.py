from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from minisgl.agentweave import (
    AgentMessage,
    AgentRequestMetadata,
    AgentRuntime,
    SandboxCoordinator,
    ToolDefinition,
    ToolLifecycleEvent,
)
from minisgl.agentweave.prewarm import ConfidencePolicy, PrewarmController, ToolSchema
from minisgl.agentweave.sandbox import (
    PythonProcessBackend,
    SandboxKey,
    SandboxSpec,
    SimulatedSandboxBackend,
    WarmPool,
)
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    LifecycleControlMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class AgentMetadataRequest(BaseModel):
    agent_id: str
    turn_id: int = Field(ge=0)
    lifecycle_id: str
    parent_lifecycle_id: str | None = None
    is_continuation: bool = False
    tool_name_hint: str | None = None

    def to_protocol(self) -> AgentRequestMetadata:
        return AgentRequestMetadata(**self.model_dump())


class LifecycleEventRequest(BaseModel):
    lifecycle_id: str
    event: Literal[
        "tool_predicted",
        "tool_started",
        "tool_finished",
        "continuation_submitted",
        "generation_released",
        "abort",
    ]
    agent_id: str | None = None
    turn_id: int | None = Field(default=None, ge=0)
    tool_name: str | None = None
    estimated_duration_s: float | None = Field(default=None, gt=0)
    detail: Dict[str, object] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False
    agentweave: AgentMetadataRequest | None = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AgentToolRequest(BaseModel):
    name: str = Field(min_length=1)
    required_keys: List[str] = Field(default_factory=list)
    sandbox_backend: Literal["python-process", "simulated"] = "python-process"
    runtime: str = "default"
    dependencies: List[str] = Field(default_factory=list)
    preparation_level: str = "runtime"
    environment: Dict[str, str] = Field(default_factory=dict)

    def to_definition(self) -> ToolDefinition:
        key = SandboxKey(
            backend=self.sandbox_backend,
            runtime=self.runtime,
            dependencies=tuple(sorted(self.dependencies)),
            preparation_level=self.preparation_level,
        )
        return ToolDefinition(
            schema=ToolSchema(
                name=self.name,
                required_keys=tuple(self.required_keys),
                sandbox_type=self.sandbox_backend,
                preparation_level=self.preparation_level,
            ),
            sandbox=SandboxSpec(
                key=key,
                environment=tuple(sorted(self.environment.items())),
            ),
        )


class AgentRunRequest(BaseModel):
    model: str
    agent_id: str = Field(min_length=1)
    run_id: str | None = None
    messages: List[Message] = Field(min_length=1)
    tools: List[AgentToolRequest] = Field(min_length=1)
    max_steps: int = Field(default=8, ge=1, le=64)
    max_tokens: int = Field(default=1024, ge=1)


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False
    agentweave: AgentMetadataRequest | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class MiniSGLStreamingGenerator:
    def __init__(self, state: FrontendManager) -> None:
        self.state = state

    async def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
        metadata: AgentRequestMetadata,
        *,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        _ = tools  # Tool declarations are rendered into the system prompt by AgentRuntime.
        uid = self.state.new_user()
        await self.state.send_one(
            TokenizeMsg(
                uid=uid,
                text=[{"role": message.role, "content": message.content} for message in messages],
                sampling_params=SamplingParams(max_tokens=max_tokens),
                agent_metadata=metadata,
            )
        )
        try:
            async for ack in self.state.wait_for_ack(uid):
                if ack.incremental_output:
                    yield ack.incremental_output
                if ack.finished:
                    break
        finally:
            if uid in self.state.ack_map:
                await self.state.abort_user(uid)


class FrontendLifecycleSink:
    def __init__(self, state: FrontendManager) -> None:
        self.state = state

    async def emit(self, event: ToolLifecycleEvent) -> None:
        await self.state.send_one(LifecycleControlMsg(event=event))


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    sandbox_coordinator: SandboxCoordinator | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.config.enable_agentweave_runtime:
            return
        self.sandbox_coordinator = SandboxCoordinator(
            {
                "python-process": PythonProcessBackend(),
                "simulated": SimulatedSandboxBackend(),
            },
            WarmPool(
                max_total=self.config.agentweave_pool_size,
                max_per_key=self.config.agentweave_pool_per_key,
                ttl_s=self.config.agentweave_pool_ttl_s,
            ),
            PrewarmController(
                ConfidencePolicy(base_threshold=self.config.agentweave_prewarm_threshold)
            ),
        )

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    async def close_agentweave(self) -> None:
        if self.sandbox_coordinator is not None:
            await self.sandbox_coordinator.close()

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        await _GLOBAL_STATE.close_agentweave()
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
            agent_metadata=(req.agentweave.to_protocol() if req.agentweave else None),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/agentweave/events", status_code=202)
async def agentweave_event(req: LifecycleEventRequest):
    state = get_global_state()
    event = ToolLifecycleEvent(**req.model_dump())
    await state.send_one(LifecycleControlMsg(event=event))
    return {
        "status": "accepted",
        "lifecycle_id": event.lifecycle_id,
        "event": event.event,
    }


@app.post("/v1/agent/runs")
async def agent_run(req: AgentRunRequest):
    state = get_global_state()
    coordinator = state.sandbox_coordinator
    if coordinator is None:
        raise HTTPException(
            status_code=403,
            detail="AgentWeave runtime is disabled; launch with --enable-agentweave-runtime",
        )
    tools = tuple(tool.to_definition() for tool in req.tools)
    names = [tool.schema.name for tool in tools]
    if len(names) != len(set(names)):
        raise HTTPException(status_code=422, detail="Tool names must be unique")

    runtime = AgentRuntime(
        MiniSGLStreamingGenerator(state),
        coordinator,
        FrontendLifecycleSink(state),
    )
    result = await runtime.run(
        [AgentMessage(message.role, message.content) for message in req.messages],
        tools,
        agent_id=req.agent_id,
        run_id=req.run_id,
        max_steps=req.max_steps,
        max_tokens=req.max_tokens,
    )
    return {
        "id": result.run_id,
        "object": "agent.run",
        "status": result.status,
        "content": result.content,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "messages": [
            {"role": message.role, "content": message.content} for message in result.messages
        ],
    }


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
            agent_metadata=(req.agentweave.to_protocol() if req.agentweave else None),
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    async for ack in state.wait_for_ack(uid):
        full_content += ack.incremental_output
        if ack.finished:
            break

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
            agent_metadata=(req.agentweave.to_protocol() if req.agentweave else None),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        await get_global_state().close_agentweave()
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
