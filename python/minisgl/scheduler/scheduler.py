from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.agentweave import EventLogger, LifecycleRegistry, ToolLifecycleEvent
from minisgl.agentweave.kv import (
    KVAction,
    KVCacheDescriptor,
    KVPressure,
    ToolKVPolicy,
    create_kv_policy,
)
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    LifecycleBackendMsg,
    UserMsg,
)
from minisgl.utils import div_ceil, init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput
    from minisgl.kvcache import BaseCacheHandle


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.agent_lifecycles = LifecycleRegistry()
        self.agent_kv_policy_name = config.agentweave_kv_policy
        self.agent_kv_policy: ToolKVPolicy = create_kv_policy(config.agentweave_kv_policy)
        self.agent_cache_handles: Dict[str, BaseCacheHandle] = {}
        self.agent_cache_tokens: Dict[str, torch.Tensor] = {}
        self.agent_cache_locks: Set[str] = set()
        self.agent_kv_started: Set[str] = set()
        self.agent_events = EventLogger(
            config.agentweave_event_log if config.tp_info.is_primary() else None,
            component="scheduler",
            rank=config.tp_info.rank,
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)
        self._process_kv_policy_tick()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)
        self._process_kv_policy_tick()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        self.agent_events.close()
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if batch.is_prefill:
                    record, created = self.agent_lifecycles.first_token(req.uid)
                    if record is not None and created:
                        self.agent_events.emit(
                            "generation_first_token",
                            lifecycle_id=record.lifecycle_id,
                            agent_id=record.agent_id,
                            turn_id=record.turn_id,
                            uid=req.uid,
                        )
                if finished:
                    record, created = self.agent_lifecycles.generation_finished(req.uid)
                    if record is not None and created:
                        self.agent_events.emit(
                            "generation_finished",
                            lifecycle_id=record.lifecycle_id,
                            agent_id=record.agent_id,
                            turn_id=record.turn_id,
                            uid=req.uid,
                        )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    cache_handle = self._free_req_resources(req)
                    self._register_agent_cache(req, cache_handle)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            if msg.agent_metadata is not None:
                record, created = self.agent_lifecycles.request_received(
                    msg.uid, msg.agent_metadata
                )
                self.agent_events.emit(
                    "request_received",
                    lifecycle_id=record.lifecycle_id,
                    agent_id=record.agent_id,
                    turn_id=record.turn_id,
                    uid=msg.uid,
                    duplicate=not created,
                    parent_lifecycle_id=record.parent_lifecycle_id,
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, LifecycleBackendMsg):
            record, created = self.agent_lifecycles.apply(msg.event)
            self.agent_events.emit(
                msg.event.event,
                lifecycle_id=record.lifecycle_id,
                agent_id=record.agent_id,
                turn_id=record.turn_id,
                timestamp_ns=msg.event.emitted_at_ns,
                duplicate=not created,
                tool_name=record.tool_name,
                estimated_duration_s=record.estimated_duration_s,
                detail=msg.event.detail,
            )
            if created and msg.event.event == "tool_started":
                self._start_agent_kv_policy(record.lifecycle_id)
            elif created and msg.event.event == "tool_finished":
                action = self.agent_kv_policy.on_tool_finished(record.lifecycle_id)
                self._apply_agent_kv_action(action)
            elif created and msg.event.event == "abort":
                action = self.agent_kv_policy.on_abort(record.lifecycle_id)
                self._apply_agent_kv_action(action)
            elif created and msg.event.event in (
                "continuation_submitted",
                "generation_released",
            ):
                if record.lifecycle_id in self.agent_cache_locks:
                    action = self.agent_kv_policy.on_tool_finished(record.lifecycle_id)
                    self._apply_agent_kv_action(action)
                else:
                    self.agent_cache_handles.pop(record.lifecycle_id, None)
                    self.agent_cache_tokens.pop(record.lifecycle_id, None)
                    self.agent_kv_started.discard(record.lifecycle_id)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
            lifecycle_id = self.agent_lifecycles.uid_to_lifecycle.get(msg.uid)
            if lifecycle_id is not None:
                record, created = self.agent_lifecycles.apply(
                    ToolLifecycleEvent(lifecycle_id=lifecycle_id, event="abort")
                )
                if created:
                    self.agent_events.emit(
                        "abort",
                        lifecycle_id=record.lifecycle_id,
                        agent_id=record.agent_id,
                        turn_id=record.turn_id,
                        uid=msg.uid,
                    )
                    action = self.agent_kv_policy.on_abort(record.lifecycle_id)
                    self._apply_agent_kv_action(action)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> BaseCacheHandle:
        self.table_manager.free(req.table_idx)
        return self.cache_manager.cache_req(req, finished=True)

    def _register_agent_cache(self, req: Req, handle: BaseCacheHandle) -> None:
        metadata = req.agent_metadata
        if metadata is None or handle.cached_len <= 0 or self.agent_kv_policy_name == "no-control":
            return
        lifecycle_id = metadata.lifecycle_id
        record = self.agent_lifecycles.records.get(lifecycle_id)
        if record is not None and record.tool_finished_at_ns is not None:
            return
        self.agent_cache_tokens[lifecycle_id] = req.input_ids[: handle.cached_len].clone()
        if record is not None and record.tool_started_at_ns is not None:
            self._start_agent_kv_policy(lifecycle_id)

    def _start_agent_kv_policy(self, lifecycle_id: str) -> None:
        if lifecycle_id in self.agent_kv_started:
            return
        tokens = self.agent_cache_tokens.get(lifecycle_id)
        record = self.agent_lifecycles.records.get(lifecycle_id)
        if tokens is None or record is None:
            return
        handle = self.cache_manager.match_prefix(tokens).cuda_handle
        if handle.cached_len <= 0:
            self.agent_cache_tokens.pop(lifecycle_id, None)
            return
        self.agent_cache_handles[lifecycle_id] = handle
        descriptor = KVCacheDescriptor(
            lifecycle_id=lifecycle_id,
            agent_id=record.agent_id or "",
            tool_name=record.tool_name or "unknown",
            num_pages=div_ceil(handle.cached_len, self.cache_manager.page_size),
            bytes_per_page=self.engine.kv_cache.bytes_per_page,
        )
        action = self.agent_kv_policy.on_tool_started(
            descriptor,
            self._agent_kv_pressure(),
            estimated_duration_s=record.estimated_duration_s,
        )
        self.agent_kv_started.add(lifecycle_id)
        self._apply_agent_kv_action(action)
        if action.action in ("none", "offload") and lifecycle_id not in self.agent_cache_locks:
            # Offload execution is enabled when the host-tier controller is attached.
            # Until then, never pin GPU memory under the TokenCake label.
            self.agent_cache_handles.pop(lifecycle_id, None)
            self.agent_cache_tokens.pop(lifecycle_id, None)

    def _agent_kv_pressure(self) -> KVPressure:
        free_pages = len(self.cache_manager.free_slots)
        waiting_pages = sum(
            div_ceil(req.input_len + req.output_len, self.cache_manager.page_size)
            for req in self.prefill_manager.pending_list
        )
        gpu_usage = 1.0 - free_pages / max(1, self.cache_manager.num_pages)
        return KVPressure(
            gpu_usage=min(1.0, max(0.0, gpu_usage)),
            free_pages=free_pages,
            waiting_pages=waiting_pages,
            host_free_pages=0,
        )

    def _process_kv_policy_tick(self) -> None:
        for action in self.agent_kv_policy.on_tick():
            self._apply_agent_kv_action(action)

    def _apply_agent_kv_action(self, action: KVAction) -> None:
        lifecycle_id = action.lifecycle_id
        handle = self.agent_cache_handles.get(lifecycle_id)
        implemented = action.action in ("none", "retain", "release")
        if action.action == "retain" and handle is not None:
            if lifecycle_id not in self.agent_cache_locks:
                self.cache_manager.lock(handle)
                self.agent_cache_locks.add(lifecycle_id)
        elif action.action == "release":
            if lifecycle_id in self.agent_cache_locks and handle is not None:
                self.cache_manager.unlock(handle)
                self.agent_cache_locks.remove(lifecycle_id)
            self.agent_cache_handles.pop(lifecycle_id, None)
            self.agent_cache_tokens.pop(lifecycle_id, None)
            self.agent_kv_started.discard(lifecycle_id)
        elif action.action in ("offload", "prefetch"):
            implemented = False

        record = self.agent_lifecycles.records.get(lifecycle_id)
        self.agent_events.emit(
            "kv_policy_decision",
            lifecycle_id=lifecycle_id,
            agent_id=record.agent_id if record is not None else None,
            turn_id=record.turn_id if record is not None else None,
            policy=self.agent_kv_policy_name,
            action=action.action,
            reason=action.reason,
            execute_at=action.execute_at,
            estimated_duration_s=action.estimated_duration_s,
            implemented=implemented,
        )

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
