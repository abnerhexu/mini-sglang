from __future__ import annotations

import torch
from minisgl.agentweave.kv import HostKVPagePool, HostKVTransferController
from minisgl.kvcache import BaseCacheHandle, MatchResult
from minisgl.kvcache.mha_pool import MHAKVCache


class FakeHandle(BaseCacheHandle):
    def __init__(self, indices: torch.Tensor) -> None:
        super().__init__(cached_len=len(indices))
        object.__setattr__(self, "indices", indices)

    def get_matched_indices(self) -> torch.Tensor:
        return self.indices


class FakeCacheManager:
    page_size = 2
    device = torch.device("cpu")

    def __init__(self, initial: FakeHandle) -> None:
        self.current = initial
        self.locks: dict[int, int] = {}
        self.free_starts = [0, 6]
        self.freed_starts: list[int] = []

    def lock(self, handle: BaseCacheHandle) -> None:
        key = id(handle)
        self.locks[key] = self.locks.get(key, 0) + 1

    def unlock(self, handle: BaseCacheHandle) -> None:
        key = id(handle)
        self.locks[key] -= 1

    def evict_handle(self, handle: BaseCacheHandle) -> torch.Tensor:
        assert self.locks.get(id(handle), 0) == 0
        indices = handle.get_matched_indices()
        self.current = FakeHandle(torch.empty(0, dtype=torch.int32))
        return indices

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        _ = input_ids
        return MatchResult(self.current)

    def allocate_page_starts(self, num_pages: int) -> torch.Tensor:
        starts = self.free_starts[:num_pages]
        self.free_starts = self.free_starts[num_pages:]
        return torch.tensor(starts, dtype=torch.int32)

    def free_page_starts(self, page_starts: torch.Tensor) -> None:
        self.freed_starts.extend(page_starts.tolist())

    def expand_page_starts(self, page_starts: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.page_size, dtype=torch.int32)
        return (page_starts.unsqueeze(1) + offsets).flatten()

    def insert_prefix(
        self, input_ids: torch.Tensor, indices: torch.Tensor
    ) -> tuple[int, BaseCacheHandle]:
        _ = input_ids
        old_len = self.current.cached_len
        self.current = FakeHandle(indices.clone())
        return old_len, self.current


def _kv_cache() -> MHAKVCache:
    cache = MHAKVCache.__new__(MHAKVCache)
    cache._kv_buffer = torch.arange(2 * 1 * 4 * 2, dtype=torch.float32).reshape(2, 1, 4, 2, 1, 1)
    cache._k_buffer = cache._kv_buffer[0]
    cache._v_buffer = cache._kv_buffer[1]
    cache._num_layers = 1
    cache._device = torch.device("cpu")
    cache._storage_shape = (8, 1, 1)
    return cache


def test_host_tier_offload_and_prefetch_round_trip() -> None:
    initial = FakeHandle(torch.tensor([2, 3, 4, 5], dtype=torch.int32))
    manager = FakeCacheManager(initial)
    kv_cache = _kv_cache()
    expected_first = kv_cache._kv_buffer[:, :, 1].clone()
    expected_second = kv_cache._kv_buffer[:, :, 2].clone()
    host = HostKVPagePool(
        4,
        num_layers=1,
        page_shape=(2, 1, 1),
        dtype=torch.float32,
        pin_memory=False,
    )
    controller = HostKVTransferController(manager, kv_cache, host, page_size=2)
    tokens = torch.tensor([10, 11, 12, 13], dtype=torch.int32)

    offload = controller.offload("life-1", tokens, initial)
    assert offload.copied_pages == 2
    assert offload.reclaimed_pages == 2
    assert host.entries["life-1"].pin_count == 1

    kv_cache._kv_buffer[:, :, 1:3].zero_()
    prefetched = controller.prefetch("life-1")
    assert prefetched.restored_pages == 2
    assert torch.equal(kv_cache._kv_buffer[:, :, 0], expected_first)
    assert torch.equal(kv_cache._kv_buffer[:, :, 3], expected_second)
    assert "life-1" in controller.gpu_handles

    controller.release("life-1")
    assert "life-1" not in controller.gpu_handles
    assert "life-1" not in controller.entries
    assert not host.contains("life-1")
