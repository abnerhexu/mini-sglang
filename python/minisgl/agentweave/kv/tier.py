from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Hashable

import torch
from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
from minisgl.utils import align_down

from .host_pool import HostKVPagePool

if TYPE_CHECKING:
    from minisgl.scheduler.cache import CacheManager


@dataclass(frozen=True)
class HostPrefixEntry:
    lifecycle_id: str
    host_key: Hashable
    input_ids: torch.Tensor
    num_pages: int

    @property
    def cached_len(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class OffloadResult:
    copied_pages: int
    reclaimed_pages: int
    evicted_host_keys: tuple[Hashable, ...]


@dataclass(frozen=True)
class PrefetchResult:
    restored_pages: int
    cached_len: int
    handle: BaseCacheHandle


class HostKVTransferController:
    """Synchronous correctness path for GPU/CPU KV page migration.

    CUDA stream/event overlap is layered on top of this controller later. Keeping
    the first implementation synchronous makes ownership and failure rollback
    testable on CPU before introducing asynchronous transfer races.
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        kv_cache: BaseKVCachePool,
        host_pool: HostKVPagePool,
        *,
        page_size: int,
    ) -> None:
        self.cache_manager = cache_manager
        self.kv_cache = kv_cache
        self.host_pool = host_pool
        self.page_size = page_size
        self.entries: Dict[str, HostPrefixEntry] = {}
        self.gpu_handles: Dict[str, BaseCacheHandle] = {}

    def offload(
        self,
        lifecycle_id: str,
        input_ids: torch.Tensor,
        handle: BaseCacheHandle,
    ) -> OffloadResult:
        cached_len = align_down(min(handle.cached_len, len(input_ids)), self.page_size)
        if cached_len <= 0:
            return OffloadResult(0, 0, ())
        token_indices = handle.get_matched_indices()[:cached_len]
        gpu_pages = self._token_indices_to_pages(token_indices)
        host_key = lifecycle_id
        slots, evicted = self.host_pool.reserve(host_key, len(gpu_pages), pin=True)
        self._remove_evicted_metadata(evicted)

        self.cache_manager.lock(handle)
        try:
            self.kv_cache.copy_pages_to(
                self.host_pool.buffer,
                gpu_pages,
                torch.tensor(slots, dtype=torch.long),
            )
            self.host_pool.bytes_stored += len(gpu_pages) * self.host_pool.bytes_per_page
        except Exception:
            self.host_pool.remove(host_key, force=True)
            raise
        finally:
            self.cache_manager.unlock(handle)

        self.entries[lifecycle_id] = HostPrefixEntry(
            lifecycle_id,
            host_key,
            input_ids[:cached_len].clone(),
            len(gpu_pages),
        )
        evicted_tokens = self.cache_manager.evict_handle(handle)
        reclaimed_pages = len(evicted_tokens) // self.page_size
        return OffloadResult(len(gpu_pages), reclaimed_pages, evicted)

    def prefetch(self, lifecycle_id: str) -> PrefetchResult:
        entry = self.entries.get(lifecycle_id)
        if entry is None:
            raise KeyError(lifecycle_id)
        host_entry = self.host_pool.entries.get(entry.host_key)
        if host_entry is None:
            raise KeyError(f"host KV entry for {lifecycle_id!r} was evicted")

        current = self.cache_manager.match_prefix(entry.input_ids).cuda_handle
        current_len = align_down(min(current.cached_len, entry.cached_len), self.page_size)
        current_pages = current_len // self.page_size
        missing_pages = entry.num_pages - current_pages
        self.cache_manager.lock(current)
        allocated_starts = torch.empty(
            0,
            dtype=torch.int32,
            device=self.cache_manager.device,
        )
        new_handle: BaseCacheHandle | None = None
        try:
            if missing_pages > 0:
                allocated_starts = self.cache_manager.allocate_page_starts(missing_pages)
                destination_pages = allocated_starts // self.page_size
                source_slots = host_entry.slots[current_pages:]
                self.kv_cache.copy_pages_from(
                    self.host_pool.buffer,
                    torch.tensor(source_slots, dtype=torch.long),
                    destination_pages,
                )
                self.host_pool.bytes_loaded += missing_pages * self.host_pool.bytes_per_page

            existing_indices = current.get_matched_indices()[:current_len]
            restored_indices = self.cache_manager.expand_page_starts(allocated_starts)
            full_indices = torch.cat([existing_indices, restored_indices])
            already_cached, new_handle = self.cache_manager.insert_prefix(
                entry.input_ids,
                full_indices,
            )

            duplicate_pages = max(0, already_cached - current_len) // self.page_size
            if duplicate_pages > 0:
                self.cache_manager.free_page_starts(allocated_starts[:duplicate_pages])
            self.cache_manager.lock(new_handle)
        except Exception:
            if len(allocated_starts) > 0:
                self.cache_manager.free_page_starts(allocated_starts)
            raise
        finally:
            self.cache_manager.unlock(current)

        assert new_handle is not None
        old = self.gpu_handles.pop(lifecycle_id, None)
        if old is not None:
            self.cache_manager.unlock(old)
        self.gpu_handles[lifecycle_id] = new_handle
        return PrefetchResult(missing_pages, new_handle.cached_len, new_handle)

    def release(self, lifecycle_id: str, *, drop_host: bool = True) -> None:
        handle = self.gpu_handles.pop(lifecycle_id, None)
        if handle is not None:
            self.cache_manager.unlock(handle)
        entry = self.entries.pop(lifecycle_id, None)
        if entry is None or not drop_host:
            return
        host_entry = self.host_pool.entries.get(entry.host_key)
        if host_entry is not None and host_entry.pin_count > 0:
            self.host_pool.unpin(entry.host_key)
        self.host_pool.remove(entry.host_key)

    def _token_indices_to_pages(self, token_indices: torch.Tensor) -> torch.Tensor:
        if len(token_indices) % self.page_size != 0:
            raise ValueError("KV token indices are not page aligned")
        starts = token_indices[:: self.page_size]
        if self.page_size > 1:
            offsets = torch.arange(
                self.page_size,
                dtype=token_indices.dtype,
                device=token_indices.device,
            )
            expected = (starts.unsqueeze(1) + offsets).flatten()
            if not torch.equal(token_indices, expected):
                raise ValueError("KV token indices do not map to complete contiguous pages")
        if bool(torch.any(starts % self.page_size != 0)):
            raise ValueError("KV page start is not aligned")
        return starts // self.page_size

    def _remove_evicted_metadata(self, evicted: tuple[Hashable, ...]) -> None:
        if not evicted:
            return
        evicted_set = set(evicted)
        for lifecycle_id, entry in list(self.entries.items()):
            if entry.host_key in evicted_set:
                del self.entries[lifecycle_id]
