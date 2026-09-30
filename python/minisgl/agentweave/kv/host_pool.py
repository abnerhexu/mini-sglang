from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Sequence

import torch


@dataclass
class HostKVEntry:
    key: Hashable
    slots: tuple[int, ...]
    last_access: float
    pin_count: int = 0

    @property
    def num_pages(self) -> int:
        return len(self.slots)


class HostKVCapacityError(RuntimeError):
    pass


class HostKVPagePool:
    """Fixed-capacity host storage with page-granular tensor copies.

    Buffers use Mini-SGLang's MHA layout:
    ``[K/V, layer, page, page_size, local_kv_head, head_dim]``.
    CPU tensors make this class testable without CUDA; on a CUDA host,
    ``pin_memory=True`` enables asynchronous transfers by the transfer layer.
    """

    def __init__(
        self,
        capacity_pages: int,
        *,
        num_layers: int,
        page_shape: tuple[int, ...],
        dtype: torch.dtype,
        pin_memory: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity_pages < 0:
            raise ValueError("capacity_pages must be non-negative")
        if num_layers <= 0 or any(dimension <= 0 for dimension in page_shape):
            raise ValueError("KV page dimensions must be positive")
        self.capacity_pages = capacity_pages
        self.clock = clock
        use_pinned_memory = pin_memory and torch.cuda.is_available()
        self.buffer = torch.empty(
            (2, num_layers, capacity_pages, *page_shape),
            dtype=dtype,
            device="cpu",
            pin_memory=use_pinned_memory,
        )
        self.entries: Dict[Hashable, HostKVEntry] = {}
        self._free_slots = deque(range(capacity_pages))
        self.bytes_stored = 0
        self.bytes_loaded = 0
        self.evictions = 0

    @property
    def free_pages(self) -> int:
        return len(self._free_slots)

    @property
    def used_pages(self) -> int:
        return self.capacity_pages - self.free_pages

    @property
    def bytes_per_page(self) -> int:
        if self.capacity_pages > 0:
            return self.buffer[:, :, 0].numel() * self.buffer.element_size()
        elements = 1
        for dimension in self.buffer.shape[:2] + self.buffer.shape[3:]:
            elements *= dimension
        return elements * self.buffer.element_size()

    def contains(self, key: Hashable) -> bool:
        return key in self.entries

    def reserve(
        self,
        key: Hashable,
        num_pages: int,
        *,
        pin: bool = False,
    ) -> tuple[tuple[int, ...], tuple[Hashable, ...]]:
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        if num_pages > self.capacity_pages:
            raise HostKVCapacityError(
                f"entry needs {num_pages} pages, host pool capacity is {self.capacity_pages}"
            )
        if key in self.entries:
            self.remove(key, force=True)
        evicted = self._ensure_available(num_pages)
        slots = tuple(self._free_slots.popleft() for _ in range(num_pages))
        self.entries[key] = HostKVEntry(
            key,
            slots,
            self.clock(),
            pin_count=1 if pin else 0,
        )
        return slots, tuple(evicted)

    def store(
        self,
        key: Hashable,
        source: torch.Tensor,
        source_pages: Sequence[int],
        *,
        pin: bool = False,
    ) -> tuple[Hashable, ...]:
        self._validate_tensor(source)
        pages = tuple(int(page) for page in source_pages)
        if any(page < 0 or page >= source.shape[2] for page in pages):
            raise IndexError("source page index is out of range")
        slots, evicted = self.reserve(key, len(pages), pin=pin)
        try:
            if pages:
                source_index = torch.tensor(pages, dtype=torch.long, device=source.device)
                host_index = torch.tensor(slots, dtype=torch.long)
                selected = source.index_select(2, source_index).to("cpu")
                self.buffer.index_copy_(2, host_index, selected)
                self.bytes_stored += len(pages) * self.bytes_per_page
        except Exception:
            self.remove(key, force=True)
            raise
        return evicted

    def load(
        self,
        key: Hashable,
        destination: torch.Tensor,
        destination_pages: Sequence[int],
    ) -> int:
        entry = self.entries.get(key)
        if entry is None:
            raise KeyError(key)
        return self.load_pages(
            key,
            destination,
            source_offsets=range(entry.num_pages),
            destination_pages=destination_pages,
        )

    def load_pages(
        self,
        key: Hashable,
        destination: torch.Tensor,
        *,
        source_offsets: Sequence[int],
        destination_pages: Sequence[int],
    ) -> int:
        self._validate_tensor(destination)
        entry = self.entries.get(key)
        if entry is None:
            raise KeyError(key)
        offsets = tuple(int(offset) for offset in source_offsets)
        pages = tuple(int(page) for page in destination_pages)
        if len(offsets) != len(pages):
            raise ValueError("source and destination page counts must match")
        if any(offset < 0 or offset >= entry.num_pages for offset in offsets):
            raise IndexError("host KV page offset is out of range")
        if any(page < 0 or page >= destination.shape[2] for page in pages):
            raise IndexError("destination page index is out of range")
        if pages:
            slots = tuple(entry.slots[offset] for offset in offsets)
            host_index = torch.tensor(slots, dtype=torch.long)
            destination_index = torch.tensor(pages, dtype=torch.long, device=destination.device)
            selected = self.buffer.index_select(2, host_index).to(destination.device)
            destination.index_copy_(2, destination_index, selected)
            self.bytes_loaded += len(pages) * self.bytes_per_page
        entry.last_access = self.clock()
        return len(pages)

    def pin(self, key: Hashable) -> None:
        entry = self.entries[key]
        entry.pin_count += 1
        entry.last_access = self.clock()

    def unpin(self, key: Hashable) -> None:
        entry = self.entries[key]
        if entry.pin_count <= 0:
            raise RuntimeError("cannot unpin an unpinned host KV entry")
        entry.pin_count -= 1
        entry.last_access = self.clock()

    def remove(self, key: Hashable, *, force: bool = False) -> bool:
        entry = self.entries.get(key)
        if entry is None:
            return False
        if entry.pin_count > 0 and not force:
            return False
        del self.entries[key]
        self._free_slots.extend(entry.slots)
        return True

    def clear(self, *, force: bool = False) -> tuple[Hashable, ...]:
        removed = []
        for key in tuple(self.entries):
            if self.remove(key, force=force):
                removed.append(key)
        return tuple(removed)

    def _ensure_available(self, needed_pages: int) -> list[Hashable]:
        reclaimable = sum(
            entry.num_pages for entry in self.entries.values() if entry.pin_count == 0
        )
        if self.free_pages + reclaimable < needed_pages:
            raise HostKVCapacityError(
                f"need {needed_pages} pages but only "
                f"{self.free_pages + reclaimable} are reclaimable"
            )
        evicted = []
        while self.free_pages < needed_pages:
            candidates = [entry for entry in self.entries.values() if entry.pin_count == 0]
            victim = min(candidates, key=lambda entry: entry.last_access)
            self.remove(victim.key)
            evicted.append(victim.key)
            self.evictions += 1
        return evicted

    def _validate_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != self.buffer.ndim:
            raise ValueError("KV tensor rank does not match host pool")
        expected = self.buffer.shape[:2] + self.buffer.shape[3:]
        actual = tensor.shape[:2] + tensor.shape[3:]
        if actual != expected:
            raise ValueError(f"KV tensor shape mismatch: expected {expected}, got {actual}")
        if tensor.dtype != self.buffer.dtype:
            raise ValueError("KV tensor dtype does not match host pool")
