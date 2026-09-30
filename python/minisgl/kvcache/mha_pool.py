from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_pages(self) -> int:
        return self._kv_buffer.shape[2]

    @property
    def bytes_per_page(self) -> int:
        return self._kv_buffer[:, :, 0].numel() * self._kv_buffer.element_size()

    def copy_pages_to(
        self,
        destination: torch.Tensor,
        source_pages: torch.Tensor,
        destination_pages: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None:
        self._validate_transfer(destination, source_pages, destination_pages)
        self._validate_indices(source_pages, self.num_pages, "source")
        self._validate_indices(destination_pages, destination.shape[2], "destination")
        source_indices = source_pages.to(self.device, dtype=torch.long)
        destination_indices = destination_pages.to(destination.device, dtype=torch.long)
        selected = self._kv_buffer.index_select(2, source_indices).to(
            destination.device, non_blocking=non_blocking
        )
        destination.index_copy_(2, destination_indices, selected)

    def copy_pages_from(
        self,
        source: torch.Tensor,
        source_pages: torch.Tensor,
        destination_pages: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None:
        self._validate_transfer(source, source_pages, destination_pages)
        self._validate_indices(source_pages, source.shape[2], "source")
        self._validate_indices(destination_pages, self.num_pages, "destination")
        source_indices = source_pages.to(source.device, dtype=torch.long)
        destination_indices = destination_pages.to(self.device, dtype=torch.long)
        selected = source.index_select(2, source_indices).to(self.device, non_blocking=non_blocking)
        self._kv_buffer.index_copy_(2, destination_indices, selected)

    def _validate_transfer(
        self,
        other: torch.Tensor,
        source_pages: torch.Tensor,
        destination_pages: torch.Tensor,
    ) -> None:
        if other.ndim != self._kv_buffer.ndim:
            raise ValueError("KV transfer tensor rank mismatch")
        expected = self._kv_buffer.shape[:2] + self._kv_buffer.shape[3:]
        actual = other.shape[:2] + other.shape[3:]
        if actual != expected or other.dtype != self.dtype:
            raise ValueError("KV transfer tensor shape or dtype mismatch")
        if source_pages.ndim != 1 or destination_pages.ndim != 1:
            raise ValueError("KV page indices must be one-dimensional")
        if len(source_pages) != len(destination_pages):
            raise ValueError("source and destination page counts must match")

    @staticmethod
    def _validate_indices(indices: torch.Tensor, num_pages: int, label: str) -> None:
        if len(indices) == 0:
            return
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= num_pages:
            raise IndexError(f"{label} KV page index is out of range")
