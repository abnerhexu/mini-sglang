from __future__ import annotations

import torch
from minisgl.kvcache.mha_pool import MHAKVCache


def _cpu_cache() -> MHAKVCache:
    cache = MHAKVCache.__new__(MHAKVCache)
    cache._kv_buffer = torch.arange(2 * 2 * 4 * 1 * 1 * 2, dtype=torch.float32).reshape(
        2, 2, 4, 1, 1, 2
    )
    cache._k_buffer = cache._kv_buffer[0]
    cache._v_buffer = cache._kv_buffer[1]
    cache._num_layers = 2
    cache._device = torch.device("cpu")
    cache._storage_shape = (4, 1, 2)
    return cache


def test_mha_cache_copies_complete_pages_round_trip_on_cpu() -> None:
    cache = _cpu_cache()
    host = torch.zeros((2, 2, 2, 1, 1, 2), dtype=torch.float32)
    cache.copy_pages_to(
        host,
        torch.tensor([1, 3], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
    )
    assert torch.equal(host[:, :, 0], cache._kv_buffer[:, :, 1])
    assert torch.equal(host[:, :, 1], cache._kv_buffer[:, :, 3])

    cache._kv_buffer[:, :, 0].zero_()
    cache._kv_buffer[:, :, 2].zero_()
    cache.copy_pages_from(
        host,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
    )
    assert torch.equal(cache._kv_buffer[:, :, 0], host[:, :, 0])
    assert torch.equal(cache._kv_buffer[:, :, 2], host[:, :, 1])
    assert cache.num_pages == 4
    assert cache.bytes_per_page == cache._kv_buffer[:, :, 0].numel() * 4


def test_mha_cache_rejects_invalid_transfer_indices() -> None:
    cache = _cpu_cache()
    host = torch.zeros((2, 2, 1, 1, 1, 2), dtype=torch.float32)
    try:
        cache.copy_pages_to(host, torch.tensor([4]), torch.tensor([0]))
    except IndexError:
        pass
    else:
        raise AssertionError("expected out-of-range source page to fail")
