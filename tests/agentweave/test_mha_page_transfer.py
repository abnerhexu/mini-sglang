from __future__ import annotations

import pytest
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mha_cache_copies_pages_between_cuda_and_pinned_host() -> None:
    cache = _cpu_cache()
    cache._kv_buffer = cache._kv_buffer.cuda()
    cache._k_buffer = cache._kv_buffer[0]
    cache._v_buffer = cache._kv_buffer[1]
    cache._device = torch.device("cuda")
    host = torch.zeros(
        (2, 2, 2, 1, 1, 2),
        dtype=torch.float32,
        pin_memory=True,
    )
    expected = cache._kv_buffer[:, :, [1, 3]].cpu()

    cache.copy_pages_to(host, torch.tensor([1, 3]), torch.tensor([0, 1]))
    assert torch.equal(host, expected)
    cache._kv_buffer.index_fill_(2, torch.tensor([0, 2], device="cuda"), 0)
    cache.copy_pages_from(host, torch.tensor([0, 1]), torch.tensor([0, 2]))
    torch.cuda.synchronize()
    assert torch.equal(cache._kv_buffer[:, :, [0, 2]].cpu(), expected)


def test_mha_cache_rejects_invalid_transfer_indices() -> None:
    cache = _cpu_cache()
    host = torch.zeros((2, 2, 1, 1, 1, 2), dtype=torch.float32)
    try:
        cache.copy_pages_to(host, torch.tensor([4]), torch.tensor([0]))
    except IndexError:
        pass
    else:
        raise AssertionError("expected out-of-range source page to fail")
