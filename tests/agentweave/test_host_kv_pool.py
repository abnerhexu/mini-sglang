from __future__ import annotations

import torch
from minisgl.agentweave.kv import HostKVCapacityError, HostKVPagePool


def _buffer(num_pages: int) -> torch.Tensor:
    tensor = torch.empty((2, 2, num_pages, 1, 1, 1), dtype=torch.float32)
    for page in range(num_pages):
        tensor[:, :, page].fill_(page + 1)
    return tensor


def test_host_kv_pool_store_and_restore_pages() -> None:
    pool = HostKVPagePool(
        4,
        num_layers=2,
        page_shape=(1, 1, 1),
        dtype=torch.float32,
        pin_memory=False,
    )
    source = _buffer(5)
    assert pool.store("prefix", source, [1, 3]) == ()
    assert pool.used_pages == 2

    destination = torch.zeros_like(source)
    assert pool.load("prefix", destination, [0, 4]) == 2
    assert torch.equal(destination[:, :, 0], source[:, :, 1])
    assert torch.equal(destination[:, :, 4], source[:, :, 3])
    assert pool.bytes_stored == 2 * pool.bytes_per_page
    assert pool.bytes_loaded == 2 * pool.bytes_per_page

    partial = torch.zeros_like(source)
    assert (
        pool.load_pages(
            "prefix",
            partial,
            source_offsets=[1],
            destination_pages=[2],
        )
        == 1
    )
    assert torch.equal(partial[:, :, 2], source[:, :, 3])


def test_host_kv_pool_evicts_lru_unpinned_entry() -> None:
    now = [0.0]
    pool = HostKVPagePool(
        4,
        num_layers=2,
        page_shape=(1, 1, 1),
        dtype=torch.float32,
        pin_memory=False,
        clock=lambda: now[0],
    )
    source = _buffer(5)
    pool.store("old", source, [0, 1])
    now[0] = 1.0
    pool.store("recent", source, [2])
    now[0] = 2.0
    pool.load("old", torch.zeros_like(source), [0, 1])
    now[0] = 3.0

    evicted = pool.store("new", source, [3, 4])
    assert evicted == ("recent",)
    assert pool.contains("old")
    assert pool.contains("new")
    assert not pool.contains("recent")


def test_host_kv_pool_never_evicts_pinned_entries() -> None:
    pool = HostKVPagePool(
        2,
        num_layers=2,
        page_shape=(1, 1, 1),
        dtype=torch.float32,
        pin_memory=False,
    )
    source = _buffer(3)
    pool.store("pinned", source, [0, 1], pin=True)

    try:
        pool.store("other", source, [2])
    except HostKVCapacityError:
        pass
    else:
        raise AssertionError("expected pinned entry to prevent eviction")

    pool.unpin("pinned")
    assert pool.store("other", source, [2]) == ("pinned",)
