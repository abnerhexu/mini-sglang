from __future__ import annotations

import torch
from minisgl.kvcache.radix_cache import RadixCacheHandle, RadixPrefixCache, RadixTreeNode


def _cache() -> RadixPrefixCache:
    cache = RadixPrefixCache.__new__(RadixPrefixCache)
    cache.device = torch.device("cpu")
    cache.page_size = 1
    cache.key_fn = lambda value: int(value[0].item())
    cache.empty_tensor = torch.empty(0, dtype=torch.int32)
    cache.evictable_size = 0
    cache.protected_size = 0
    cache.root_node = RadixTreeNode(cache.key_fn)
    cache.root_node.ref_count = 1
    return cache


def _child(cache: RadixPrefixCache, parent: RadixTreeNode, token: int) -> RadixTreeNode:
    node = RadixTreeNode(cache.key_fn)
    value = torch.tensor([token], dtype=torch.int32)
    node.set_key_value(value, value.clone())
    node.set_parent(parent)
    cache.evictable_size += 1
    return node


def test_targeted_evict_removes_unshared_leaf_chain() -> None:
    cache = _cache()
    first = _child(cache, cache.root_node, 1)
    leaf = _child(cache, first, 2)

    evicted = cache.evict_handle(RadixCacheHandle(2, leaf))
    assert evicted.tolist() == [1, 2]
    assert cache.evictable_size == 0
    assert cache.root_node.children == {}


def test_targeted_evict_stops_at_shared_or_locked_node() -> None:
    cache = _cache()
    shared = _child(cache, cache.root_node, 1)
    leaf = _child(cache, shared, 2)
    sibling = _child(cache, shared, 3)

    evicted = cache.evict_handle(RadixCacheHandle(2, leaf))
    assert evicted.tolist() == [2]
    assert shared.children[3] is sibling
    assert cache.evictable_size == 2

    sibling.ref_count = 1
    cache.evictable_size -= 1
    cache.protected_size += 1
    assert len(cache.evict_handle(RadixCacheHandle(2, sibling))) == 0
