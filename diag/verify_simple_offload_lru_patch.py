"""CPU source-isolation checks for eager SimpleCPUOffload prefix LRU refresh."""

import ast
import pathlib
import types
from unittest.mock import Mock


class Block:
    def __init__(self, name, *, is_null=False):
        self.name = name
        self.is_null = is_null
        self.ref_cnt = 0


class Pool:
    def __init__(self, blocks):
        self.free = list(blocks)
        self.touches = []
        self.releases = []

    def touch(self, blocks):
        self.touches.append(list(blocks))
        for block in blocks:
            assert not block.is_null
            if block.ref_cnt == 0:
                self.free.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        blocks = list(blocks)
        self.releases.append(blocks)
        for block in blocks:
            assert not block.is_null and block.ref_cnt > 0
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free.append(block)


def verify(source_root: pathlib.Path) -> None:
    tree = ast.parse((source_root / "v1/simple_kv_offload/manager.py").read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(item, ast.FunctionDef)
            and item.name == "get_num_new_matched_tokens"
            for item in node.body
        )
    )
    method = next(
        item
        for item in owner.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "get_num_new_matched_tokens"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, method], type_ignores=[])
    )
    namespace = {"logger": Mock()}
    exec(compile(module, "simple-offload-lru-tripwire", "exec"), namespace)
    lookup = namespace[method.name]

    def fixture(groups, blocks, *, lazy=False, skip=False):
        coordinator = Mock()
        coordinator.find_longest_cache_hit.return_value = (groups, 32, 0)
        pool = Pool(blocks)
        state = types.SimpleNamespace(
            _pending_cpu_hits={},
            _free_pending_cpu_hit=Mock(),
            _lazy_mode=lazy,
            block_size=16,
            hash_block_size=16,
            cpu_coordinator=coordinator,
            cpu_block_pool=pool,
        )
        request = types.SimpleNamespace(
            request_id="request",
            skip_reading_prefix_cache=skip,
            block_hashes=["first", "second"],
            num_tokens=32,
        )
        return state, request, pool, coordinator

    a0, a1, b0, b1 = [Block(name) for name in ("a0", "a1", "b0", "b1")]
    state, request, pool, coordinator = fixture([[a0, a1]], [a0, a1, b0, b1])
    assert lookup(state, request, 32) == (0, False)
    coordinator.find_longest_cache_hit.assert_called_once_with(request.block_hashes, 32)
    assert pool.free == [b0, b1, a1, a0]
    # The two cold blocks, not the GPU-hit prefix, are next to be evicted.
    assert [pool.free.pop(0), pool.free.pop(0)] == [b0, b1]
    assert state._pending_cpu_hits == {}
    assert a0.ref_cnt == a1.ref_cnt == 0

    a0, a1, m0, m1, cold = [Block(name) for name in ("a0", "a1", "m0", "m1", "cold")]
    null = Block("null", is_null=True)
    state, request, pool, _ = fixture([[a0, a1], [null, m1]], [a0, a1, m0, m1, cold])
    assert lookup(state, request, 32) == (0, False)
    assert pool.free == [m0, cold, m1, a1, a0]
    assert pool.releases == [[m1, a1, a0]]
    assert all(block.ref_cnt == 0 for block in pool.free)
    # Retried lookups refresh again without retaining a pin.
    assert lookup(state, request, 32) == (0, False)
    assert len(pool.touches) == 2 and state._pending_cpu_hits == {}

    for lazy, skip in ((True, False), (False, True)):
        a = Block("a")
        state, request, pool, coordinator = fixture([[a]], [a], lazy=lazy, skip=skip)
        assert lookup(state, request, 32) == (0, False)
        coordinator.find_longest_cache_hit.assert_not_called()
        assert pool.touches == pool.releases == []

    # Stale suffix pins are released before either lookup or opt-out.
    state, request, _, coordinator = fixture([], [], skip=True)
    state._pending_cpu_hits[request.request_id] = "stale"
    assert lookup(state, request, 32) == (0, False)
    state._free_pending_cpu_hit.assert_called_once_with("stale")
    coordinator.find_longest_cache_hit.assert_not_called()

    # Existing external-suffix matching remains asynchronous and pins its hits.
    hit = Block("suffix")
    state, request, pool, coordinator = fixture([[hit]], [hit], lazy=True)
    request.num_tokens = 48
    coordinator.find_longest_cache_hit.return_value = ([[hit]], 16, 0)
    assert lookup(state, request, 0) == (16, True)
    assert state._pending_cpu_hits[request.request_id] == ([[hit]], 16, 0)
    assert hit.ref_cnt == 1 and pool.releases == []


if __name__ == "__main__":
    import sys

    verify(pathlib.Path(sys.argv[1]))
    print(
        "SimpleCPU eager GPU-hit MRU refresh, hybrid/null handling, lazy/skip behavior and async suffix matching OK (CPU source isolation)"
    )
