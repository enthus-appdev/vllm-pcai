"""CPU source-isolation checks for held-block SWA admission accounting."""

import ast
import pathlib
import types


def verify(source_root: pathlib.Path) -> None:
    tree = ast.parse(
        (source_root / "v1/core/single_type_kv_cache_manager.py").read_text()
    )
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SingleTypeKVCacheManager"
    )
    method = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_num_blocks_to_allocate"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, method], type_ignores=[])
    )
    namespace = {"cdiv": lambda value, divisor: (value + divisor - 1) // divisor}
    exec(compile(module, "swa-admission-tripwire", "exec"), namespace)
    predict = namespace[method.name]

    def block(is_null=False, ref_cnt=1):
        return types.SimpleNamespace(is_null=is_null, ref_cnt=ref_cnt)

    null = block(is_null=True, ref_cnt=0)

    def state(held=(), *, running=False, cap=3, partial=False):
        return types.SimpleNamespace(
            block_size=16,
            max_admission_blocks_per_request=cap,
            req_to_blocks={"request": list(held)},
            num_cached_block={"request": len(held)} if running else {},
            get_num_skipped_tokens=lambda tokens: 64 * 16,
            _get_num_evictable_blocks=lambda blocks: sum(
                not item.is_null and item.ref_cnt == 0 for item in blocks
            ),
            _has_partial_local_hit=lambda blocks, tokens: partial,
        )

    def estimate(manager, hits=(), *, capped=True):
        return predict(
            manager,
            request_id="request",
            num_tokens=67 * 16,
            new_computed_blocks=list(hits),
            total_computed_tokens=66 * 16,
            num_local_computed_tokens=66 * 16 if hits else 0,
            num_tokens_main_model=67 * 16,
            apply_admission_cap=capped,
        )

    held = [null] * 64 + [block(), block()]
    # A logical table of 66 entries holds only two physical window blocks.
    assert estimate(state(held, running=True)) == 1
    assert estimate(state(held)) == 1
    assert estimate(state(held, running=True), capped=False) == 1
    assert estimate(state(held), capped=False) == 1
    assert estimate(state(held, cap=None)) == 1

    for ref_cnt, expected in ((1, 1), (0, 3)):
        hits = [null] * 64 + [block(ref_cnt=ref_cnt), block(ref_cnt=ref_cnt)]
        assert estimate(state(), hits) == expected
        assert estimate(state(), hits, capped=False) == expected
    hits = [null] * 64 + [block(ref_cnt=0), block(ref_cnt=0)]
    assert estimate(state(partial=True), hits) == 4
    assert estimate(state([null] * 64 + [block(), block(), block()], running=True)) == 0

    # Uncapped running allocation must retain its constant-time fast path.
    running = state(held, running=True)

    def unexpected_skip_lookup(tokens):
        raise AssertionError("uncapped fast path consulted the admission window")

    running.get_num_skipped_tokens = unexpected_skip_lookup
    assert estimate(running, capped=False) == 1


if __name__ == "__main__":
    import sys

    verify(pathlib.Path(sys.argv[1]))
    print(
        "SWA held-block admission, shared/evictable hits, partial-hit reserve and uncapped fast path OK (CPU source isolation)"
    )
