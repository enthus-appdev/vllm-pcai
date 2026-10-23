"""CPU source-isolation checks; CUDA kernel correctness needs GPU qualification."""

import ast
import pathlib
import types


def verify(source_root: pathlib.Path) -> None:
    tree = ast.parse((source_root / "models/deepseek_v4/compressor.py").read_text())
    capacity = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_c128_ring_capacity"
    )
    cache = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CompressorStateCache"
    )
    spec = next(
        n
        for n in cache.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_spec"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    cuda = types.SimpleNamespace(is_cuda=lambda: True)

    def factory(kind):
        return lambda **kwargs: types.SimpleNamespace(kind=kind, **kwargs)

    namespace = {
        "_C128_RATIO": 128,
        "current_platform": cuda,
        "CircularBufferSpec": factory("circular"),
        "SlidingWindowMLASpec": factory("sliding"),
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[future, capacity, spec], type_ignores=[])
    )
    exec(compile(module, "c128-tripwire", "exec"), namespace)
    ring_capacity = namespace["_c128_ring_capacity"]
    assert [ring_capacity(n) for n in (0, 1, 5, 127, 128, 129)] == [
        128,
        256,
        256,
        256,
        256,
        384,
    ]
    config = types.SimpleNamespace(
        num_speculative_tokens=5,
        cache_config=types.SimpleNamespace(cache_dtype="fp8_ds_mla"),
    )
    state = types.SimpleNamespace(
        compress_ratio=128,
        state_dim=1024,
        dtype="float32",
        block_size=8,
        sliding_window=128,
    )
    ring = namespace["get_kv_cache_spec"](state, config)
    assert (
        ring.kind,
        ring.block_size,
        ring.head_size,
        ring.head_size_v,
        ring.num_kv_heads,
    ) == ("circular", 256, 1024, 0, 1)
    assert ring.dtype == "float32"
    cuda.is_cuda = lambda: False
    legacy = namespace["get_kv_cache_spec"](state, config)
    assert (
        legacy.kind,
        legacy.block_size,
        legacy.sliding_window,
        legacy.alignment,
    ) == ("sliding", 8, 128, 576)
    cuda.is_cuda = lambda: True
    state.compress_ratio, state.block_size, state.sliding_window = 4, 4, 8
    c4 = namespace["get_kv_cache_spec"](state, config)
    assert (c4.kind, c4.block_size, c4.sliding_window) == ("sliding", 4, 8)
    # At every group phase, a six-row speculative write avoids committed history.
    for start in range(128, 256):
        committed = set(range(start - start % 128, start))
        writes = {position % ring_capacity(5) for position in range(start, start + 6)}
        assert writes.isdisjoint(
            {position % ring_capacity(5) for position in committed}
        )
    cute = ast.parse(
        (
            source_root
            / "models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py"
        ).read_text()
    )
    assert any(
        isinstance(n, ast.ClassDef) and n.name == "SparseAttnCompressC128RingKernel"
        for n in cute.body
    )
    assert (
        "_SPARSE_ATTN_COMPRESS_C128_RING_KERNEL"
        in (
            source_root
            / "models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py"
        ).read_text()
    )


if __name__ == "__main__":
    import sys

    verify(pathlib.Path(sys.argv[1]))
    print(
        "C128 k=5 ring geometry, CUDA-only dispatch, C4 fallback and speculative history bounds OK (CPU source isolation)"
    )
