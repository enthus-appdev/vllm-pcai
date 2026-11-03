# CUDA V4 C128 state geometry

The [Dockerfile](../Dockerfile) selects and pins the active C128 patch. Its `_c128_ring_capacity` function and `CompressorStateCache.get_kv_cache_spec` implementation are authoritative for allocation behavior.

For CUDA C128 state, ring capacity is:

```text
rows_per_request = 128 × ceil((128 + num_speculative_tokens) / 128)
bytes_per_layer_per_request = rows_per_request × state_width × element_size
```

A compressor storing a 512-element key and a 512-element score in FP32 has a 4096-byte row. With five speculative tokens, the ring holds 256 rows: **1 MiB per layer per resident request**. Total raw state depends on the model's C128 layer count, resident requests and replication across ranks; it is independent of request context length. Cache grouping, packing and padding can add allocation overhead.

The circular path consumes current-chunk rows before overwriting the ring tail, preserving committed history across speculative lookahead. Non-CUDA and C4 paths retain their non-circular behavior.

## Limits

This patch changes Python, Triton JIT and CuTe DSL JIT source. It adds no compiled `csrc` operators and does not requantize weights; a compatible CUDA/JIT environment is still required at runtime.

The ring is not a fix for indexer-logit or transient-activation memory use. Automatic KV sizing can consume reclaimed capacity, so raw state geometry does not imply additional activation headroom or a particular serving capacity.

CPU image builds do not establish GPU correctness. Qualification must cover long prefill chunks, group-boundary wraparound, speculative rejection, prefix reuse and CPU-tier reloads on the selected deployment configuration. Deployment limits and measurements belong in the [serving repository](https://github.com/enthus-appdev/pcai-llm-serving).
