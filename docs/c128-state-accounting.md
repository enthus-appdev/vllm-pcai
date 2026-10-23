# Experimental CUDA V4 C128 state accounting

The image carries the production Python diff from [vLLM #58074](https://github.com/vllm-project/vllm/pull/58074), pinned to `f8bb0ba871415a779de33fadcb40ab2ba5acbc09`. It changes Python, Triton JIT and CuTe DSL JIT source only. It adds no compiled `csrc` operators and does not requantize weights. A CUDA toolchain/JIT environment is still needed when the model starts; a CPU image build cannot qualify those kernels.

## Geometry

For [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json), 20 of the 43 target layers use compression ratio 128. Each C128 compressor stores a 512-element key and a 512-element score in FP32: **1024 × 4 = 4096 bytes per stored row per layer**. State is replicated at TP2, not halved.

On v0.32.0 with packed fp8-DS-MLA (the V4 `fp8` alias), the legacy compressor spec has eight rows per page and 576-byte alignment:

- Payload: `8 × 4096 = 32768` bytes.
- Aligned page: `ceil(32768 / 576) × 576 = 32832` bytes.
- Amortized cost across 20 layers: `32832 / 8 × 20 = 82080` bytes per **additionally retained input-token row**, per rank (80.15625 KiB).
- One 256-token span: `82080 × 256 = 21012480` bytes, or 20.0390625 MiB per rank. The attention KV block size 256 is not the old compressor page size, which is eight.

The patch selects ring capacity `128 × ceil((128 + k) / 128)`. At DSpark `k=5` this is 256 rows:

- One layer/request: `256 × 4096 = 1048576` bytes (1 MiB).
- Twenty C128 layers/request/rank: **20 MiB**, independent of context length.
- Six resident requests/rank: **120 MiB** raw state. Across TP2: **240 MiB summed**.
- Normalized over a fully occupied 1,048,576-token request: 20 bytes/context-token/rank. This normalization is not a token-wise allocation rule; a short request still owns the ring.

## Bounded legacy state: do not multiply the old cost by 1M

The original state is already sliding-window managed. It does **not** retain all historical input rows. The release `SlidingWindowSpec.max_admission_blocks_per_request` bounds it by:

```text
old_pages = ceil((127 + max_in_flight_tokens) / 8) + 1
max_in_flight_tokens = max_concurrent_batches × max_num_batched_tokens
```

The extra page covers an unaligned window start. For one pipeline stage, the release uses one concurrent batch with synchronous scheduling and two with V2 asynchronous scheduling. With batched tokens 4096, the theoretical per-request admission bounds are:

| In-flight token bound | Old pages/layer | Old 20-layer state/rank | Ring state/rank | Raw reduction/rank |
|---:|---:|---:|---:|---:|
| 4096 (one batch) | 529 | 331.270752 MiB | 20 MiB | 311.270752 MiB |
| 8192 (two batches) | 1041 | 651.895752 MiB | 20 MiB | 631.895752 MiB |

Multiplying those per-request bounds by six gives reductions of 1867.624512 MiB and 3791.374512 MiB per rank, respectively. **These are conservative per-request sizing bounds, not simultaneous observed usage:** six requests share the scheduled-token budget, and cache grouping/packing/padding, shared pools and admission policy determine actual allocation. They are not promised increases in available HBM or usable KV capacity.

At a six-row decode-sized bound, the legacy formula gives 18 pages/layer, or 11.271973 MiB across 20 layers; the ring is 20 MiB. Thus this is not uniformly less raw state at every step. Its benefit must be assessed on actual long-prefill allocation and cache-group behavior.

## Qualification boundaries

The patch changes neither indexer logits sizing nor transient activation allocation. Automatic KV sizing can spend reclaimed state capacity on more KV, leaving no extra activation headroom. This experiment alone is not a fix for long-prefill indexer OOM or allocator fragmentation.

Build tripwires execute isolated source methods on CPU to verify the CUDA-only spec dispatch, k=5 ring geometry, legacy/C4 fallbacks and speculative-history bounds. They do not execute Triton/CuTe kernels, prove CUDA-graph replay correctness, or validate CPU offload/reload. Those require a separate GPU qualification with long prefill chunks, group-boundary wraparound, DSpark rejection, prefix reuse and CPU-tier reloads.
