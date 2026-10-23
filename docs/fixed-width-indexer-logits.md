# Experimental fixed-width sparse-indexer logits

This image backports the production Python changes from [vLLM #58068](https://github.com/vllm-project/vllm/pull/58068), head `1631c22e88f5f08e7977b69863a1c85fb9efb479`, to v0.32.0. Two stale contexts were rebased: the existing GLM empty-local-chunk guard and the shared builder's row-sharding metadata construction. No native operator, DeepGEMM pin, checkpoint weight or quantization format changes.

## Explicit Hopper opt-in

```text
VLLM_SPARSE_INDEXER_FIXED_LOGITS_WIDTH=1
```

Unset retains upstream behavior: integrated GPUs opt in, discrete Hopper does not. `0` explicitly restores variable-width behavior. The image does not bake an enable flag. DCP greater than one keeps variable width.

The experiment uses DeepGEMM's existing `max_seqlen_k` keyword to allocate a stable logits stride, then narrows the logical view to the actual context. The chunk planner budgets the fixed width too. For V4 compression ratio four, allocation width is `ceil(max_model_len / 4)`.

At max length 786432 and a 128 MiB logical logits budget, the global query subchunk bound is 170 rows (25 calls for 4096 rows). At max length 1048576 it is 128 rows (16 calls for 2048 rows). Actual local calls can be reduced by TP prefill-row sharding. These are source-derived counts, not throughput measurements.

## What this does not fix

- This is a shape-reuse / fragmentation experiment, not an unconditional peak-memory reduction.
- Earlier logits remain live during replacement allocation in the current upstream loop. Two full matrices can coexist; this patch does not add a lifetime fix.
- Fixed-width allocations can be larger and require more launches on early/short prompts. No matched Hopper prefill-speed measurement is available.
- Native output row padding and allocator bins still affect real allocation. The logical budget is not a strict total peak cap.
- The patch does not reduce attention Q/output, hc4 residual, metadata-builder, graph-private or shared FlashMLA workspace memory.

The pinned DeepGEMM already supports the keyword; no rebuild is required. CPU build tripwires verify environment tri-state semantics, the DCP guard, planner budgets/row coverage, model-context widths and wrapper keyword forwarding by isolating actual patched source. They do not execute GPU kernels or establish long-context serving correctness.

## Qualification boundary

Keep this as a draft experiment. Compare disabled/enabled behavior over short and long cold prefills, cached extensions, tool traffic and concurrent requests; observe real driver-free slack, reserved/allocated memory, TTFT and output correctness. Retain all existing image tripwires. No workflow dispatch, merge or deployment is part of this change.
