# Experimental EAQ v1.1 qualification

This is a separate image/adapter experiment. It does not replace the production
Dockerfile, and is **not yet GPU-qualified or deployment-ready**.

## Build

From the repository root:

```sh
docker build --progress=plain -f experimental/eaq/Dockerfile -t pcai-eaq:qualification .
```

The base and both EXL3 source revisions are pinned in the Dockerfile. The image
retains complete dependency sources/notices under /opt/pcai-sources. vllm-exl3
is AGPL-3.0-only; ExLlamaV3 is MIT. Preserve those notices and source availability
when distributing the derivative. Our adapter does not copy their implementation.

## EXL3 dense layers (`PCAI_EXL3_DENSE=bf16`)

Mixed packs such as EAQ v1.1 also quantize attention (`wq_a`, `wkv`, `wq_b`,
`wo_a` as eight group slices, `wo_b`), shared experts, Engram `wkv` and the LM
head. vLLM's DeepSeek V4.1 model has no EXL3 path for those layers. With this
variable set, the plugin routes them to ordinary BF16 layers and reconstructs
their EXL3 tensors once while loading, using ExLlamaV3's reference
reconstruction (`LinearEXL3.get_weight_tensor`). Routed experts stay on the
EXL3 kernels. The cost is about 13.8 GiB of BF16 weights across both GPUs.

`verify_dense.py` checks this path on a GPU against the original checkpoint.
On EAQ v1.1 it measured relative errors of 2.1% (6-bit), 3.7-4.3% (5-bit) and
7.0-7.3% (4-bit), cosine >= 0.997, and confirmed `wo_a` group order.

## Adapter contract

PCAI_ENGRAM_DIR activates the plugin and points at the staged directory holding
the original 1017 index and shards 47/48. The chart init must hash-check all three
files before serving starts. O_DIRECT is mandatory; unsupported filesystems fail
closed. Only selected rows are read into an 8 KiB reusable window per tensor.
There is no full-table mmap, pinning, prefetch, or row cache.

The adapter intentionally supports eager mode, DP1, TP-sharded heads, BF16
staging and no stock Engram host/shared/THP allocation. It reuses vLLM's hashing,
residual graph and TP collective. Lookup is synchronous and reference-oriented;
it is not a throughput claim. A native CUDA parity check is required before
treating the torch UE8M0/FP8 reference decoder as qualified.

## Tests

```sh
python3 -m unittest discover -s experimental/eaq/tests -v
```

Row/partition tests use only Python; runtime fixture tests additionally need torch.
The fixture graph mocks vLLM and tests both TP head partitions, inactive/padded
heads, capacity errors and fail-closed graph settings. They do not execute
multi-GPU collectives or the full model.

## Status (2026-10-28)

Qualified on two H100 NVL (TP2) with the published image
`ghcr.io/enthus-appdev/vllm-pcai@sha256:0f7f625b259d50d3a071f7aaee3aa51eae1658fa023eabadb7836ce4ccc8050a`:

- Synthetic TP2 test: K2/K3 GEMV within 5.4e-4 of a dequantized reference; disk Engram rows bit-identical
  to vLLM's CUDA lookup on both ranks, including padded heads.
- Dense EXL3 reconstruction against the original weights: relative error 2.1% (6-bit) to 7.3% (4-bit),
  cosine >= 0.997.
- `exl3_moe` / `exl3_moe_mixedk` read pinned-host (UVA) trellises correctly.
- Full model, eager, 8K context, one request, routed experts in pinned host RAM (183 GiB): loads in
  about 14 minutes, answers correctly, **decode about 12 tok/s**, prefill about 140 tok/s. The bound is
  PCIe: every token reads its experts from host memory.

Not covered: long context, concurrency, CUDA graphs, DSpark, systematic quality, partial expert placement
on the GPUs. This build is experimental and is never published under the production tags.

Test Jobs are bounded, one-shot and run outside any chart: `gpu_qualification.py` (synthetic TP2),
`verify_dense.py` (dense reconstruction vs original weights) and `moe_uva_probe.py` (MoE kernels with
host-resident experts).
