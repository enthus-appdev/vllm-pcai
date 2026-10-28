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

## Next test phase

`gpu_qualification.py` is a bounded synthetic two-H100/TP2 smoke, not a model
load or serving endpoint. Run it only in an explicitly approved one-shot Job
with a small disk `/fixtures` volume, a timeout and no retries. It compares K2/K3
native GEMV with a dequantized reference and disk Engram with the stock CUDA
lookup, including real vLLM TP gathering. A pass does not qualify full-model
loading, cooperative expert kernels, UVA placement, quality or performance.

The companion `chart-support.patch` contains the staging chart and its exact EAQ
candidate defaults (published image, verified PVCs, disk staging and bounded
resources). No override file is required. Its default replica count is zero;
installing it does not start inference or run the synthetic GPU test. Source and
destination staging files are hash-checked before serving begins.

## Release gates

1. Native sm_90 extensions compile and import without replacing the base torch.
2. Audit the pinned fork's Hopper synchronization behaviour; small GPU kernels
   must finish under a timeout and match a dequantized reference for K2 and K3.
3. Verify both TP ranks' selected Engram bytes/scales and numerical results against
   the stock CUDA lookup, including padding and DEAD_ID.
4. Audit all real EAQ dense/BF16/expert key/shape mappings and UVA placement.
5. Only then publish the candidate digest and finalize attended test values.

No inference deployment or rollback is performed by the build/tests. The attended
boot uses C1/8K/512 prefill, no DSpark, no vision, no async scheduling or RAM KV tier.
Do not raise the memory limit on failure or leave Deployment restart loops running.
