# DSpark auxiliary-only residual buffer

`VLLM_PCAI_DSPARK_SKIP_MTP_BUFFER=1` omits the target model's `_mtp_hidden_buffer` only for V2 DSpark, pipeline parallelism1, and a nonempty valid auxiliary-layer configuration. Default and all other configurations retain existing behavior, including ordinary MTP and missing auxiliary configuration.

The buffer holds pre-head residuals for MTP. DSpark uses configured auxiliary states through its existing projection instead. This patch changes allocation policy only; auxiliary arithmetic, copying of an existing buffer, workspace lanes, graph capture and target computation are unchanged. The existing `None` fallback returns ordinary target hidden states; DSpark's auxiliary-state consumer does not read them.

At4096 batched tokens, hidden4096, hc_mult4 and bf16, omitted storage is128MiB per GPU. This is a source-derived allocation size, not a measured KV refund. Automatic profiling should expose the reduction, but the real available-KV budget and startup/transient peaks must be measured.

## Evaluation

Use the same image and serving arguments twice, first without the environment variable and then with it set to1. Do not combine this with packed scales or other experimental options. Confirm the enabled log:

```
PCAI DSpark auxiliary-only: skipping MTP residual buffer
```

Compare available KV bytes, reported token capacity, model memory, graph memory and peak memory. Run golden, vision, auxiliary-state/speculative acceptance, concurrency and long-prefill tests. Check FULL graph capture and replay, including variable DSpark k and transitions between prefill and decode. Restore the baseline image/settings on the serving-owned deadline.

CPU checks: `python3 -m unittest discover -s tests -q`; installed-package consumer probe: `python3 tools/dspark_mtp_buffer_probe.py`. The latter uses synthetic auxiliary states and a stub projection to verify partial/all graph-staging branches do not read the MTP residual. Neither check executes CUDA graphs or establishes model-output parity.
