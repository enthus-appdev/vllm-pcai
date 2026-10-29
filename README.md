# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) images for **HPE Private Cloud AI (PCAI)** — used by all production, secondary, and experimental model deployments (Qwen3.6-27B, Gemma 4 31B, DeepSeek V4 Flash).

## Why this image exists

This image adds four layers on top of the stock `vllm/vllm-openai` base. They are baked in because MLIS cannot mount volumes (see [Deployment via MLIS](#deployment-via-mlis)). Custom-framework charts can mount PVCs; that is a separate deployment path:

1. **Enhanced chat templates** — Qwen3.5/3.6 hardened templates (hidden historical reasoning, XML tool-call formatting, proper ` response` handling) from [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix), which are *not* in the base image. (Gemma 4 uses vLLM's **in-image** template at `/vllm-workspace/examples/tool_chat_template_gemma4.jinja`.)

2. **Diagnostics endpoint** — `GET /collect_env` on the serving port reports runtime versions and GPU details. Registered on the active ASGI factory (`vllm.entrypoints.launchers.app`), with a build-time HTTP regression test. The route has no authentication of its own; keep it behind the appliance's authenticated ingress.

3. **Vendored patches** — source fixes applied to the v0.32.0 base:
   - **DeepSeek generation controls** ([#46257](https://github.com/vllm-project/vllm/pull/46257), head `3dd748819f25fa64250258cdb059c5b0e4487563`) — honor `add_generation_prompt` and `continue_final_message` in the V4/V3.2 encoders.
   - **Speculative drafter weight source** ([#48023](https://github.com/vllm-project/vllm/pull/48023)) — inherit the target's weight source when a drafter shares its checkpoint.
   - **Consecutive assistant messages** ([#50686](https://github.com/vllm-project/vllm/pull/50686)) — merge split assistant turns before DeepSeek V4 encoding.
   - **PCAI queue sizing** — make worker response queues honor `VLLM_MQ_MAX_CHUNK_BYTES_MB` for constrained `/dev/shm`.
   - **C128 circular state** ([#58074](https://github.com/vllm-project/vllm/pull/58074), head `f8bb0ba871415a779de33fadcb40ab2ba5acbc09`) — use bounded per-request CUDA V4 compressor history with speculative lookahead.
   - **SHM notification backlog** ([#63165](https://github.com/vllm-project/vllm/pull/63165), head `d3ada0390ca556e62150a3c751cdd893a2306ebf`) — poll busy readers' notification sockets to bound the libzmq command backlog.
   - **Sliding-window admission** ([#60045](https://github.com/vllm-project/vllm/pull/60045), head `843db1462a066076d407bb628098140da4455be5`) — count held physical blocks, not null-padded positions, when reserving capacity for asynchronous KV loads.
   - **Eager CPU-cache LRU** ([#62444](https://github.com/vllm-project/vllm/pull/62444), head `a7f3bd29157fb6c96a1eff083cf8fc14136be624`) — refresh CPU-copy recency when a prefix hits on GPU, keeping hot RAM copies from being evicted first.
   - **Host-tier load ordering** ([#60078](https://github.com/vllm-project/vllm/pull/60078), head `520208b140dbca471a76c752598b6fff2f8a7092`) — make SimpleCPUOffload loads wait for queued compute that still reads the destination blocks (the race async scheduling exposes).

   Plugin **`pcai_embed_offload`** (`plugins/`): with `--offload-backend uva --cpu-offload-params ... embed_tokens`, DeepSeek V4's embedding table moves to pinned host memory (about 0.5 GiB per GPU at TP2). vLLM offloads decoder layers and vision towers through `--cpu-offload-params`, but not the embedding. Inactive unless requested.

   The base supplies the EOS reasoning-leak fix ([#48748](https://github.com/vllm-project/vllm/pull/48748)); the image checks it with a regression assertion rather than another patch.

4. **Build-time tripwire assertions** — regression assertions and CPU source-isolation checks verify parser classes, engine features, config knobs and patched behavior during the image build. They catch incompatible bases without claiming GPU serving correctness.

## Combined experiment C2

C2 adds the indexer-memory component to C1 (stability + FP8 MoE), on PR #62
source commit `21da126678a5e9fa4d37223516297f8a9bf627e0`. Component patch hashes
and runtime controls are unchanged. It does not include the int8 bundle.
Build/CPU checks do not qualify the combined GPU runtime.

## Experimental stability bundle

The stability component adds the following bundle:
- [#63427](https://github.com/vllm-project/vllm/pull/63427), head `6ee3f40d6137b487617e2ab1e5ff1a13402a0cda`: recursively weak-reference nested graph outputs.
- [#54163](https://github.com/vllm-project/vllm/pull/54163), head `3088df4c8770f859e23de3326d0c6dc8ce37f1a7`: preserve DFlash/DSpark's trailing prefix block. Config context is adapted to v0.32.0 (without newer XPRESS code); broad EAGLE lookahead and #48023 remain intact.
- [#60553](https://github.com/vllm-project/vllm/pull/60553), head `5a675fa4d91def4031bc19e5d9bb016a69494d2d`: encoder-cache progress at adjacent-image/lookahead boundaries.
- [#59030](https://github.com/vllm-project/vllm/pull/59030), head `54e0b57e7c95f48d04f0d365cb2da0cc126071da`: requeue encoder-stalled waiting requests.

The production-only bundle is pinned in `patches/vision-stability-v032.patch`.
No extra runtime flag is needed. The CPU tripwire executes isolated source predicates,
nested-reference behavior with stand-ins, and encoder reclamation boundaries; it does
not qualify actual CUDA graph lifetimes, multi-image scheduling or GPU outputs.
Run `python3 -m unittest discover -s tests -v` for local integrity checks.

## Experimental Hopper FP8 MoE activation image

The FP8 activation component contains:
- [#62751](https://github.com/vllm-project/vllm/pull/62751), head `da3762163f542e5ca347dc358d976c86a795b1dc`: shared Humming activation-schema resolution.
- [#61875](https://github.com/vllm-project/vllm/pull/61875), head `1ef812f7be96f5ce000fb002e8c8edc74e122cb1`: DeepSeek V4/V4.1 checkpoint MXFP8 resolves to **block FP8 group128 on SM90** by default.
- [#62816](https://github.com/vllm-project/vllm/pull/62816), head `b6e146e7f32597af02621c5576fffcfcee497a28`: Humming **0.1.18** and removal of the obsolete DeviceInfo workaround.

Production hunks are unchanged; the requirements-file bump is implemented as a
hash-pinned x86_64 wheel installation with `--no-deps`, preserving the base Torch
and CUDA dependency set. The bundle is `patches/vision-fp8-moe-v032.patch`.

**FP8 test:** select `--moe-backend humming`, leave
`VLLM_HUMMING_INPUT_QUANT_CONFIG` unset, and do not set
`--quantization-config.moe.activation mxfp8` (an explicit override requests
group32 instead). Expect the log:
`Running checkpoint MXFP8 MoE activations as block FP8 (group 128) on Hopper.`

**Same-image BF16 control:** set
`VLLM_HUMMING_INPUT_QUANT_CONFIG='{"allow_fallback": true}'`, as in the upstream
A/B. This image changes Humming as well as the activation policy; comparison
against a prior image alone does not isolate the dependency upgrade.

CPU tripwires check default/override resolution with stand-ins, model declaration,
installed package version and base Torch version. They do not execute Humming
kernels or qualify numerical accuracy, graph compilation or performance.

## Experimental indexer-memory image

The shared indexer-memory component contains:
- [#63047](https://github.com/vllm-project/vllm/pull/63047), head `4f96a21bb5126dc930d6eaec424100cf1e748418`: decode-only expanded block tables are bounded by decode rows; previous prefill logits are released after their last consumer. **No V4.1/Engram model changes.**
- [#58068](https://github.com/vllm-project/vllm/pull/58068), head `1631c22e88f5f08e7977b69863a1c85fb9efb479`, using image PR #55's v0.32 adaptation at `b8a6fc992c0fa8bb4a9ab388e537a4bea1ece7b2`: fixed-width prefill allocation and matching subchunk budgets.
- [#62860](https://github.com/vllm-project/vllm/pull/62860) prerequisite: adapted **SM90 FP32** decode-logits profiling before KV sizing. Newer SM100 BF16/copy handling is deliberately excluded because this base does not contain that dispatch.

The single patch is `patches/vision-indexer-memory-v032.patch`. The metadata hunk
from #63047 applies unchanged; its lifetime hunk is rebased after fixed-width
view creation and after DCP merging. The profiler reserves the larger of the
prefill budget and padded SM90 decode logits, so safer sizing can reduce reported
KV capacity rather than always increasing it.

**This image defaults `VLLM_SPARSE_INDEXER_FIXED_LOGITS_WIDTH=1`.** No extra flag
is needed. Set it to `0` for a same-image comparison without fixed-width logits;
the table/lifetime/profiling fixes remain active. Fixed width can increase
early-context allocation and subchunk count, so memory and throughput require
measurement. DCP retains the upstream guard disabling fixed width.

CPU tripwires cover env selection, DCP guard, chunk budgets, DeepGEMM argument
forwarding, decode table bounds, profile arithmetic and final-use deletion.
There is no GPU/OOM/accuracy qualification. The existing DeepGEMM
`max_seqlen_k` API is reused; no native extension rebuild or new dependency.
Run `python3 -m unittest discover -s tests -v` locally.

## Base image: `v0.32.0`

The `FROM` uses the **[v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)**. This retains the ParserEngine, hybrid DFlash, DeepSeek V4 DSpark, packed heterogeneous-KV support, and OffloadingConnector bounds fix ([#52923](https://github.com/vllm-project/vllm/pull/52923)). It also updates Hopper's quantization backend selection; performance still requires workload-specific GPU validation. Building or publishing this image does not redeploy an existing model service.

DeepSeek V4 DSpark external-cache reuse needs separate validation ([#47890](https://github.com/vllm-project/vllm/issues/47890), proposed fix [#47891](https://github.com/vllm-project/vllm/pull/47891)). This image does not carry that experimental backport or enable offloading. An engine upgrade alone does not establish cache correctness or benefit.

**Bumping is not a date comparison.** vLLM cuts release branches, so validate required behavior with the Dockerfile tripwires and GPU workloads rather than relying only on release dates or commit ancestry.

## Layout

```
vllm-pcai/
├── Dockerfile                # FROM vllm/vllm-openai:v0.32.0
│                               + Qwen enhanced templates
│                               + /collect_env diagnostics route and HTTP regression test
│                               + Parser, loader, KV-cache and transport patches
│                               + Build-time tripwires for all three models
├── chat-template-fix/        # git submodule → allanchan339/Qwen templates
├── diag/                     # diagnostics route and CPU patch tripwires
├── patches/                  # v0.32.0 parser, drafter, KV-cache and transport fixes
├── tests/                    # patch integrity checks (python3 -m unittest discover -s tests)
└── .dockerignore
```

## Deployment via MLIS

**Limitations**

- No volume mounts through the UI. Templates and patches must be in the image.
- `/dev/shm` is fixed at 64 MiB and cannot be enlarged.

**Requirements**

- Set `VLLM_MQ_MAX_CHUNK_BYTES_MB=1` in the deployment's environment. The image leaves vLLM's default (16 MiB), and with the bundled queue patch the message queues then need about `(TP + 1) × 160 MiB` of `/dev/shm`: 480 MiB at TP=2. vLLM refuses to start when they do not fit. At 1 MiB they need about 30 MiB at TP=2; larger messages fall back to a local socket.

Deployments that mount a larger memory-backed `/dev/shm` volume need no override.

## Templates available at runtime

| Path | Source |
|------|--------|
| `/templates/qwen3.6-enhanced.jinja` | this image (allanchan339 fix) |
| `/templates/qwen3.5-enhanced.jinja` | this image (allanchan339 fix) |
| `/vllm-workspace/examples/*.jinja` | stock vLLM templates (incl. Gemma 4) |

## Documentation scope

This public repository documents the image: build instructions, templates, patches, diagnostics and tests. Deployment-specific configurations, operational measurements and research are maintained privately and are not published here.

The vendored-patch list above describes the current image. [C128 state accounting](docs/c128-state-accounting.md) explains the bundled patch's source-level memory geometry; it is not a deployment configuration or GPU qualification.

## Clone

```bash
git clone --recurse-submodules https://github.com/enthus-appdev/vllm-pcai.git
```

## Build & push

CI builds and pushes automatically (`.github/workflows/build.yml`) to
`ghcr.io/enthus-appdev/vllm-pcai` (`:latest`, `:main`, `:sha-…`; push a `v*` tag for semver tags). Manually:

```bash
docker build -t ghcr.io/enthus-appdev/vllm-pcai:latest .
docker push ghcr.io/enthus-appdev/vllm-pcai:latest
```

## Update the templates

```bash
cd chat-template-fix && git fetch && git checkout <commit-or-tag> && cd ..
git commit -am "chore: bump chat-template-fix"
```

## License

Repo files: Apache-2.0. The enhanced templates retain their upstream license.
