# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) images for **HPE Private Cloud AI (PCAI)** — used by all production, secondary, and experimental model deployments (Qwen3.6-27B, Gemma 4 31B, DeepSeek V4 Flash).

## Why this image exists

This image adds four layers on top of the stock `vllm/vllm-openai` base. They are baked in because MLIS cannot mount volumes (see [Deployment via MLIS](#deployment-via-mlis)). Custom-framework charts can mount PVCs; that is a separate deployment path:

1. **Enhanced chat templates** — Qwen3.5/3.6 hardened templates (hidden historical reasoning, XML tool-call formatting, proper ` response` handling) from [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix), which are *not* in the base image. (Gemma 4 uses vLLM's **in-image** template at `/vllm-workspace/examples/tool_chat_template_gemma4.jinja`.)

2. **Diagnostics endpoint** — `GET /collect_env` on the serving port reports runtime versions and GPU details. Registered on the active ASGI factory (`vllm.entrypoints.launchers.app`), with a build-time HTTP regression test. The route has no authentication of its own; keep it behind the appliance's authenticated ingress.

3. **Vendored patches** — source fixes applied to the digest-pinned v0.33.0rc3 source (Docker tag `v0.33.0`):
   - **DeepSeek generation controls** ([#46257](https://github.com/vllm-project/vllm/pull/46257), head `3dd748819f25fa64250258cdb059c5b0e4487563`) — honor `add_generation_prompt` and `continue_final_message` in the V4/V3.2 encoders.
   - **Speculative drafter weight source** ([#48023](https://github.com/vllm-project/vllm/pull/48023)) — inherit the target's weight source when a drafter shares its checkpoint.
   - **Consecutive assistant messages** ([#50686](https://github.com/vllm-project/vllm/pull/50686)) — merge split assistant turns before DeepSeek V4 encoding.
   - **PCAI queue sizing** — make worker response queues honor `VLLM_MQ_MAX_CHUNK_BYTES_MB` for constrained `/dev/shm`.
   - **C128 circular state** ([#58074](https://github.com/vllm-project/vllm/pull/58074), head `f8bb0ba871415a779de33fadcb40ab2ba5acbc09`) — use bounded per-request CUDA V4 compressor history with speculative lookahead.
   - **Sliding-window admission** ([#60045](https://github.com/vllm-project/vllm/pull/60045), head `843db1462a066076d407bb628098140da4455be5`) — count held physical blocks, not null-padded positions, when reserving capacity for asynchronous KV loads.
   - **Eager CPU-cache LRU** ([#62444](https://github.com/vllm-project/vllm/pull/62444), head `a7f3bd29157fb6c96a1eff083cf8fc14136be624`) — refresh CPU-copy recency when a prefix hits on GPU, keeping hot RAM copies from being evicted first.
   - **Host-tier load ordering** ([#60078](https://github.com/vllm-project/vllm/pull/60078), head `520208b140dbca471a76c752598b6fff2f8a7092`) — make SimpleCPUOffload loads wait for queued compute that still reads the destination blocks (the race async scheduling exposes).

   Plugin **`pcai_embed_offload`** (`plugins/`): with `--offload-backend uva --cpu-offload-params ... embed_tokens`, DeepSeek V4's embedding table moves to pinned host memory (about 0.5 GiB per GPU at TP2). vLLM offloads decoder layers and vision towers through `--cpu-offload-params`, but not the embedding. Inactive unless requested.

   The base supplies the EOS reasoning-leak fix ([#48748](https://github.com/vllm-project/vllm/pull/48748)); the image checks it with a regression assertion rather than another patch.

4. **Build-time tripwire assertions** — inline regression assertions verify parser classes, engine features, config knobs and selected patched behavior during the image build. They catch incompatible bases without claiming GPU serving correctness.

## Combined experiment C1c

C1c retains all C1 stability and FP8 components, adding a bounded SWA checkpoint
fallback for no-drop, single-lookahead speculative paths (including DSpark/DFlash).
It keeps the latest semantic boundary and the preceding aligned boundary, so a
nearby changed-question prefix can hit without losing the exact-resend benefit of
#54163. EAGLE-drop paths, non-speculative paths and non-SWA groups are unchanged.
This does not guarantee reuse for edits before both retained boundaries.

The extra patch is `patches/dspark-swa-prefix-fallback-v033.patch`.
No new runtime flag or periodic-retention override is baked in.
GPU/host cache occupancy and end-to-end correctness still need qualification.

The v0.33 update preserves this policy while retaining upstream's separate
EAGLE group/drop-group tracking. Runtime controls are unchanged.
This does not include the indexer-memory or int8 bundle.
Build checks do not qualify the combined GPU runtime.

## Experimental stability bundle

The stability component retains the following behavior (only #63427 still needs a patch):
- [#63427](https://github.com/vllm-project/vllm/pull/63427), head `6ee3f40d6137b487617e2ab1e5ff1a13402a0cda`: recursively weak-reference nested graph outputs.
- [#54163](https://github.com/vllm-project/vllm/pull/54163), head `3088df4c8770f859e23de3326d0c6dc8ce37f1a7`: preserve DFlash/DSpark's trailing prefix block. Now supplied upstream; broad EAGLE lookahead and #48023 remain intact.
- [#60553](https://github.com/vllm-project/vllm/pull/60553), head `5a675fa4d91def4031bc19e5d9bb016a69494d2d`: encoder-cache progress at adjacent-image/lookahead boundaries.
- [#59030](https://github.com/vllm-project/vllm/pull/59030), head `54e0b57e7c95f48d04f0d365cb2da0cc126071da`: requeue encoder-stalled waiting requests.

The remaining patch is `patches/63427-nested-weak-ref-tensors-v033.patch`.
#54163, #60553 and #59030 are supplied upstream.
No extra runtime flag is needed. CUDA graph lifetimes, multi-image scheduling and
GPU outputs require workload qualification.
Run `python3 -m unittest discover -s tests -v` for local integrity checks.

## Experimental Hopper FP8 MoE activation image

The base now supplies all three former FP8 backports:
- [#62751](https://github.com/vllm-project/vllm/pull/62751), head `da3762163f542e5ca347dc358d976c86a795b1dc`: shared Humming activation-schema resolution.
- [#61875](https://github.com/vllm-project/vllm/pull/61875), head `1ef812f7be96f5ce000fb002e8c8edc74e122cb1`: DeepSeek V4/V4.1 checkpoint MXFP8 resolves to **block FP8 group128 on SM90** by default.
- [#62816](https://github.com/vllm-project/vllm/pull/62816), head `b6e146e7f32597af02621c5576fffcfcee497a28`: Humming **0.1.18** and removal of the obsolete DeviceInfo workaround.

The FP8 patch bundle and custom wheel installation have been removed. The base
supplies Humming **0.1.19**; we do not downgrade it to the former 0.1.18 pin.
No additional dependency installation is needed.

**FP8 test:** select `--moe-backend humming`, leave
`VLLM_HUMMING_INPUT_QUANT_CONFIG` unset, and do not set
`--quantization-config.moe.activation mxfp8` (an explicit override requests
group32 instead). Expect the log:
`Running checkpoint MXFP8 MoE activations as block FP8 (group 128) on Hopper.`

**Same-image BF16 control:** set
`VLLM_HUMMING_INPUT_QUANT_CONFIG='{"allow_fallback": true}'`, as in the upstream
A/B. This image changes Humming as well as the activation policy; comparison
against a prior image alone does not isolate the dependency upgrade.

Numerical accuracy, graph compilation and performance require GPU testing.
Run `python3 -m unittest discover -s tests -v` locally.

## Base image: `v0.33.0` (RC3 source, digest-pinned)

The `FROM` uses Docker tag `v0.33.0`, pinned to manifest
`sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e`.
Both architectures report build commit `4ba787ec9959e566d3b587a796dff07673324da5`,
which is the [v0.33.0rc3 release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3).
There is no Docker `v0.33.0rc3` tag, and the Git `v0.33.0` tag points elsewhere;
the digest avoids silently switching to unaudited source. SHM notification handling
and the FP8 bundle are supplied upstream; the remaining patches are listed above.
Building or publishing this image does not redeploy an existing model service.

DeepSeek V4 DSpark external-cache reuse needs separate validation ([#47890](https://github.com/vllm-project/vllm/issues/47890), proposed fix [#47891](https://github.com/vllm-project/vllm/pull/47891)). This image does not carry that experimental backport or enable offloading. An engine upgrade alone does not establish cache correctness or benefit.

**Bumping is not a date comparison.** vLLM cuts release branches, so validate required behavior with the Dockerfile tripwires and GPU workloads rather than relying only on release dates or commit ancestry.

## Layout

```
vllm-pcai/
├── Dockerfile                # FROM vllm/vllm-openai:v0.33.0@sha256:…
│                               + Qwen enhanced templates
│                               + /collect_env diagnostics route and HTTP regression test
│                               + Parser, loader, KV-cache and transport patches
│                               + Build-time tripwires for all three models
├── chat-template-fix/        # git submodule → allanchan339/Qwen templates
├── diag/                     # diagnostics route
├── patches/                  # remaining parser, drafter, KV-cache and transport fixes
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
