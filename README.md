# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) images for **HPE Private Cloud AI (PCAI)** — used by all production, secondary, and experimental model deployments (Qwen3.6-27B, Gemma 4 31B, DeepSeek V4 Flash).

## Why this image exists

PCAI cannot mount volumes through its UI, so anything a model needs at runtime that isn't in the base `vllm/vllm-openai` image **must be baked in**. This image adds four layers on top of the stock vLLM base:

1. **Enhanced chat templates** — Qwen3.5/3.6 hardened templates (hidden historical reasoning, XML tool-call formatting, proper ` response` handling) from [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix), which are *not* in the base image. (Gemma 4 uses vLLM's **in-image** template at `/vllm-workspace/examples/tool_chat_template_gemma4.jinja`.)

2. **Diagnostics endpoint** — `GET /collect_env` on the serving port reports runtime versions and GPU details. Registered on the active ASGI factory (`vllm.entrypoints.launchers.app`), with a build-time HTTP regression test. The route has no authentication of its own; keep it behind the appliance's authenticated ingress.

3. **Vendored patches** — the `deepseek_v4` generation-control fix ([#46257](https://github.com/vllm-project/vllm/pull/46257), exact source diff from rebased head `3dd748819f25fa64250258cdb059c5b0e4487563`, cleanly applicable to v0.30.0), the speculative drafter weight-source fix ([#48023](https://github.com/vllm-project/vllm/pull/48023)), the consecutive-assistant-message fix ([#50686](https://github.com/vllm-project/vllm/pull/50686)), and PCAI's `/dev/shm` queue-size fix. The former EOS reasoning-leak patch ([#48748](https://github.com/vllm-project/vllm/pull/48748)) is upstream and retained as a regression assertion only.

4. **Build-time tripwire assertions** — each layer ends with a `RUN python3 -c` that asserts the base image carries the expected parser classes, engine features, and config knobs. A bump that breaks any of them fails **here**, not on a GPU pod.

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
│                               + DeepSeek V4 parser patches
│                               + Build-time tripwires for all three models
├── chat-template-fix/        # git submodule → allanchan339/Qwen templates
├── diag/                     # collect_env_route.py
├── patches/                  # v0.30.0-compatible parser, drafter, and PCAI queue fixes
├── tests/                    # patch syntax checks (python3 -m unittest discover -s tests)
└── .dockerignore
```

## Shared memory (`/dev/shm`)

The image does not set `VLLM_MQ_MAX_CHUNK_BYTES_MB`, so vLLM's default (16 MiB) applies. With the bundled queue patch, tensor-parallel workers then need about `(TP + 1) × 160 MiB` of `/dev/shm` for vLLM's message queues: 480 MiB at TP=2.

**Deployment via MLIS requires `VLLM_MQ_MAX_CHUNK_BYTES_MB=1`.** MLIS fixes `/dev/shm` at 64 MiB with no way to enlarge it, and vLLM refuses to start when the queues do not fit. At 1 MiB the queues need about 30 MiB at TP=2. Larger messages fall back to a local socket instead of failing. Add the variable to the MLIS deployment's environment.

Deployments that mount a larger memory-backed `/dev/shm` volume need no override.

## Templates available at runtime

| Path | Source |
|------|--------|
| `/templates/qwen3.6-enhanced.jinja` | this image (allanchan339 fix) |
| `/templates/qwen3.5-enhanced.jinja` | this image (allanchan339 fix) |
| `/vllm-workspace/examples/*.jinja` | stock vLLM templates (incl. Gemma 4) |

## Model-specific deployment configs

Operational knowledge — validated serve args, performance figures, and issue history — is documented in a separate internal repo.

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
