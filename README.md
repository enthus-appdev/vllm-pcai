# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) images for **HPE Private Cloud AI (PCAI)** — used by all production, secondary, and experimental model deployments (Qwen3.6-27B, Gemma 4 31B, DeepSeek V4 Flash).

## Why this image exists

PCAI cannot mount volumes through its UI, so anything a model needs at runtime that isn't in the base `vllm/vllm-openai` image **must be baked in**. This image adds five layers on top of the stock vLLM base:

1. **Enhanced chat templates** — Qwen3.5/3.6 hardened templates (hidden historical reasoning, XML tool-call formatting, proper ` response` handling) from [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix), which are *not* in the base image. (Gemma 4 uses vLLM's **in-image** template at `/vllm-workspace/examples/tool_chat_template_gemma4.jinja`.)

2. **Diagnostics endpoint** — `GET /collect_env` on the serving port (same bearer-gate) so PCAI's shell-less pods can still report versions, GPU topology, and env vars.

3. **Vendored patches** — upstream fixes the base image does not carry yet, including the spec-decode drafter weight-source fix ([#48023](https://github.com/vllm-project/vllm/pull/48023), fixing [#42060](https://github.com/vllm-project/vllm/issues/42060)) and the PCAI `/dev/shm` queue-size fix.

4. **Experimental DeepSeek V4 Vision support** — vendors the exact eleven-commit patch series from [vllm#54566](https://github.com/vllm-project/vllm/pull/54566) at head `1576a46008f2411ec51391710c8886293f7a580f`. The image rebuilds vLLM because that PR changes both Python and the compiled MoE routing operator. Do not pass an architecture override; the PR's config converter selects `DeepseekV4ForConditionalGeneration` when it detects the Vision checkpoint. The upstream PR now explicitly rejects speculative decoding for the Vision variant because its image sentinel token IDs are outside the drafter vocabulary, so start this experimental checkpoint without speculative decoding.

5. **Build-time tripwire assertions** — each layer ends with a `RUN python3 -c` that asserts the base image carries the expected parser classes, engine features, and config knobs. A bump that breaks any of them fails **here**, not on a GPU pod.

## Base image: `nightly-44fe2a39`

The `FROM` is a pinned **nightly**, back off the `v0.26.0` release it briefly reached. The DeepSeek-V4 KV-capacity work all landed after the `v0.26.0` branch cut: [#48993](https://github.com/vllm-project/vllm/pull/48993) (packed KV group overlays — per-block cost drops from `sum(groups)` to `max(groups)`) and [#48317](https://github.com/vllm-project/vllm/pull/48317) (a correctness fix to `get_max_concurrency_for_kv_cache_config`, which counted only one group's page size and therefore overstated every concurrency figure). Riding along are the later DeepSeek V4 kernel, DSpark, parser, and multimodal-framework changes through the pinned 2026-08-31 nightly. `v0.26.1rc0` carries the first two KV fixes but is a git tag only — no image is published.

**Nightly tags are pruned after roughly two weeks.** If a rebuild fails on an unresolvable `FROM`, that is the cause; move to the first *release* tag that is a superset rather than silently picking a newer nightly.

**Bumping is not a date comparison.** vLLM cuts release branches, so a later tag can be *missing* commits present in an earlier nightly — #47914 merged 2026-07-08 yet is absent from `v0.25.0` (tagged 07-11). Before any bump, verify the target is a superset:

```bash
gh api repos/vllm-project/vllm/compare/<current-sha-or-tag>...<new-tag> --jq .status   # want: "ahead"
```

## Layout

```
vllm-pcai/
├── Dockerfile                # FROM vllm/vllm-openai:nightly-44fe2a39
│                               + Qwen enhanced templates
│                               + /collect_env diagnostics route
│                               + DeepSeek V4 parser patches
│                               + Build-time tripwires for all three models
├── chat-template-fix/        # git submodule → allanchan339/Qwen templates
├── diag/                     # collect_env_route.py
├── patches/                  # 54566-deepseek-v4-vision.patch (exact upstream PR series)
│                             # 48023-spec-draft-inherit-model-weights.patch (#48023)
└── .dockerignore
```

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
