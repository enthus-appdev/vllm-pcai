# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) images for **HPE Private Cloud AI (PCAI)** — used by all production, secondary, and experimental model deployments (Qwen3.6-27B, Gemma 4 31B, DeepSeek V4 Flash).

## Why this image exists

PCAI cannot mount volumes through its UI, so anything a model needs at runtime that isn't in the base `vllm/vllm-openai` image **must be baked in**. This image adds four layers on top of the stock vLLM base:

1. **Enhanced chat templates** — Qwen3.5/3.6 hardened templates (hidden historical reasoning, XML tool-call formatting, proper ` response` handling) from [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix), which are *not* in the base image. (Gemma 4 uses vLLM's **in-image** template at `/vllm-workspace/examples/tool_chat_template_gemma4.jinja`.)

2. **Diagnostics endpoint** — `GET /collect_env` on the serving port (same bearer-gate) so PCAI's shell-less pods can still report versions, GPU topology, and env vars.

3. **Vendored parser patches** — a small set of upstream-before-merge patches that remain open upstream. Currently one: the `deepseek_v4` `add_generation_prompt` / `continue_final_message` honor fix ([#46257](https://github.com/vllm-project/vllm/pull/46257)). Previously carried and now merged: `#45877`, `#46995`, `#46875`.

4. **Build-time tripwire assertions** — each layer ends with a `RUN python3 -c` that asserts the base image carries the expected parser classes, engine features, and config knobs. A bump that breaks any of them fails **here**, not on a GPU pod.

## Base image: `v0.26.0`

The `FROM` is the **`v0.26.0` release**. This image rode pinned `cu129` nightlies from June through July because each capability it needs landed after a tag: the streaming **ParserEngine** (#45413 / #45588 / #45877) so DFlash's large multi-token drafts don't corrupt streaming tool calls in agents; **DFlash** core (#43445) plus **hybrid SWA + full-attention drafters** (#47914); and **DeepSeek-V4 DSpark** (#46995). `v0.26.0` is the first release carrying all of it.

**Bumping is not a date comparison.** vLLM cuts release branches, so a later tag can be *missing* commits present in an earlier nightly — #47914 merged 2026-07-08 yet is absent from `v0.25.0` (tagged 07-11). Before any bump, verify the target is a superset:

```bash
gh api repos/vllm-project/vllm/compare/<current-sha-or-tag>...<new-tag> --jq .status   # want: "ahead"
```

## Layout

```
vllm-pcai/
├── Dockerfile                # FROM vllm/vllm-openai:v0.26.0
│                               + Qwen enhanced templates
│                               + /collect_env diagnostics route
│                               + DeepSeek V4 parser patches
│                               + Build-time tripwires for all three models
├── chat-template-fix/        # git submodule → allanchan339/Qwen templates
├── diag/                     # collect_env_route.py
├── patches/                  # deepseek-add-gen-prompt-on-nightly.patch (#46257)
└── .dockerignore
```

## Templates available at runtime

| Path | Source |
|------|--------|
| `/templates/qwen3.6-enhanced.jinja` | this image (allanchan339 fix) |
| `/templates/qwen3.5-enhanced.jinja` | this image (allanchan339 fix) |
| `/vllm-workspace/examples/*.jinja` | stock vLLM templates (incl. Gemma 4) |

## Model-specific deployment configs

Operational knowledge — validated serve args, performance figures, and issue history — lives in the companion [`pcai-llm-serving`](https://github.com/enthus-appdev/pcai-llm-serving) repo:

- [Qwen3.6-27B-FP8](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/models/qwen3.6-27b-fp8.md)
- [Gemma 4 31B](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/models/gemma-4-31b.md)
- [DeepSeek V4 Flash](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/models/deepseek-v4-flash.md)
- [Upstream tracking](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/upgrades.md)

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
