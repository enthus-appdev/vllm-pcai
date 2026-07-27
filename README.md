# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) images for **HPE Private Cloud AI (PCAI)** — the image that serves **Qwen3.6-27B**, **Gemma 4 31B**, and **DeepSeek V4 Flash**, used across all production, secondary, and experimental model deployments.

## Why this image exists

PCAI cannot mount volumes through its UI, so anything a model needs at runtime that isn't in the base `vllm/vllm-openai` image **must be baked in**. This image adds four layers on top of the stock vLLM base:

1. **Enhanced chat templates** — Qwen3.5/3.6 hardened templates (hidden historical reasoning, XML tool-call formatting, proper ` response` handling) from [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix), which are *not* in the base image. (Gemma 4 uses vLLM's **in-image** template at `/vllm-workspace/examples/tool_chat_template_gemma4.jinja` — the baked fix was byte-identical and was dropped.)

2. **Diagnostics endpoint** — `GET /collect_env` on the serving port (same bearer-gate) so PCAI's shell-less pods can still report versions, GPU topology, and env vars.

3. **Vendored parser patches** — a small set of upstream-before-merge patches that remain open upstream. Currently one: the `deepseek_v4` `add_generation_prompt` / `continue_final_message` honor fix ([vLLM #46257](https://github.com/vllm-project/vllm/pull/46257)). Previously carried and now merged: the full `#45877` DeepSeek streaming parser engine port, `#46995` DSpark block-parallel spec decode, and `#46875` general special-token-leak fix.

4. **Build-time tripwire assertions** — each layer ends with a `RUN python3 -c` that asserts the base image carries the expected parser classes, engine features, and config knobs. A bump that breaks any of them fails **here**, not on a GPU pod.

The stock vLLM chat templates are already present at `/vllm-workspace/examples/*.jinja` (vLLM's own Dockerfile does `COPY examples examples`), so this image does **not** re-add them.

## Base image: `v0.26.0` — back on a release tag

The `FROM` is the **`v0.26.0` release**. This image rode pinned `cu129` nightlies from June through July because each capability it needs landed after a tag: the streaming **ParserEngine** (vLLM #45413 / #45588 / #45877) so DFlash's large multi-token drafts don't corrupt streaming tool calls in agents; **DFlash** core (#43445) plus **hybrid SWA + full-attention drafters** (#47914); and **DeepSeek-V4 DSpark** (#46995). `v0.26.0` is the first release carrying all of it, so the nightly's trade-off (less battle-tested, unpinnable by Dependabot) is no longer worth paying.

**Bumping is not a date comparison.** vLLM cuts release branches, so a later tag can be *missing* commits present in an earlier nightly — #47914 merged 2026-07-08 yet is absent from `v0.25.0` (tagged 07-11) because it landed after that branch cut. Before any bump, verify the target is a superset of what's deployed:

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
├── chat-template-fix/        # git submodule → allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix
├── diag/                     # collect_env_route.py — /collect_env endpoint
├── patches/                  # deepseek-add-gen-prompt-on-nightly.patch (vLLM #46257)
└── .dockerignore             # keeps only the .jinja files from the submodule
```

No vLLM submodule — we build *from* vLLM, so its code and stock templates are already present.

## Templates available at runtime

| Path | Source |
|------|--------|
| `/templates/qwen3.6-enhanced.jinja` | this image (allanchan339 fix) |
| `/templates/qwen3.5-enhanced.jinja` | this image (allanchan339 fix) |
| `/vllm-workspace/examples/*.jinja` | stock vLLM templates (incl. Gemma 4) |

## Which model uses which image features?

| Model | Template | Patches | Diagnostics | Tripwires |
|-------|----------|---------|-------------|-----------|
| **Qwen3.6-27B-FP8** | `/templates/qwen3.6-enhanced.jinja` | — | ✓ | DFlash, hybrid SWA |
| **Gemma 4 31B (W4A16)** | `/vllm-workspace/examples/…gemma4.jinja` | — | ✓ | — |
| **DeepSeek V4 Flash** | checkpoint-shipped (`--trust-remote-code`) | #46257 | ✓ | DeepSeek V4 parser, DSpark |

## Use on PCAI

### Qwen3.6-27B-FP8 (primary brain — n8n agents, bots, opencode)

```bash
Qwen/Qwen3.6-27B-FP8 --served-model-name Qwen3.6-27B --tensor-parallel-size 1 \
  --max-model-len 262144 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 16384 --max-num-seqs 24 \
  --kv-cache-dtype auto --enable-prefix-caching --enable-chunked-prefill \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config={"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":12} \
  --chat-template /templates/qwen3.6-enhanced.jinja \
  --attention-backend flash_attn \
  --port 8080
```

> The enhanced template uses XML-style tool calls — match the tool-call parser to it per the [fix repo's docs](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix) rather than assuming `qwen3_coder`. The documented pairing (enhanced template + `qwen3_coder`) is verified live.

### Gemma 4 31B (secondary — overflow / agentic analysis)

```bash
google/gemma-4-31B-it-qat-w4a16-ct --served-model-name gemma-4-31B \
  --tensor-parallel-size 1 --max-model-len 262144 --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --speculative-config={"method":"mtp","model":"google/gemma-4-31B-it-qat-q4_0-unquantized-assistant","num_speculative_tokens":6} \
  --chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja \
  --port 8080
```

### DeepSeek V4 Flash (experimental — frontier-quality, 2× H100 TP=2)

```bash
deepseek-ai/DeepSeek-V4-Flash --served-model-name DeepSeek-V4-Flash \
  --distributed-executor-backend mp --tensor-parallel-size 2 \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs={"thinking":true,"reasoning_effort":"high"} \
  --attention_config.use_fp4_indexer_cache False \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --max-num-seqs 4 --enforce-eager \
  --speculative-config={"method":"dspark","num_speculative_tokens":5} \
  --port 8080
```

## Clone (submodule must be initialised)

```bash
git clone --recurse-submodules https://github.com/enthus-appdev/vllm-pcai.git
# or: git submodule update --init
```

## Build & push

CI builds and pushes automatically (`.github/workflows/build.yml`) to
`ghcr.io/enthus-appdev/vllm-pcai` (`:latest`, `:main`, `:sha-…`; push a `v*` tag for semver tags). Manually:

```bash
docker build -t ghcr.io/enthus-appdev/vllm-pcai:latest .
docker push ghcr.io/enthus-appdev/vllm-pcai:latest
```

## Update the fix

```bash
cd chat-template-fix && git fetch && git checkout <commit-or-tag> && cd ..
git commit -am "chore: bump chat-template-fix"
```

## Related documentation

The operational knowledge for all three models — validated serve args, performance numbers, issue history — lives in the
[`pcai-llm-serving`](https://github.com/enthus-appdev/pcai-llm-serving) repo.

- [Qwen3.6-27B-FP8](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/models/qwen3.6-27b-fp8.md)
- [Gemma 4 31B](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/models/gemma-4-31b.md)
- [DeepSeek V4 Flash](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/models/deepseek-v4-flash.md)
- [Upstream tracking](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/upgrades.md)

## License

Repo files: Apache-2.0. The enhanced templates are from
[allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)
via submodule and retain their upstream license.
