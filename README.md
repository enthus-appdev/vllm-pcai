# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) image for serving Qwen3.x on **HPE Private Cloud AI (PCAI)**.

## Why this image exists

PCAI cannot mount volumes through its UI, so a custom chat template can't be mounted at runtime — it has to be **baked into the image**.

The **stock** vLLM chat templates are already inside `vllm/vllm-openai` at `/vllm-workspace/examples/*.jinja` (vLLM's own Dockerfile does `COPY examples examples`), so this image does **not** re-add them. It only adds the **enhanced Qwen3.5/3.6 templates** from
[allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix), which are *not* in the base image and harden the 27B template (proper `</think>` handling before tool calls, hidden historical reasoning across turns, XML tool-call formatting that avoids premature stop tokens).

## Layout

```
vllm-pcai/
├── Dockerfile           # FROM vllm/vllm-openai:v0.23.0-cu129-ubuntu2404  +  COPY enhanced templates → /templates/
├── chat-template-fix/   # git submodule → allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix
└── .dockerignore        # keeps only the enhanced .jinja in the build context
```

No vLLM submodule — we build *from* vLLM, so its code and stock templates are already present.

## Templates available at runtime

| Path | Source |
|------|--------|
| `/templates/qwen3.6-enhanced.jinja` | this image (allanchan339 fix) |
| `/templates/qwen3.5-enhanced.jinja` | this image (allanchan339 fix) |
| `/vllm-workspace/examples/*.jinja` | stock vLLM templates, already in the base image |

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

## Use on PCAI

Point the deployment at this image and select a baked-in template:

```
Qwen/Qwen3.6-27B-FP8 --served-model-name Qwen3.6-27B --tensor-parallel-size 1 \
  --max-model-len 262144 --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 --mamba-cache-dtype float16 \
  --enable-auto-tool-choice --reasoning-parser qwen3 \
  --chat-template /templates/qwen3.6-enhanced.jinja \
  --port 8080
```

> The enhanced template uses XML-style tool calls — match the tool-call parser to it per the
> [fix repo's docs](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix) rather than assuming `qwen3_coder`.

## DeepSeek-V4-Flash MTP image (experimental, separate package)

A second image — `Dockerfile.deepseek` → `ghcr.io/enthus-appdev/vllm-pcai-deepseek-v4`
— bakes two not-yet-released vLLM fixes ([#44837](https://github.com/vllm-project/vllm/pull/44837),
[#44847](https://github.com/vllm-project/vllm/pull/44847)) needed to serve the
Canada-Quant `DeepSeek-V4-Flash-W4A16-FP8-MTP` checkpoint end-to-end with MTP.
It's based on a **pinned nightly** (the O-proj file only exists on `main`) and
overlays pre-baked patched Python — no CUDA recompile. See
[`vllm-overlay/README.md`](vllm-overlay/README.md). The Qwen image above is
unaffected.

## Update the fix

```bash
cd chat-template-fix && git fetch && git checkout <commit-or-tag> && cd ..
git commit -am "chore: bump chat-template-fix"
```

## License

Repo files: Apache-2.0. The enhanced templates are from
[allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)
via submodule and retain their upstream license.
