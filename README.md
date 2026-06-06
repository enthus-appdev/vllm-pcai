# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) image with chat templates **baked in**, for serving on **HPE Private Cloud AI (PCAI)**.

## Why

PCAI cannot mount volumes through its UI, so a chat-template file can't be mounted at runtime. This image ships vLLM's chat templates inside it; you reference them with `--chat-template /templates/<file>.jinja` in the serve args.

## How it works

```
vllm-pcai/
├── Dockerfile        # FROM vllm/vllm-openai:v0.22.0  →  COPY templates from submodule
├── .dockerignore     # keeps the build context to just the templates
├── vllm/             # git submodule → vllm-project/vllm @ v0.22.0 (source of templates)
└── LICENSE
```

The templates are **not copied into this repo** — they come from the pinned `vllm` submodule, so the base image tag and the template set always match. Build copies `vllm/examples/*.jinja` → `/templates/` in the image.

## Clone (submodule must be initialised)

```bash
git clone --recurse-submodules https://github.com/enthus-appdev/vllm-pcai.git
# or, if already cloned:
git submodule update --init
```

## Build & push

```bash
docker build -t ghcr.io/enthus-appdev/vllm-pcai:v0.22.0 .
docker push ghcr.io/enthus-appdev/vllm-pcai:v0.22.0
```

## Use on PCAI

Point your deployment at this image and reference a baked-in template, e.g. for Qwen3.6-27B:

```
Qwen/Qwen3.6-27B-FP8 --served-model-name Qwen3.6-27B --tensor-parallel-size 1 \
  --max-model-len 262144 --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 --mamba-cache-dtype float16 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --chat-template /templates/tool_chat_template_qwen3coder.jinja \
  --port 8080
```

> On PCAI, pass the args as a single space-separated string and keep any JSON values (e.g. `--speculative-config`) minified with no spaces.

## Available templates

All 36 chat templates from vLLM v0.22.0 `examples/` (e.g. `tool_chat_template_qwen3coder.jinja`, `tool_chat_template_hermes.jinja`, `tool_chat_template_mistral.jinja`, `template_chatml.jinja`, …). List them with:

```bash
ls vllm/examples/*.jinja
```

## Upgrade vLLM / templates

Bump the submodule to a new tag and rebuild:

```bash
cd vllm && git fetch --tags && git checkout v0.22.1 && cd ..
# update the Dockerfile FROM tag to match
git commit -am "chore: bump vLLM to v0.22.1"
```

## Customising a template

To apply a fix (e.g. the Qwen3.5/3.6 chat-template robustness patches from
[allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)),
add a patched `.jinja` under a top-level `overrides/` directory and `COPY overrides/*.jinja /templates/`
*after* the submodule copy in the Dockerfile (so it wins). This keeps upstream templates pristine in the submodule.

## License

Repo files: Apache-2.0. Chat templates are from [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) via the `vllm` submodule and retain their upstream license.
