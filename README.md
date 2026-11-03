# vllm-pcai

Custom [vLLM](https://github.com/vllm-project/vllm) image for HPE Private Cloud AI (PCAI).
This repository builds images; it does not select or update the image running in a model service.

## Sources of truth

| Question | Authoritative source |
| --- | --- |
| Which upstream image and custom patches does this build use? | [`Dockerfile`](Dockerfile): `FROM` pins the base; `COPY patches/…` and the following apply steps define the active patch series and order. Patch files contain the actual changes. |
| How is an image built and published? | [Build workflow](.github/workflows/build.yml). Successful run output and the registry manifest identify the published digest. |
| Which image and arguments should a service use? | The private [serving configuration repository](https://github.com/enthus-appdev/pcai-llm-serving). The [DeepSeek deployment configuration owner](https://github.com/enthus-appdev/pcai-llm-serving/blob/main/models/deepseek-v4-vision/README.md) maintains its selected image and arguments. |
| What is actually running? | The live deployment's container image/image ID and arguments. A published image or configuration document is not evidence that it has been deployed. |

Use the Dockerfile at an image's source revision when inspecting an older image, not the current branch. Deployment references should use immutable registry digests; `latest` and `main` are moving build tags, not declarations of production use.

Patch provenance and purpose belong beside the apply steps in the Dockerfile. Removal decisions, upgrade evidence and historical issue status belong in pull requests and Git history, not a second README inventory. Connector-specific runtime findings and GPU qualification belong with the serving configuration.

## Image additions

- **Chat templates:** the pinned [template submodule](chat-template-fix) provides Qwen templates under `/templates/`. Stock vLLM templates remain under `/vllm-workspace/examples/`.
- **Diagnostics:** [`diag/collect_env_route.py`](diag/collect_env_route.py) exposes `GET /collect_env`. It has no authentication of its own; keep the serving port behind authenticated ingress.
- **Source patches:** applied explicitly in the [Dockerfile](Dockerfile); the presence of a file in [`patches/`](patches) alone does not enable it.
- **Embedding offload:** [`pcai_embed_offload`](plugins/pcai_embed_offload/pcai_embed_offload.py) extends UVA offloading to the DeepSeek V4 embedding table when `embed_tokens` is explicitly requested. It is inactive otherwise.

Inline Dockerfile checks verify selected CPU-accessible behavior. They do not qualify GPU kernels, CUDA graphs, speculative decoding, cache reload correctness or performance.

The separate [EAQ image](experimental/eaq/README.md) has its own [Dockerfile](experimental/eaq/Dockerfile) and pinned base; it is not selected by the default build.

## Build and verify

```bash
git clone --recurse-submodules https://github.com/enthus-appdev/vllm-pcai.git
cd vllm-pcai
python3 -m unittest discover -s tests -v
docker build --platform linux/amd64 -t vllm-pcai:local .
```

Publication targets and tag rules are defined only in the [build workflow](.github/workflows/build.yml). Publishing does not change service configuration or authorize rollout.

## Deployment considerations

The MLIS UI deployment path cannot mount volumes and has constrained `/dev/shm`. Templates are therefore baked into the image, and the queue-sizing patch allows worker response queues to honor `VLLM_MQ_MAX_CHUNK_BYTES_MB`. Choose that setting in the deployment configuration to fit its actual shared-memory allocation; the image does not set it.

Custom-framework deployments can mount PVCs and a larger memory-backed `/dev/shm`. Their chart values, model arguments, connector choice and operational limits are owned by the serving repository, not duplicated here. Different KV connectors require their own qualification; a finding about `OffloadingConnector` does not establish the same defect in `SimpleCPUOffloadConnector`.

[C128 state geometry](docs/c128-state-accounting.md) explains the custom ring allocation. It is not a capacity plan or a claim about the deployed model.

## License

Repository files are Apache-2.0. The enhanced templates retain their upstream license.
