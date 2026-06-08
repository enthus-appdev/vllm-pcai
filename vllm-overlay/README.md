# vllm-overlay — DeepSeek-V4-Flash MTP fixes

Pre-baked Python files that patch two not-yet-released vLLM fixes onto a pinned
nightly base image (see `../Dockerfile.deepseek`). Pure-Python, no CUDA rebuild.

## What's patched

| File | From PR | Fix |
|------|---------|-----|
| `deepseek_v4/nvidia/mtp.py` | [vllm#44837](https://github.com/vllm-project/vllm/pull/44837) | pass `prefix=` to MTP `e_proj`/`h_proj` so compressed-tensors can match the layer |
| `deepseek_v4/nvidia/ops/o_proj.py` | [vllm#44847](https://github.com/vllm-project/vllm/pull/44847) | BF16 MTP O-proj fallback for the unquantized BF16 draft tower |

Raw PR diffs are kept in `patches/` for provenance. Only the **NVIDIA** path is
overlaid (H100 target); `amd/` and `xpu/` are intentionally left stock.

## Pinned commit

`VLLM_SHA = 303916e93d66da301231c9aee80489951a5cd8f6`

Both patch preimage blobs match this commit exactly, so the patches apply
cleanly with no 3-way merge. The base image is the prebuilt nightly for the
same commit (`vllm/vllm-openai:cu129-nightly-<VLLM_SHA>`).

## Regenerating after a VLLM_SHA bump

```bash
SHA=<new-commit-on-main-that-has-nvidia/ops/o_proj.py>
W=$(mktemp -d); cd "$W"; mkdir -p vllm/models/deepseek_v4/nvidia/ops
curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/$SHA/vllm/models/deepseek_v4/nvidia/mtp.py"        -o vllm/models/deepseek_v4/nvidia/mtp.py
curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/$SHA/vllm/models/deepseek_v4/nvidia/ops/o_proj.py" -o vllm/models/deepseek_v4/nvidia/ops/o_proj.py
# refresh patches/ via `gh pr diff 44837/44847` if the PRs were updated, then:
git apply -p1 --include='vllm/models/deepseek_v4/nvidia/*'        patches/44837-mtp-prefix.patch
git apply -p1 --include='vllm/models/deepseek_v4/nvidia/ops/o_proj.py' patches/44847-bf16-o-proj.patch
# copy the two files back over this directory and bump VLLM_SHA in Dockerfile.deepseek
```

If a future nightly already contains these fixes (PRs merged), delete this
overlay and just base on that nightly (or a release) directly.
