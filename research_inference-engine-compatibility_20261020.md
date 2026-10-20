# Alternative inference engines for DeepSeek on two H100 NVL GPUs

**Date:** 2026-10-20

**Depth:** exhaustive

**Confidence:** 96% in the documentary classifications below; **not** a 96% probability that an alternative boots on this appliance.

**Coverage:** 20 web-discovery queries, seven search/retrieval rounds, 55 distinct successfully retrieved repository READMEs, plus model implementations, hardware guides, release/PR states and bug reports. Sources lists 94 distinct linked public URLs; additional bounded issue-query metadata is in the companion evidence.

## Executive Summary

**There are substantially more credible alternatives than TensorFold alone: start the hardware-compatibility shortlist with SGLang, LMDeploy and TensorRT-LLM; add ExLlamaV3, ik_llama.cpp and llama.cpp if converted/quantized weights are acceptable.** FastLLM, DwarfStar/ds4, Arc, vllm.cpp, KTransformers and Fucina are additional real implementations worth distinguishing from generic CUDA-compatible engines. Sonar/Aphrodite is an alternative vLLM-derived fork, not an independent engine architecture. These are candidates for qualification, not verified replacements on our exact two-card configuration. [SGLang](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) · [LMDeploy](https://github.com/InternLM/lmdeploy/pull/4554) · [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md)

**TensorFold is not currently a released native DeepSeek replacement:** its current README explicitly says release binaries do not serve DeepSeek checkpoints; the older V4 recipe belongs to frozen Python 0.6.6, documents MLX on a Mac, and does not establish Hopper compatibility. Conversely, MLX itself now has an official Linux CUDA backend, so dismissing everything MLX-related as Apple-only would also be wrong. [TensorFold README](https://github.com/ashhart/TensorFold/blob/main/README.md) · [legacy recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/deepseek-v4-flash.md) · [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html)

This is a **hardware/model compatibility survey only**. No alternative engine was installed, benchmarked or deployed. Missing KServe recipes, parser parity, performance figures and support for our other models are **not rejection criteria**. Exhaustive means a broad, recorded, bounded search—not a claim to have found every repository or private engine on the internet.

## Key Findings

1. **SGLang and LMDeploy have actual model-specific Hopper paths.** SGLang documents native packed FP4 experts through Hopper-compatible W4A16/W4A8 runners; LMDeploy has merged Hopper support, a current packed-FP4 Triton implementation, and an exact `Flash-0731` test configuration. Neither establishes our exact TP=2 memory fit. [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) · [LMDeploy kernel selection](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/backends/cuda/moe/v4_fp4.py) · [0731 test configuration](https://github.com/InternLM/lmdeploy/blob/main/autotest/configs/deepseek-ai/DeepSeek-V4-Flash-0731.yml)
2. **TensorRT-LLM is not categorically an eight-GPU-only option.** Its V4 README explicitly supports SM90, says GPU minimums depend on precision/cache/batch, and tells users to adjust TP/EP to their world size. Its eight-GPU example is not proof that two H100 NVLs cannot work. [V4 hardware requirements](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md)
3. **Lesser-known engines with genuine V4 implementation evidence exist.** FastLLM documents a V4 two-device TP command; DwarfStar documents CUDA placement and paired expert splitting; Arc publishes a V4 quant artifact with A100/H200 observations; Fucina ships a dedicated V4 runner with optional CUDA GEMM offload. These are different confidence levels, not four proven H100 replacements. [FastLLM V4 guide](https://github.com/ztxz16/fastllm/blob/master/docs/deepseek.md) · [DwarfStar CUDA guide](https://github.com/antirez/ds4/blob/main/docs/CUDA_MULTI_GPU.md) · [Arc artifact](https://huggingface.co/aeonmind/DeepSeek-V4-Flash-UQFF-qtip2/blob/main/README.md) · [Fucina runner](https://github.com/matteo-grella/fucina/blob/main/apps/deepseek4/README.md)
4. **Issue searches materially change classifications.** LightLLM has an open V4 implementation PR; RTP-LLM has an open Hopper-specific implementation PR; ORT GenAI has a draft V4 PR. AirLLM's V4 streaming PR was closed unmerged, and Candle's prototype request was closed as not planned. None should be represented as either fully released support or permanent impossibility. [LightLLM #1593](https://github.com/ModelTC/LightLLM/pull/1593) · [RTP-LLM #1508](https://github.com/alibaba/rtp-llm/pull/1508) · [ORT #2366](https://github.com/microsoft/onnxruntime-genai/pull/2366) · [AirLLM #323](https://github.com/lyogavin/airllm/pull/323) · [Candle #3924](https://github.com/huggingface/candle/issues/3924)
5. **Chitu requires a precision/edition distinction.** Open model documentation lists Flash-FP8 and Flash-Base; native mixed-FP4 Flash is in the pro table, and its YAML has `is_pro: True`. A generic Chitu support claim is therefore insufficient for our current checkpoint and memory envelope. [model matrix](https://github.com/thu-pacman/chitu/blob/public-main/docs/en/SUPPORTED_MODELS.md) · [native Flash configuration](https://github.com/thu-pacman/chitu/blob/public-main/chitu/config/models/DeepSeek-V4-Flash.yaml)
6. **Some exclusions really are hardware-specific.** DeepInfer currently lists SM100 and Ascend 950, while Ember explicitly supports only AMD Strix Halo/gfx1151. These are stronger negative findings for H100 than a missing model-list entry. [DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/main/README.md) · [Ember](https://github.com/otheru-ai/ember/blob/main/README.md)
7. **MLX is no longer categorically Apple-only.** Its official CUDA package supports NVIDIA SM7.5+, which includes Hopper. That does not establish that MLX-LM's V4 PR, exo's V4 path or TensorFold's frozen custom kernels run correctly across two H100s. [MLX CUDA requirements](https://ml-explore.github.io/mlx/build/html/install.html) · [MLX-LM V4 PR #1797](https://github.com/ml-explore/mlx-lm/pull/1797)
8. **No alternative received an exact-appliance compatibility pass.** Published Hopper support, an architecture implementation, a plausible checkpoint size and a two-device option are separate evidence layers. Existing public reports on four H100s or two H200s do not close the two-H100-NVL case. [four-H100 SGLang report](https://forums.developer.nvidia.com/t/running-deepseek-v4-flash-0731-on-a-4xh100-node-using-sglang/384552) · [two-H200 issue](https://github.com/sgl-project/sglang/issues/33397)

## Detailed Analysis

### Q1. What exactly must be compatible?

<a id="hardware-baseline"></a>

The target is **two H100 NVL cards, nominally 94 GB each, Hopper/SM90**, running **`deepseek-ai/DeepSeek-V4-Flash-0731`**. The current vLLM deployment provides a measured reference: approximately **78.99 GiB of model/drafter allocations per GPU** and **7.12 GiB per GPU available to KV**, from the existing 2026-10-18 operational record. These are prior local observations, not fresh measurements during this survey. The record also describes a **192 GiB container RAM limit**, a **100 GB authorized host-offload budget**, and a currently configured **64 GiB CPU KV tier**. The budget is not measured free RAM. No serving endpoint, credential or private repository URL is included here.

An engine must intersect four requirements: **SM90 kernels**, **V4's particular compressed attention/mHC/routing architecture**, **a supported representation of this checkpoint**, and **a placement strategy that fits two cards with working memory**. The physical VRAM is not one flat allocation pool. Nominal GB, binary GiB, file sizes, resident weights and peak allocations must not be interchanged. A generic NVIDIA badge, DeepSeek-V3/R1 support, or BF16 fallback cannot establish that intersection. The model-specific Hopper implementations in Q2 are much stronger evidence than a generic CUDA backend. [LMDeploy packed-expert path](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/backends/cuda/moe/v4_fp4.py) · [TensorRT-LLM V4 cache/precision requirements](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md)

**H100 lacks native Blackwell FP4 tensor instructions, but that does not prohibit packed FP4 storage with unpack/dequantization kernels.** This distinction explains why vLLM already runs the checkpoint and why SGLang and LMDeploy can be plausible alternatives. Conversely, a path that expands routed experts to FP8 may materially enlarge resident weights. The existing vLLM memory figure cannot be transferred to that path. [SGLang Hopper note](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) · [RTP-LLM Hopper proposal](https://github.com/alibaba/rtp-llm/pull/1508)

CPU-heavy candidates also require the **host CPU's ISA, assigned CPU throughput, RAM headroom and storage bandwidth**. We have not established those here. KT-Kernel documents AVX2-or-better CPU variants and explicitly includes SM90/H100; this supports investigating it, not copying a many-core NUMA benchmark onto the current container. [KT-Kernel requirements](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)

### Q2. Which established independent engines deserve first consideration?

**SGLang — high-priority native-checkpoint candidate.** The current cookbook explicitly supports H100/H200, including native FP4 through Marlin/FlashInfer SM90 W4A16 and a newer FlashInfer Humming W4A8 option. Its Flash-0731/DSpark documentation is directly relevant. A September community report reaches inference using the exact checkpoint on **four H100 80 GB cards**, TP4 and `flashinfer_mxfp4`. This is useful executed evidence, but not our two-card test. The cookbook's TP8 H100 cells are verified recipes, not hardware lower bounds. Our remaining question is the smaller-world-size footprint and exact runner selection. [cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) · [four-H100 report](https://forums.developer.nvidia.com/t/running-deepseek-v4-flash-0731-on-a-4xh100-node-using-sglang/384552)

SGLang's reported failures should become **version/backend-specific caution labels**, not a rejection of Hopper. Issue #33397 is open for concurrency corruption and #33549 for long-context DSPARK hangs. The related FlashInfer race report identifies an old **0.6.10–0.6.15.post1** range and reports clean tests on **0.6.18**; it does not prove every downstream corruption report fixed. The current cookbook's newer path matters more than an assumption that those old pins still describe today's package. [SGLang #33397](https://github.com/sgl-project/sglang/issues/33397) · [#33549](https://github.com/sgl-project/sglang/issues/33549) · [FlashInfer #5438](https://github.com/flashinfer-ai/flashinfer/issues/5438)

**LMDeploy — high-priority, and stronger than the earlier survey suggested.** Merged PR #4554 explicitly says its initial V4 implementation was Hopper-only and supports TP and DPEP. Current source selects `TritonFusedMoEV4FP4TPImpl` for the TP path, retaining checkpoint-native packed FP4 weights, and warns that the separate DeepGEMM FP8×FP4 path is SM100-only. The exact 0731 checkpoint has a current PyTorch/NCCL test configuration at TP4. Thus a Blackwell-only assertion about one DeepGEMM operator is not an engine-wide incompatibility. Two-H100 fit and the complete dependency build remain unverified. [merged implementation](https://github.com/InternLM/lmdeploy/pull/4554) · [current dispatch](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/backends/cuda/moe/v4_fp4.py) · [0731 test](https://github.com/InternLM/lmdeploy/blob/main/autotest/configs/deepseek-ai/DeepSeek-V4-Flash-0731.yml)

**TensorRT-LLM — credible, not disqualified by its example topology.** NVIDIA documents V4 in the PyTorch backend on Hopper SM90 and Blackwell, model-specific KV layout `fp8_ds_mla`, and 128/256-token blocks. The README says GPU minimums depend on checkpoint precision, KV, context and batch, and explicitly allows changing TP/EP from the eight-GPU example. It also describes mixed FP4/FP8 checkpoint handling. This is enough to keep it on the compatibility shortlist, but not enough to assert this exact 0731 two-H100 deployment fits. [V4 README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md)

Taken together, Q1 and these three sources support **investigating the two-GPU configuration**, not buying additional GPUs on the strength of example commands and not declaring a migration winner without executing anything.

### Q3. Which quantized, hybrid and lesser-known implementations are real candidates?

**ExLlamaV3 + TabbyAPI** is one of the clearest converted-weight candidates. Its architecture table lists `DeepseekV4ForCausalLM`, its README supports tensor/expert parallelism, and issue #210 identifies a **Flash-0731 EXL3 artifact at 3.04 bpw, reported 117.5 GB**. That is promising for two 94 GB cards by artifact size, but the report's actual generation failure is on SM80, not a success receipt for SM90. Its conversion, placement, KV and peak allocations remain separate checks. TabbyAPI is the recommended API server for this engine, not another independent inference implementation. [ExLlamaV3](https://github.com/turboderp-org/exllamav3/blob/master/README.md) · [V4 issue](https://github.com/turboderp-org/exllamav3/issues/210)

**ik_llama.cpp and llama.cpp** both have real V4 implementations, not merely a generic GGUF claim. ik_llama.cpp lists V4 through merged #2165 and documents CUDA, tensor parallelism and CPU/GPU placement. Upstream llama.cpp has `src/models/deepseek4.cpp`, an exact 0731 chat template, and merged V4 MTP/DSpark work #25784. These are distinct codebases worth tracking, with format/kernel and multi-GPU differences. A suitable GGUF can provide a plausible memory route; no specific quant plus 256K context plus batching combination is qualified on our appliance. [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md) · [V4 implementation](https://github.com/ikawrakow/ik_llama.cpp/pull/2165) · [llama.cpp model](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek4.cpp) · [speculation support](https://github.com/ggml-org/llama.cpp/pull/25784)

**FastLLM** is a particularly relevant lesser-known candidate. Its V4 guide explicitly documents `--tp 0,1`, GPU/NUMA/disk expert placement and bundled DSpark. This is better evidence for a two-device execution path than a generic multi-GPU README. However, its published sparse-attention benchmark is SM120 TP8, and its guide explicitly warns against extrapolating that tuning to other architectures. Issue #666 reports multicuda corruption for a specific FP8-Q4 artifact while single-card output works. Keep it as **model + multi-GPU implemented, exact SM90 path/fit unverified**, not as a proven H100 acceleration. [V4 guide](https://github.com/ztxz16/fastllm/blob/master/docs/deepseek.md) · [benchmark limits](https://github.com/ztxz16/fastllm/blob/master/docs/benchmarks/deepseek_v4.md) · [issue #666](https://github.com/ztxz16/fastllm/issues/666)

**DwarfStar / ds4** is a narrow C engine focused on selected models including V4 Flash. Its CUDA guide supports generic local CUDA builds, multi-card layer placement, and pairs of GPUs splitting routed experts. The published tests cover L40S/Ada and GB10, not our H100 pair. The guide explicitly says an even GPU count alone does not guarantee memory fit. It consumes its own supported GGUF layouts; do not assume every llama.cpp GGUF is interchangeable. This is a genuine **plausible H100 build/placement candidate**, not an H100-validated release. Flandre6/deepseekv4 and other mirrors/forks should not inflate the count of independent engine designs. [DwarfStar](https://github.com/antirez/ds4/blob/main/README.md) · [CUDA guide](https://github.com/antirez/ds4/blob/main/docs/CUDA_MULTI_GPU.md) · [build targets](https://github.com/antirez/ds4/blob/main/Makefile)

**Arc** is a distinct mistral.rs fork, not upstream mistral.rs support. Its V4 qtip2 artifact totals about **74.19 GB**, with **75.7 GB load-only residency reported on A100** and generation observations on **one H200**. The card recommends a practical **96 GB minimum**, slightly above the nominal 94 GB per H100 NVL. It also requires the source checkpoint beside the quant overlay. These facts make Arc interesting but do **not** prove one-card H100 usability or its V4 two-card split. The artifact is based on the original Flash checkpoint rather than automatically establishing exact 0731 equivalence; sparse-indexer behavior and revision-specific fixes are documented. [Arc](https://github.com/aeonmindai/arc/blob/main/README.md) · [artifact and requirements](https://huggingface.co/aeonmind/DeepSeek-V4-Flash-UQFF-qtip2/blob/main/README.md)

**vllm.cpp** is a C++ reimplementation, distinct from the Python vLLM project. Its README claims coherent V4 Flash GGUF execution on GB10, while explicitly describing Hopper/Ampere/Ada builds as **build-tested without runtime proof** and the V4 EXL3 rank-sliced layout as partial. It deserves a development-candidate entry, not a ready H100 verdict. [vllm.cpp hardware/model status](https://github.com/localai-org/vllm.cpp/blob/main/README.md)

**KTransformers / KT-Kernel** explicitly lists V4 Flash and NVIDIA SM90/H100 support. Its current serving integration uses SGLang plus CPU kernels; it is not simply a separate all-GPU replacement. It remains relevant for changing placement and freeing GPU memory even though the current model already fits on vLLM. The missing compatibility evidence is CPU ISA/assigned resources and total host-memory requirements on this particular appliance. [KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/README.md) · [H100/CPU requirements](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)

**Fucina** is another Zig project with actual V4 code, unlike an engine advertising only V3 support. Its dedicated runner documents streamed-expert GGUF execution, MTP and optional Linux CUDA GEMM offload. Its headline is CPU-first execution, including a 164.6 GB Q4K release streamed on a 64 GB host. That proves neither full GPU execution nor two-H100 distribution; CPU/storage behavior is central. Keep it as a research/hybrid candidate, not as hardware-excluded. [Fucina](https://github.com/matteo-grella/fucina/blob/main/README.md) · [V4 runner](https://github.com/matteo-grella/fucina/blob/main/apps/deepseek4/README.md)

The shared lesson is that **quantization can improve memory feasibility without proving kernel compatibility**, while a compatible kernel can still consume too much memory after conversion. These candidates broaden the possibilities in Q2, not the set of already qualified replacements.

### Q4. Where do TensorFold, development branches, wrappers and other hardware fit?

**TensorFold:** the current native README says DeepSeek checkpoints are not served by release binaries. Open #827 tracks a V4.1 lane backend and whole-model serving gate; it must not be misread as released V4-Flash-0731 support. The older Python 0.6.6 V4 recipe describes MLX and a Mac, and explicitly calls CUDA on two DGX Sparks not implemented in that family. H100 compatibility for that frozen Python implementation is unverified, not a substitute for a current native release. **Watch the model gate and an executed CUDA/Hopper receipt**, not just whether H100 appears in a marketing list. [README](https://github.com/ashhart/TensorFold/blob/main/README.md) · [gate #827](https://github.com/ashhart/TensorFold/pull/827) · [legacy recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/deepseek-v4-flash.md)

**Chitu:** native mixed-FP4 Flash is explicitly marked pro; the open table instead includes Flash-FP8 and Flash-Base. The config source marks native Flash `is_pro: True`. This is a conditional software/edition and memory question, not proof that H100 is the wrong GPU. A larger FP8 representation cannot simply inherit our current packed-FP4 fit. [model matrix](https://github.com/thu-pacman/chitu/blob/public-main/docs/en/SUPPORTED_MODELS.md) · [native Flash](https://github.com/thu-pacman/chitu/blob/public-main/chitu/config/models/DeepSeek-V4-Flash.yaml) · [Flash-FP8](https://github.com/thu-pacman/chitu/blob/public-main/chitu/config/models/DeepSeek-V4-Flash-FP8.yaml)

**Sonar / Aphrodite:** the old Aphrodite repository resolves to `dphnAI/sonar`. Its current README explicitly says it is based on vLLM, and source contains a full V4 NVIDIA model/quantization path. This is a genuine alternative fork with plausible compatibility, but it should not count as independent evidence that a different kernel architecture outperforms or even behaves differently from our baseline. Its recipe even references upstream vLLM reports; those are not Sonar-specific executed tests. [Sonar](https://github.com/dphnAI/sonar/blob/main/README.md) · [V4 quantization](https://github.com/dphnAI/sonar/blob/main/aphrodite/models/deepseek_v4/quant_config.py) · [recipe](https://github.com/dphnAI/sonar/blob/main/recipes/models/deepseek-ai/DeepSeek-V4-Flash.yaml)

**Development work:** LightLLM #1593 is open, while the older #1355 closed **without merging**; current main's tree had no V4-named model path in this scan. RTP-LLM #1508 is open and specifically adds Hopper FP8-routed experts because its existing V4 FP8×FP4 kernel path is SM100-only. That branch also raises a memory question on two cards. ORT GenAI #2366 is a draft. AirLLM #323 closed unmerged, so its selective-streaming branch is an experiment, not official current support. Candle #3924 was closed **not planned**, with the author explaining the remaining GPU-format work. These are useful leads rather than absent-from-README dismissals. [LightLLM #1593](https://github.com/ModelTC/LightLLM/pull/1593) · [#1355](https://github.com/ModelTC/LightLLM/pull/1355) · [RTP #1508](https://github.com/alibaba/rtp-llm/pull/1508) · [ORT #2366](https://github.com/microsoft/onnxruntime-genai/pull/2366) · [AirLLM #323](https://github.com/lyogavin/airllm/pull/323) · [Candle #3924](https://github.com/huggingface/candle/issues/3924)

**MLX-LM and exo:** MLX's Linux CUDA package documents SM7.5+ support, so H100 clears the framework-level requirement. Upstream MLX-LM's V4-Flash PR #1797 remains open; the current main tree scan found a V4.1 file, not a V4-Flash file. exo has MLX V4 model cards and uses MLX as its backend. Neither framework-level CUDA support nor Apple's published V4 examples prove exo/MLX-LM's exact V4 model path works on two H100s. Classify **CUDA hardware possible; exact V4 path unverified/development**, not Apple-only. [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html) · [V4 PR](https://github.com/ml-explore/mlx-lm/pull/1797) · [exo](https://github.com/exo-explore/exo/blob/main/README.md)

**Actual architecture mismatches:** DeepInfer's published release supports **SM100 or Ascend 950** and V4.1 Flash, not our published SM90/current-model combination. Ember explicitly restricts its implementation to **gfx1151/Strix Halo**. PowerServe's documented accelerated path targets **Qualcomm QNN/Snapdragon**, not H100. This is stronger exclusion evidence than the unknown V4 support of generic CUDA engines. [DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/main/README.md) · [Ember](https://github.com/otheru-ai/ember/blob/main/README.md) · [PowerServe](https://github.com/powerserve-project/PowerServe/blob/main/README.md)

**Wrappers and model frameworks:** Dynamo, Triton, Xinference, Ollama and Parallax must be evaluated through the underlying execution backend. Transformers has a V4 model implementation but is a model/runtime framework, not automatically a memory-efficient serving replacement. Nano-deepseek-v4 explicitly disclaims a production FP4/official-0731 speculative runtime. Their presence in this survey is coverage, not an inflated count of independent candidate engines. [Dynamo](https://github.com/ai-dynamo/dynamo/blob/main/README.md) · [Triton](https://github.com/triton-inference-server/server/blob/main/README.md) · [Transformers V4](https://huggingface.co/docs/transformers/model_doc/deepseek_v4) · [nano-deepseek-v4](https://github.com/hebo1221/nano-deepseek-v4/blob/main/README.md)

### Q5. What can and cannot be concluded without executing candidates?

The answer is a **priority-ordered compatibility backlog**, not a list of engines guaranteed to work. Published model-specific Hopper support gives SGLang, LMDeploy and TensorRT-LLM their position. Existing converted V4 artifacts and multi-GPU implementations justify the quantized/hybrid candidates. Explicit model gates or architectural restrictions justify the hold/mismatch categories. The absence of a recipe justifies only **unverified**. These distinctions follow directly from Q1–Q4, not from relative popularity.

The minimum future hardware qualification would identify an exact engine commit/build and checkpoint representation, confirm two visible SM90 GPUs and the chosen operator backends, load with conservative working-memory settings, and complete actual inference on both ranks without hidden weight expansion or unsupported fallback. That is a description of missing evidence, **not instructions executed by this report**. CPU/storage candidates additionally need measured container headroom and host capabilities. CUDA toolkit/driver ABI checks remain engine-build-specific; the current successful vLLM CUDA stack is encouraging, not blanket approval of arbitrary binaries.

## Comparison

`Candidate` means documentary reason to investigate. `Conditional` means a branch, conversion, edition or hardware-specific step is unresolved. No alternative below has an exact-appliance pass from this research.

| Engine/path | V4 evidence | H100/SM90 evidence | Two-card/memory status | Compatibility disposition |
|---|---|---|---|---|
| **SGLang** | Exact 0731 cookbook and executed report | Explicit Hopper kernels; four-H100 execution | TP2/94GB pair unverified | **First-tier candidate** |
| **LMDeploy** | Merged V4; exact 0731 test config | Explicit Hopper and packed-FP4 TP source | Test config TP4; TP2 unverified | **First-tier candidate** |
| **TensorRT-LLM** | V4 PyTorch documentation | Explicit SM90 and Hopper KV layout | Eight GPUs are an example; two-card fit unresolved | **First-tier candidate** |
| **ExLlamaV3** | V4 architecture; 0731 EXL3 artifact | CUDA implementation; no exact H100 receipt here | 117.5GB artifact promising across two cards; runtime unverified | **Strong converted-weight candidate** |
| **ik_llama.cpp** | Merged V4 implementation | CUDA/Turing+ documented | Quant/placement-dependent | **Converted-weight candidate** |
| **llama.cpp** | V4 model + 0731 template + speculation | CUDA implementation | Quant/placement-dependent | **Converted-weight candidate** |
| **FastLLM** | V4 guide, DSpark, two-device TP command | CUDA; published V4 sparse tuning is SM120 | Exact SM90 kernels, artifact and fit unverified | **Investigate** |
| **DwarfStar/ds4** | Dedicated V4 engine | Generic CUDA path; Ada/GB10 executed | CUDA pairs/layer placement; H100 pair unverified | **Investigate** |
| **Arc** | V4 UQFF/qtip2 implementation | A100 load/H200 generation reports | 96GB practical minimum quoted; 94GB card/two-rank path unverified | **Investigate cautiously** |
| **KTransformers** | V4 native-format/hybrid work | KT-Kernel explicitly supports H100/SM90 | CPU ISA/RAM/placement unresolved | **Hybrid candidate** |
| **vllm.cpp** | V4 GGUF executed on GB10 | Hopper builds, explicitly no runtime proof | Partial EXL3 path; two-card proof absent | **Experimental candidate** |
| **Fucina** | Dedicated V4 GGUF runner | Optional Linux CUDA GEMM offload | CPU/storage-first; no two-H100 proof | **Experimental/hybrid candidate** |
| **Sonar/Aphrodite** | V4 NVIDIA source/recipe | vLLM-derived CUDA path | Exact current fork/TP2 unverified | **Alternative fork, not independent engine** |
| **Chitu** | Open FP8/Base; native Flash pro | NVIDIA support, no exact two-card receipt here | Precision/edition and memory unresolved | **Conditional candidate** |
| **LightLLM** | Open V4 PR #1593 | Proposed implementation, not release qualification | Unknown | **Development watch** |
| **RTP-LLM** | V4 exists; Hopper PR #1508 open | Existing routed FP4 path SM100; proposed Hopper FP8 | Weight expansion may exceed two cards | **Development watch** |
| **MLX-LM / exo** | V4 PR/MLX model cards | MLX CUDA supports SM7.5+ | V4 CUDA and two-H100 path unverified | **Development/portability watch** |
| **TensorFold** | No released native DeepSeek serving | Generic CUDA engine is not V4 proof | Model gate precedes fit question | **Not released for this task** |
| **DeepInfer** | V4.1, not current V4-0731 | Published SM100/Ascend only | Wrong published architecture | **Hardware mismatch today** |
| **Ember** | V4-0731/vision-specific | Explicit gfx1151-only | Wrong GPU backend | **Hardware mismatch today** |

The important distinction is **native-checkpoint feasibility versus converted-weight feasibility**. The first tier avoids making a lower-bit re-quant a prerequisite, while ExLlamaV3/GGUF/Arc can create different and potentially easier memory envelopes. Their published artifacts do not make their quality, peak memory or kernel behavior identical to the current checkpoint. This is a representation distinction, not a quality rejection criterion.

The table also prevents two common overclaims: a Hopper-compatible engine is not automatically a two-H100-compatible deployment, and a failure in one optional backend is not an engine-wide prohibition. SGLang and LMDeploy show why those distinctions matter; RTP-LLM, Arc and the experimental engines show why they must be retained rather than replaced by simple yes/no badges.

## Contradictions & Debates

- **“Eight H200s are the minimum for TensorRT-LLM” versus the actual README.** The README explicitly describes topology as tunable and minimums as resource-dependent. **Resolution:** eight GPUs are the documented bring-up example, not a proved minimum or a two-card rejection.
- **“SGLang requires converted FP8 on Hopper” versus current native-FP4 paths.** The cookbook documents both W4A16 and Humming W4A8 for native packed FP4. **Resolution:** retain SGLang; qualify the exact runner/precision combination instead of expanding experts by assumption.
- **“LMDeploy is Blackwell-only” versus current dispatch.** Its legacy DeepGEMM FP8×FP4 path is SM100-only; its TP implementation selects packed-FP4 Triton and initial V4 support explicitly targeted Hopper. **Resolution:** backend restriction, not whole-engine restriction.
- **“TensorFold supports DeepSeek” versus released native serving.** The recipe is explicitly for frozen Python 0.6.6, while the current README refuses DeepSeek release-binary serving. **Resolution:** distinguish implementation generations and model gates; no native CUDA/H100 pass.
- **“MLX is Apple-only” versus current installation documentation.** The official Linux CUDA package supports SM7.5+. **Resolution:** do not hardware-exclude MLX; still require V4 operator and multi-GPU proof.
- **“mistral.rs supports V4 because a UQFF exists” versus the artifact's own instructions.** The card requires **Arc**, and says an upstream mistral.rs binary will not read qtip2. **Resolution:** separate the fork from upstream.
- **“Chitu lists V4, so the current native checkpoint is open-supported” versus its edition/config tables.** Native Flash is pro; open Flash-FP8 is a different representation. **Resolution:** keep both edition and footprint qualifications.
- **“Closed issue/PR means implemented.”** Candle's V4 request closed not-planned and AirLLM #323 closed unmerged; llama.cpp #25784 and ik_llama.cpp #2165 actually merged. **Resolution:** inspect merge/state reason, not the word closed.

Sources for these resolutions are the primary links in Q2–Q4; they are not unresolved votes between equally strong claims. In particular, search-provider summaries that inferred “no two-H100 deployment found → cannot run” were rejected as unsupported.

## Uncertainties & Gaps

- **No alternative's exact two-H100-NVL deployment was executed.** Memory fit, backend selection and completed inference remain open, even for first-tier candidates.
- **Exact checkpoint matters.** Original Flash, Flash-Base, Flash-0731, Vision and V4.1 are not interchangeable. Arc/DwarfStar/Fucina and some converted artifacts do not automatically establish 0731 equivalence.
- **CPU and memory constraints remain measured-evidence gaps.** A 100GB authorization is not free memory; API-process RSS is not total container usage; historical shm limits do not categorically block all CPU offload.
- **MLX's CUDA support does not settle the V4 path.** Metal-specific custom operators, model branches and distributed communication may require additional work. exo is not proved compatible merely because MLX can use CUDA.
- **Build-only means build-only.** vllm.cpp explicitly separates Hopper compilation from runtime proof; DwarfStar's generic architecture target is not an H100 benchmark.
- **Community receipts are not universal guarantees.** Four-H100 SGLang success, H200 Arc generation and SM80 ExLlama failures each describe their own configurations.
- **Source branches and issues move.** All documentary statuses are as inspected on 2026-10-20. Selected commit/blob references and PR merge states are recorded in the companion JSON.
- **Negative searches are bounded.** No V4 match in a README/model listing and recorded issue query means no positive evidence found in that scan—not permanently unsupported hardware or model architecture.
- **Coverage is not literal omniscience.** The 55-repository appendix includes wrappers, libraries, historical forks and research prototypes; it is not 55 independent production-ready V4 engines. Private, proprietary and newly created engines can be missing.
- **Initial access failures were not treated as negative evidence.** Old Aphrodite issue scopes returned 422 before its canonical Sonar identity was found; initial ScaleLLM/NanoFlow/PowerServe probes were corrected. Those failures are excluded from successful-coverage counts.

## Recommendations

### Primary recommendation

For the next **hardware-compatibility investigation**, prioritize **SGLang and LMDeploy**, followed by **TensorRT-LLM**. Each has model-specific Hopper evidence. Record their exact native-0731 representation, kernel selection and TP2 memory requirements before deciding whether a trial is worth production downtime. This priority is an evidence-based research recommendation, **not authorization to deploy or a claim that one is faster**.

### Alternatives

If changing weight representation is acceptable, prioritize **ExLlamaV3**, **ik_llama.cpp** and **llama.cpp**. Keep **FastLLM and DwarfStar** as the most directly documented additional multi-GPU implementations, and **Arc** as the single-Hopper-family quantization lead whose 94GB/two-rank case still needs resolution. **KTransformers and Fucina** are genuine hybrid options whose compatibility depends on host capabilities. **Sonar** deserves a separate fork comparison only if moving to another vLLM-derived stack is itself attractive.

### Not currently qualified

Do not select **TensorFold's released native binary** for the current model based on its historical recipe. Do not select **DeepInfer or Ember** for this H100 setup on their published hardware contracts. Do not call **RTP-LLM/LightLLM/ORT/AirLLM/Candle development work** a released compatibility pass. Keep unverified general-purpose engines in the discovery appendix rather than deleting them or declaring them impossible.

## Methodology

- **Depth/type:** exhaustive comparative compatibility survey, with discovery of unfamiliar engines.
- **Search:** 20 web queries across seven search/retrieval rounds; direct GitHub README/model-tree/issue/PR retrieval supplemented web search. Invalid repo scopes were corrected where possible; no authorization failures or empty searches became incompatibility claims.
- **Baseline:** existing dated local operational records for the H100 NVL pair and Flash-0731; no fresh production probing in this task.
- **Evidence hierarchy:** published hardware/model contracts and implemented dispatch paths > release/merge status > configuration-specific executed community reports > generic compatibility badges > search snippets. Issue state is authoritative metadata; performance/correctness observations inside issues remain reporter evidence.
- **Multi-hop chains:** generic engine discovery → actual model registry → quantization/kernel dispatch → hardware backend → multi-device strategy → issue/release status. Examples: Chitu's model list → pro flag; LMDeploy's DeepGEMM error → alternative Triton dispatch; Arc artifact → fork identity; MLX Apple branding → actual CUDA installation contract.
- **Final confidence:** 96% average documentary-classification confidence from the research checkpoint (Q1 96%, Q2/TensorFold 97%, quantized/obscure paths 94%, other families 95%, evidence limits 97%). This does not supply missing runtime evidence.
- **Search scope changes:** the source set exceeded the skill's usual 15–30-source planning band because the explicitly exhaustive discovery request led to 55 codebases and important branch/format distinctions. The 20-query discovery budget was retained; additional direct source retrieval resolved the MLX CUDA counterexample.
- **Research only:** no model downloads, dependency installs, GPU tests, production changes, commits or pushes. The report/evidence are local uncommitted artifacts; existing deployment and earlier research files are untouched.

## Appendix: complete 55-repository discovery inventory

This is a repository-level inventory, **not 55 independent engine recommendations**. Redirected duplicates and failed probes are excluded. “No evidence found” is a bounded research outcome; the successful issue-query scopes, selected PR states and source-tree pins are recorded in the [companion evidence](research_evidence/inference-engine-compatibility_20261020.json).

| # | Repository | Classification | Hardware/model finding |
|---:|---|---|---|
| 1 | [aeonmindai/arc](https://github.com/aeonmindai/arc/blob/master/README.md) | Experimental quantized candidate | V4 qtip2; A100/H200 observations; 94GB-per-card and two-H100 path unverified. Fork, not upstream mistral.rs. |
| 2 | [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/blob/main/README.md) | Orchestration/backend layer | H100 compatibility and V4 precision/footprint inherit selected SGLang/vLLM/TRT-LLM workers. |
| 3 | [alibaba/rtp-llm](https://github.com/alibaba/rtp-llm/blob/main/README.md) | Development watch | V4 exists; SM90 routed-expert support proposed by open PR1508; FP8 expansion and two-card fit unresolved. |
| 4 | [antirez/ds4](https://github.com/antirez/ds4/blob/main/README.md) | Independent CUDA/GGUF candidate | V4 Flash and paired CUDA execution documented; Ada/GB10 receipts, no H100-pair receipt. |
| 5 | [ashhart/tensorfold](https://github.com/ashhart/TensorFold/blob/main/README.md) | Native model gate / watch | Current README says no release binary serves DeepSeek; legacy Python MLX recipe is not native Hopper proof. |
| 6 | [deepseek-ai/deepinfer](https://github.com/deepseek-ai/DeepInfer/blob/main/README.md) | Published hardware/model mismatch | Current release supports SM100/Ascend950 and V4.1, not SM90/current Flash0731. |
| 7 | [deepspeedai/deepspeed](https://github.com/deepspeedai/DeepSpeed/blob/master/README.md) | No verified V4 serving path | Training and generic GPU inference capabilities are not a complete V4-0731 SM90 serving implementation; issue hits included training work. |
| 8 | [deepspeedai/deepspeed-mii](https://github.com/deepspeedai/DeepSpeed-MII/blob/main/README.md) | No V4 evidence found in scan | NVIDIA serving engine; README and recorded issue scan did not establish V4-0731 support. |
| 9 | [dphnai/sonar](https://github.com/dphnAI/sonar/blob/main/README.md) | Alternative vLLM-derived fork | Canonical successor to old Aphrodite URL; explicit V4 NVIDIA source and recipes; exact TP2 remains unverified. |
| 10 | [efeslab/nanoflow](https://github.com/efeslab/Nanoflow/blob/Nanoflow-python/README.md) | No V4 evidence found in scan | NVIDIA throughput-serving research with Llama/Qwen support; README and issue scan did not establish V4. |
| 11 | [ericlbuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs/blob/master/README.md) | No upstream V4 evidence beyond wishlist | H100 support shown for other models; V4 qtip2 artifact requires separate Arc fork. |
| 12 | [exo-explore/exo](https://github.com/exo-explore/exo/blob/main/README.md) | MLX-backed conditional path | V4 MLX model cards/work exist; MLX itself supports CUDA, but exo V4 on H100/TP2 unverified. |
| 13 | [flexflow/flexflow-serve](https://github.com/flexflow/flexflow-serve/blob/inference/README.md) | No V4 evidence found in scan | GPU/distributed serving capabilities; no V4 support established by README/issue scan. |
| 14 | [fminference/flexllmgen](https://github.com/FMInference/FlexLLMGen/blob/main/README.md) | Historical offload engine; no V4 evidence | Canonical archived successor of FlexGen URL; GPU/CPU/disk execution is not V4 architecture support. |
| 15 | [foxengine-ai/aphrodite](https://github.com/FoxEngine-ai/aphrodite/blob/main/README.md) | Historical fork; not another candidate family | Old Aphrodite-derived repository; no independent current V4/H100 qualification. Do not double-count with Sonar. |
| 16 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/README.md) | Converted-weight candidate | V4 implementation, 0731 template and merged MTP/DSpark; suitable quant/placement and two-H100 working memory unresolved. |
| 17 | [ghyathmoussa/rllm](https://github.com/ghyathmoussa/rLLM/blob/main/README.md) | No V4 evidence found in scan | Rust CUDA engine with H100-oriented claims; no V4-named model path or positive V4 issue result in bounded scan. |
| 18 | [gradienthq/parallax](https://github.com/GradientHQ/parallax/blob/main/README.md) | Distributed/backend layer | Heterogeneous/distributed inference architecture; no standalone V4 SM90 qualification established. |
| 19 | [hebo1221/nano-deepseek-v4](https://github.com/hebo1221/nano-deepseek-v4/blob/main/README.md) | Reference/conformance implementation | Real V4 equations; explicitly not a production FP4/official0731 speculative serving runtime. |
| 20 | [huggingface/candle](https://github.com/huggingface/candle/blob/main/README.md) | Prototype lead closed not-planned | V4 prototype issue3924 closed not-planned; GPU dtype work remained. Not a released implementation pass. |
| 21 | [huggingface/optimum-nvidia](https://github.com/huggingface/optimum-nvidia/blob/main/README.md) | TensorRT-related integration | No separate V4 engine qualification established; follow the underlying TensorRT runtime/model support. |
| 22 | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference/blob/main/README.md) | No current V4 path established | Historical CUDA serving project; generic supported models do not establish V4 compressed attention/checkpoint support. |
| 23 | [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/README.md) | Model implementation/framework | Official V4 model documentation; optimized SM90 serving, packed-weight placement and TP2 fit require separate evidence. |
| 24 | [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md) | Converted-weight candidate | Merged V4 PR2165, CUDA and TP; exact quant/backend/two-H100 memory qualification missing. |
| 25 | [infinitensor/infinilm](https://github.com/InfiniTensor/InfiniLM/blob/main/README.md) | No V4 evidence found in scan | Multi-backend inference project; no positive V4 model evidence established by README/issue scan. |
| 26 | [intel/ipex-llm](https://github.com/intel/ipex-llm/blob/main/README.md) | Different accelerator focus / no path established | Intel-oriented runtime; no verified NVIDIA H100 V4 execution path in inspected documentation. |
| 27 | [internlm/lmdeploy](https://github.com/InternLM/lmdeploy/blob/main/README.md) | First-tier native-checkpoint candidate | Merged Hopper V4, current packed-FP4 Triton TP path, exact0731 test config; two-H100 fit unverified. |
| 28 | [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers/blob/main/README.md) | Hybrid candidate | V4 Flash plus KT-Kernel SM90/H100 support; CPU ISA, assigned compute and RAM footprint unresolved. |
| 29 | [localai-org/vllm.cpp](https://github.com/localai-org/vllm.cpp/blob/main/README.md) | Experimental independent C++ candidate | V4 GGUF runtime proof on GB10; Hopper builds explicitly lack runtime proof; partial EXL3 path. |
| 30 | [lyogavin/airllm](https://github.com/lyogavin/airllm/blob/main/README.md) | Unmerged branch lead | V4-0731 selective-expert streaming PR323 closed unmerged; do not call it current main support. |
| 31 | [matteo-grella/fucina](https://github.com/matteo-grella/fucina/blob/main/README.md) | Experimental CPU/hybrid candidate | Dedicated V4 GGUF runner and optional CUDA GEMM offload; no complete two-H100 GPU-execution proof. |
| 32 | [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/blob/main/README.md) | Draft implementation watch | Draft V4 PR2366; CUDA execution provider alone does not establish current V4 support or memory fit. |
| 33 | [ml-explore/mlx](https://github.com/ml-explore/mlx/blob/main/README.md) | Framework hardware eligible; model path unresolved | Official Linux CUDA backend supports SM7.5+, including H100; not categorically Apple-only. |
| 34 | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/README.md) | Development/model-path watch | V4 Flash PR1797 open; main model-tree scan contained V4.1 instead; V4 CUDA/TP2 unverified. |
| 35 | [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm/blob/main/README.md) | No V4 evidence found in scan | CUDA-capable compiler/runtime; no positive V4 model support found in README/issue scan. |
| 36 | [modelscope/dash-infer](https://github.com/modelscope/dash-infer/blob/main/README.md) | No V4 evidence found in scan | README explicitly includes H100/SM90a, but no V4 architecture support established by issue scan. |
| 37 | [modeltc/lightllm](https://github.com/ModelTC/LightLLM/blob/main/README.md) | Open implementation watch | V4 PR1593 open; older1355 closed unmerged; no released/main V4 pass established. |
| 38 | [neuralmagic/deepsparse](https://github.com/neuralmagic/deepsparse/blob/main/README.md) | CPU-oriented; no H100 V4 path established | CPU sparse-inference runtime, not an evidenced H100 V4 replacement. |
| 39 | [nvidia/fastertransformer](https://github.com/NVIDIA/FasterTransformer/blob/main/README.md) | Legacy GPU engine; no V4 path established | Historical CUDA model implementations do not establish current V4-0731 support. |
| 40 | [nvidia/tensorrt-llm](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md) | First-tier native-checkpoint candidate | Explicit V4 SM90 support and tunable topology; examples are not proved GPU minima; exact0731 TP2 fit unknown. |
| 41 | [ollama/ollama](https://github.com/ollama/ollama/blob/main/README.md) | Backend packaging/server | Do not infer V4 support from a library label; verify bundled model implementation/format and GPU placement. |
| 42 | [openbmb/cpm.cu](https://github.com/OpenBMB/CPM.cu/blob/main/README.md) | No V4 path established | Compact CUDA inference implementation; inspected README did not establish V4 architecture/checkpoint execution. |
| 43 | [opencpil/prima.cpp](https://github.com/OpenCPIL/prima.cpp/blob/main/README.md) | No V4 evidence found in scan | Distributed/heterogeneous inference; generic DeepSeek claims did not establish V4 in README/issue scan. |
| 44 | [opennmt/ctranslate2](https://github.com/OpenNMT/CTranslate2/blob/master/README.md) | No V4 evidence found in scan | CUDA-capable inference library; no V4 support established in README/issue scan. |
| 45 | [otheru-ai/ember](https://github.com/otheru-ai/ember/blob/main/README.md) | Explicit hardware mismatch | DeepSeekV4 engine explicitly restricted to AMD Strix Halo gfx1151; not NVIDIA H100. |
| 46 | [paddlepaddle/fastdeploy](https://github.com/PaddlePaddle/FastDeploy/blob/develop/README.md) | No V4 evidence found in scan | NVIDIA and other accelerator support; no positive V4 evidence found in README/recorded issue scan. |
| 47 | [powerserve-project/powerserve](https://github.com/powerserve-project/PowerServe/blob/main/README.md) | Different accelerated hardware target | Documented Qualcomm QNN/Snapdragon path; no H100 V4 path established. |
| 48 | [sgl-project/sglang](https://github.com/sgl-project/sglang/blob/main/README.md) | First-tier native-checkpoint candidate | Explicit Hopper packed-FP4 runners and exact0731 support; four-H100 report; exact two-H100-NVL fit unverified. |
| 49 | [thu-pacman/chitu](https://github.com/thu-pacman/chitu/blob/public-main/README.md) | Precision/edition-conditional candidate | Open FlashFP8/Base; native mixed-FP4 Flash marked pro; SM90 two-card fit unresolved. |
| 50 | [tiiny-ai/powerinfer](https://github.com/Tiiny-AI/PowerInfer/blob/main/README.md) | No current-model path established | Original activation-sparse engine lists ReLU model families, not V4's native architecture; not a two-H100 V4 pass. |
| 51 | [triton-inference-server/server](https://github.com/triton-inference-server/server/blob/main/README.md) | Serving/backend layer | GPU/model compatibility inherits chosen execution backend; not an independent V4 implementation. |
| 52 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3/blob/master/README.md) | Strong converted-weight candidate | Explicit V4 model; 0731 EXL3 artifact reported117.5GB; two-card runtime/H100 proof remains open. |
| 53 | [vectorch-ai/scalellm](https://github.com/vectorch-ai/ScaleLLM/blob/main/README.md) | No V4 evidence found in scan | GPU production-serving engine; no V4 support established by README/recorded issue scan. |
| 54 | [xorbitsai/inference](https://github.com/xorbitsai/inference/blob/main/README.md) | Multi-engine serving layer | Xinference hosts underlying engines; assess V4 precision/H100 placement in the selected backend. |
| 55 | [ztxz16/fastllm](https://github.com/ztxz16/fastllm/blob/master/README.md) | Independent multi-GPU candidate | V4 guide documents two-device TP and DSpark; exact SM90 kernels/fit unverified; specific multicuda bug reported. |

### Additional vendor-only lead

[TensorFoundry Forge](https://tensorfoundry.io/products/forge) claims validated H100 support, but describes itself as **single-GPU** and lists generic DeepSeek rather than a verified V4-Flash-0731 path. The page is an early-access vendor claim, not a public exact-model/multi-GPU compatibility receipt. Keep as an unresolved lead rather than hardware-incompatible or production-qualified. Forge is not included in the 55-repository count.

## Sources

Dates below are **inspection dates**, not invented publication dates. Publication/merge dates are given in the analysis or machine receipts where relevant. Tier 1 denotes primary project documentation/code/PR metadata; Tier 3 denotes configuration-specific community reports. A primary project's own compatibility claim is not independent runtime validation.

| # | Source | URL | Inspected | Credibility |
|---:|---|---|---|---|
| 1 | SGLang | https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx | 2026-10-20 | Tier 1 primary |
| 2 | LMDeploy | https://github.com/InternLM/lmdeploy/pull/4554 | 2026-10-20 | Tier 1 primary |
| 3 | TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md | 2026-10-20 | Tier 1 primary |
| 4 | TensorFold README | https://github.com/ashhart/TensorFold/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 5 | legacy recipe | https://github.com/ashhart/TensorFold/blob/main/docs/recipes/deepseek-v4-flash.md | 2026-10-20 | Tier 1 primary |
| 6 | MLX installation | https://ml-explore.github.io/mlx/build/html/install.html | 2026-10-20 | Tier 1 primary |
| 7 | LMDeploy kernel selection | https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/backends/cuda/moe/v4_fp4.py | 2026-10-20 | Tier 1 primary |
| 8 | 0731 test configuration | https://github.com/InternLM/lmdeploy/blob/main/autotest/configs/deepseek-ai/DeepSeek-V4-Flash-0731.yml | 2026-10-20 | Tier 1 primary |
| 9 | FastLLM V4 guide | https://github.com/ztxz16/fastllm/blob/master/docs/deepseek.md | 2026-10-20 | Tier 1 primary |
| 10 | DwarfStar CUDA guide | https://github.com/antirez/ds4/blob/main/docs/CUDA_MULTI_GPU.md | 2026-10-20 | Tier 1 primary |
| 11 | Arc artifact | https://huggingface.co/aeonmind/DeepSeek-V4-Flash-UQFF-qtip2/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 12 | Fucina runner | https://github.com/matteo-grella/fucina/blob/main/apps/deepseek4/README.md | 2026-10-20 | Tier 1 primary |
| 13 | LightLLM #1593 | https://github.com/ModelTC/LightLLM/pull/1593 | 2026-10-20 | Tier 1 primary |
| 14 | RTP-LLM #1508 | https://github.com/alibaba/rtp-llm/pull/1508 | 2026-10-20 | Tier 1 primary |
| 15 | ORT #2366 | https://github.com/microsoft/onnxruntime-genai/pull/2366 | 2026-10-20 | Tier 1 primary |
| 16 | AirLLM #323 | https://github.com/lyogavin/airllm/pull/323 | 2026-10-20 | Tier 1 primary |
| 17 | Candle #3924 | https://github.com/huggingface/candle/issues/3924 | 2026-10-20 | Tier 3 report / primary issue state |
| 18 | model matrix | https://github.com/thu-pacman/chitu/blob/public-main/docs/en/SUPPORTED_MODELS.md | 2026-10-20 | Tier 1 primary |
| 19 | native Flash configuration | https://github.com/thu-pacman/chitu/blob/public-main/chitu/config/models/DeepSeek-V4-Flash.yaml | 2026-10-20 | Tier 1 primary |
| 20 | DeepInfer | https://github.com/deepseek-ai/DeepInfer/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 21 | Ember | https://github.com/otheru-ai/ember/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 22 | MLX-LM V4 PR #1797 | https://github.com/ml-explore/mlx-lm/pull/1797 | 2026-10-20 | Tier 1 primary |
| 23 | four-H100 SGLang report | https://forums.developer.nvidia.com/t/running-deepseek-v4-flash-0731-on-a-4xh100-node-using-sglang/384552 | 2026-10-20 | Tier 3 report / primary issue state |
| 24 | two-H200 issue | https://github.com/sgl-project/sglang/issues/33397 | 2026-10-20 | Tier 3 report / primary issue state |
| 25 | KT-Kernel requirements | https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md | 2026-10-20 | Tier 1 primary |
| 26 | #33549 | https://github.com/sgl-project/sglang/issues/33549 | 2026-10-20 | Tier 3 report / primary issue state |
| 27 | FlashInfer #5438 | https://github.com/flashinfer-ai/flashinfer/issues/5438 | 2026-10-20 | Tier 3 report / primary issue state |
| 28 | ExLlamaV3 | https://github.com/turboderp-org/exllamav3/blob/master/README.md | 2026-10-20 | Tier 1 primary |
| 29 | V4 issue | https://github.com/turboderp-org/exllamav3/issues/210 | 2026-10-20 | Tier 3 report / primary issue state |
| 30 | ik_llama.cpp | https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 31 | V4 implementation | https://github.com/ikawrakow/ik_llama.cpp/pull/2165 | 2026-10-20 | Tier 1 primary |
| 32 | llama.cpp model | https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek4.cpp | 2026-10-20 | Tier 1 primary |
| 33 | speculation support | https://github.com/ggml-org/llama.cpp/pull/25784 | 2026-10-20 | Tier 1 primary |
| 34 | benchmark limits | https://github.com/ztxz16/fastllm/blob/master/docs/benchmarks/deepseek_v4.md | 2026-10-20 | Tier 1 primary |
| 35 | issue #666 | https://github.com/ztxz16/fastllm/issues/666 | 2026-10-20 | Tier 3 report / primary issue state |
| 36 | DwarfStar | https://github.com/antirez/ds4/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 37 | build targets | https://github.com/antirez/ds4/blob/main/Makefile | 2026-10-20 | Tier 1 primary |
| 38 | Arc | https://github.com/aeonmindai/arc/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 39 | vllm.cpp hardware/model status | https://github.com/localai-org/vllm.cpp/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 40 | KTransformers | https://github.com/kvcache-ai/ktransformers/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 41 | Fucina | https://github.com/matteo-grella/fucina/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 42 | gate #827 | https://github.com/ashhart/TensorFold/pull/827 | 2026-10-20 | Tier 1 primary |
| 43 | Flash-FP8 | https://github.com/thu-pacman/chitu/blob/public-main/chitu/config/models/DeepSeek-V4-Flash-FP8.yaml | 2026-10-20 | Tier 1 primary |
| 44 | Sonar | https://github.com/dphnAI/sonar/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 45 | V4 quantization | https://github.com/dphnAI/sonar/blob/main/aphrodite/models/deepseek_v4/quant_config.py | 2026-10-20 | Tier 1 primary |
| 46 | recipe | https://github.com/dphnAI/sonar/blob/main/recipes/models/deepseek-ai/DeepSeek-V4-Flash.yaml | 2026-10-20 | Tier 1 primary |
| 47 | #1355 | https://github.com/ModelTC/LightLLM/pull/1355 | 2026-10-20 | Tier 1 primary |
| 48 | exo | https://github.com/exo-explore/exo/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 49 | PowerServe | https://github.com/powerserve-project/PowerServe/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 50 | Dynamo | https://github.com/ai-dynamo/dynamo/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 51 | Triton | https://github.com/triton-inference-server/server/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 52 | Transformers V4 | https://huggingface.co/docs/transformers/model_doc/deepseek_v4 | 2026-10-20 | Tier 1 primary |
| 53 | nano-deepseek-v4 | https://github.com/hebo1221/nano-deepseek-v4/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 54 | aeonmindai/arc | https://github.com/aeonmindai/arc/blob/master/README.md | 2026-10-20 | Tier 1 primary |
| 55 | alibaba/rtp-llm | https://github.com/alibaba/rtp-llm/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 56 | deepspeedai/deepspeed | https://github.com/deepspeedai/DeepSpeed/blob/master/README.md | 2026-10-20 | Tier 1 primary |
| 57 | deepspeedai/deepspeed-mii | https://github.com/deepspeedai/DeepSpeed-MII/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 58 | efeslab/nanoflow | https://github.com/efeslab/Nanoflow/blob/Nanoflow-python/README.md | 2026-10-20 | Tier 1 primary |
| 59 | ericlbuehler/mistral.rs | https://github.com/EricLBuehler/mistral.rs/blob/master/README.md | 2026-10-20 | Tier 1 primary |
| 60 | flexflow/flexflow-serve | https://github.com/flexflow/flexflow-serve/blob/inference/README.md | 2026-10-20 | Tier 1 primary |
| 61 | fminference/flexllmgen | https://github.com/FMInference/FlexLLMGen/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 62 | foxengine-ai/aphrodite | https://github.com/FoxEngine-ai/aphrodite/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 63 | ggml-org/llama.cpp | https://github.com/ggml-org/llama.cpp/blob/master/README.md | 2026-10-20 | Tier 1 primary |
| 64 | ghyathmoussa/rllm | https://github.com/ghyathmoussa/rLLM/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 65 | gradienthq/parallax | https://github.com/GradientHQ/parallax/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 66 | huggingface/candle | https://github.com/huggingface/candle/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 67 | huggingface/optimum-nvidia | https://github.com/huggingface/optimum-nvidia/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 68 | huggingface/text-generation-inference | https://github.com/huggingface/text-generation-inference/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 69 | huggingface/transformers | https://github.com/huggingface/transformers/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 70 | infinitensor/infinilm | https://github.com/InfiniTensor/InfiniLM/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 71 | intel/ipex-llm | https://github.com/intel/ipex-llm/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 72 | internlm/lmdeploy | https://github.com/InternLM/lmdeploy/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 73 | lyogavin/airllm | https://github.com/lyogavin/airllm/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 74 | microsoft/onnxruntime-genai | https://github.com/microsoft/onnxruntime-genai/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 75 | ml-explore/mlx | https://github.com/ml-explore/mlx/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 76 | ml-explore/mlx-lm | https://github.com/ml-explore/mlx-lm/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 77 | mlc-ai/mlc-llm | https://github.com/mlc-ai/mlc-llm/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 78 | modelscope/dash-infer | https://github.com/modelscope/dash-infer/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 79 | modeltc/lightllm | https://github.com/ModelTC/LightLLM/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 80 | neuralmagic/deepsparse | https://github.com/neuralmagic/deepsparse/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 81 | nvidia/fastertransformer | https://github.com/NVIDIA/FasterTransformer/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 82 | nvidia/tensorrt-llm | https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 83 | ollama/ollama | https://github.com/ollama/ollama/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 84 | openbmb/cpm.cu | https://github.com/OpenBMB/CPM.cu/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 85 | opencpil/prima.cpp | https://github.com/OpenCPIL/prima.cpp/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 86 | opennmt/ctranslate2 | https://github.com/OpenNMT/CTranslate2/blob/master/README.md | 2026-10-20 | Tier 1 primary |
| 87 | paddlepaddle/fastdeploy | https://github.com/PaddlePaddle/FastDeploy/blob/develop/README.md | 2026-10-20 | Tier 1 primary |
| 88 | sgl-project/sglang | https://github.com/sgl-project/sglang/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 89 | thu-pacman/chitu | https://github.com/thu-pacman/chitu/blob/public-main/README.md | 2026-10-20 | Tier 1 primary |
| 90 | tiiny-ai/powerinfer | https://github.com/Tiiny-AI/PowerInfer/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 91 | vectorch-ai/scalellm | https://github.com/vectorch-ai/ScaleLLM/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 92 | xorbitsai/inference | https://github.com/xorbitsai/inference/blob/main/README.md | 2026-10-20 | Tier 1 primary |
| 93 | ztxz16/fastllm | https://github.com/ztxz16/fastllm/blob/master/README.md | 2026-10-20 | Tier 1 primary |
| 94 | TensorFoundry Forge | https://tensorfoundry.io/products/forge | 2026-10-20 | Tier 1 primary |
