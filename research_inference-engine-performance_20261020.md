# DeepSeek engine performance comparison: context, prefill, generation and serving

**Date:** 2026-10-20

**Depth:** exhaustive

**Confidence:** 95% in the evidence classification and comparison limits—not a 95% probability that an untested engine is faster or fits this appliance.

**Target:** two H100 NVL cards, nominally 94 GB each; `DeepSeek-V4-Flash-0731`; current reference vLLM 0.32.0. The comparison covers the baseline and 18 alternative paths from the preceding compatibility survey. All speed figures below are prior local observations or attributed published reports; this research executed no inference.

**Sources:** 38 linked public source URLs, seven research rounds, 20 web queries; 22 structured records and 19 comparison paths in the [companion ledger](research_evidence/inference-engine-performance_20261020.json).

Read alongside the [hardware/model compatibility survey](research_inference-engine-compatibility_20261020.md).

## Executive Summary

**There is no defensible universal winner for your exact two-H100 deployment. SGLang is the strongest first performance comparison; ExLlamaV3/TabbyAPI is the most interesting converted-weight/context-efficiency experiment.** The closest published same-hardware Hopper comparison found SGLang+DSpark faster for one request, but vLLM faster at concurrency 8/32 and on repeated-prefix TTFT. LMDeploy and TensorRT-LLM have credible V4/Hopper implementations, but neither has a demonstrated current-0731/two-H100 performance advantage in the sources inspected. [Hopper comparison](https://github.com/sgl-project/sglang/issues/37348) · [ExLlamaV3 evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) · [LMDeploy V4](https://github.com/InternLM/lmdeploy/pull/4554) · [TensorRT-LLM V4](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md)

Among the less-known implementations, **DwarfStar, FastLLM and ik_llama.cpp have concrete performance evidence**, but on different hardware/placements. Arc's measured results are much less impressive than its theoretical projections; KTransformers demonstrates large contexts at the cost of very long cold prefills on its CPU-heavy reference configuration. TensorFold has an interesting community fork reporting roughly 2× improvements on two Sparks—but that is V4.1, different hardware and a different branch, not released native TensorFold serving your current checkpoint. [DwarfStar](https://github.com/antirez/ds4/blob/main/docs/PERFORMANCE.md) · [FastLLM](https://github.com/ztxz16/fastllm/blob/master/docs/benchmarks/deepseek_v4.md) · [ik toolkit](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/README.md) · [Arc measurements](https://github.com/aeonmindai/arc/blob/master/docs/FLEET.md) · [KTransformers long context](https://ktransformers.net/en/docs/inference/long-context-deployment) · [TensorFold community fork](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/README.md)

**“1M context” is not a useful ranking by itself.** It can mean a model limit, a configured server limit, a shared cache pool, an estimated admission limit, or a genuinely filled request. The report separates those cases and does not convert Blackwell/Mac/CPU figures into predicted H100 speeds.

## Key Findings

1. **SGLang is a real competitor, not an established upgrade.** A same-checkpoint 4×H800 report gives single-stream decode of 221 tok/s for SGLang+DSpark versus 149.8 for tuned vLLM 0.26; at concurrency 32 the respective per-stream figures are 39.0 and 58.6. Versions, tuning and some timing details differ. [report](https://github.com/sgl-project/sglang/issues/37348)
2. **Warm-prefix latency can matter more than peak decode.** That report measured hot-prefix TTFT of 520 ms for SGLang+DSpark versus 238 ms for vLLM, with both reporting 99.2% prefix hits. A cache-hit percentage alone does not establish lower TTFT. [report](https://github.com/sgl-project/sglang/issues/37348)
3. **ExLlamaV3 has unusually concrete large-context evidence.** A pinned single-RTX-PRO-6000 report filled 249,985 prompt tokens: 2,265 prompt tok/s, 79.25 decode tok/s after filling context, and a 262,144-token allocation. It is a 2.04-bpw quant on Blackwell, not H100/native weights. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md)
4. **Speculative decoding is not automatically a win in every configuration.** The same ExLlamaV3 report's mode labelled MTP raised short decode from 95.33 to 180.61 tok/s but reduced its lightweight score from 22/22 to 15/22; DwarfStar's V4 session-batched path does not use DSpark. Neither observation proves correctly implemented speculation is intrinsically lossy. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) · [DwarfStar serving](https://github.com/antirez/ds4/blob/main/docs/SERVER.md)
5. **Hardware compatibility does not imply current draft-head parity.** LMDeploy's V4 loader skips `mtp.*` weights. TensorRT-LLM documents V4 MTP, not a verified equivalent of your bundled 0731 DSpark path. A base engine needs enough kernel improvement to compensate if it loses an effective drafter. [LMDeploy source](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/models/deepseek_v4.py) · [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md)
6. **Large context can be technically available but operationally slow.** KTransformers' 1M-configured four-5090 recipe reports roughly 20–25 minutes of prefill for a 600K prompt and requires radix caching off in that configuration. This is not a prediction for an all-GPU H100 setup; it illustrates the danger of comparing context limits without latency. [guide](https://ktransformers.net/en/docs/inference/long-context-deployment)
7. **Do not confuse aggregate throughput with one user's speed.** Arc reports 111.69 aggregate decode tok/s at batch 256, while per-user p50 is 0.53 tok/s; at batch one it reports 17.99 per-user tok/s. SGLang's cookbook also labels one throughput field as input-plus-output/GPU rather than generation. [Arc](https://github.com/aeonmindai/arc/blob/master/docs/BENCHMARKS.md) · [SGLang data definition](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx)
8. **Most alternatives lack a usable same-target numerical comparison.** Sonar, Chitu, LMDeploy, TensorRT-LLM, Fucina and the development branches cannot honestly be ordered by H100/V4 tokens/sec from the evidence found. Their strengths, missing measurements and model-specific limitations are still comparable; missing data is not zero performance.
9. **TensorFold's community result is meaningful but narrowly scoped.** It reports 56.8 prose/92.2 code tok/s and a filled ~1.035M-token V4.1 request on two GB10 Sparks. Its 2× claim is against an older vLLM kit using the same files, not your vLLM 0.32 H100 configuration, and draft settings differ. [fork report](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/README.md)

## Detailed Analysis

### Q1. What context is actually available?

A useful context claim needs five labels: **model/checkpoint, engine/build, per-request configured window, actual populated prompt length, and memory/placement**. “The model supports 1,048,576 tokens” answers only the first question. A configured limit must also reserve output tokens; a 262,144-token window is not permission to send a 262,144-token prompt and then generate a long answer. [V4 model/runtime documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md)

<a id="local-baseline"></a>

**Your local baseline, not remeasured in this task:** the existing 2026-10-18 operational record reports vLLM 0.32.0 configured for **262,144 tokens**, max sequences **6**, and a GPU KV budget of **1,062,087 tokens**, with a boot-reported full-window concurrency of **4.05×**. Six admitted sequences do not mean six simultaneous full windows. The 64 GiB CPU tier has demonstrated restores, not an expansion of each request's advertised window or a controlled latency benefit. Prior local memory limits are 192 GiB and a 100 GB authorized offload budget—not measured free RAM. The companion ledger records this local provenance without publishing private links.

| Path | Strongest context evidence found | What it does **not** establish |
|---|---|---|
| **vLLM baseline** | 262,144 configured locally; ~1.062M shared GPU-token pool, prior operational observation | Fresh current-version speed figures, 6 full windows, or a qualified 1M request |
| **SGLang** | Model-specific 1M support/recipes; a matched Hopper comparison configures 262K | A filled 1M request or TP2 capacity on this appliance. [docs](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) |
| **LMDeploy** | Implemented V4 compressed attention and exact-0731 TP4 test configuration | A measured long-context/H100 pair capacity. [test config](https://github.com/InternLM/lmdeploy/blob/main/autotest/configs/deepseek-ai/DeepSeek-V4-Flash-0731.yml) |
| **TensorRT-LLM** | Documents the model's 1M target and specialized cache manager; initial example uses 4K | A 4K engine ceiling, or a measured two-H100 1M deployment. [guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md) |
| **ExLlamaV3/TabbyAPI** | **249,985 prompt tokens filled**, 262,144 allocation, one Blackwell card; also four 262K slots in an agent run | A 1M single request: four slots totaling 1M cache are different. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) |
| **ik_llama.cpp** | Toolkit's 512K-configured profile measured generation at **256K depth**, with host experts/KV and local patches | A filled 512K run, all-GPU speed, or a two-H100 result. [profile](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/README.md) |
| **llama.cpp** | A pinned 0731 GGUF filled **249,985 prompt tokens**, allocated 262K, on one Blackwell card | Same performance/quality as EXL3 or native weights. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) |
| **FastLLM** | Guide exposes a 131,072-token configuration; published speed point is only 128 input/256 output | A tested filled 128K/H100 context. [guide](https://github.com/ztxz16/fastllm/blob/master/docs/deepseek.md) |
| **DwarfStar/ds4** | Recorded V4 sweep reaches **65,536 depth**; CUDA multi-GPU serving example configures 100K and 16 slots | Sixteen simultaneously filled 100K requests or a universal 1M H100 pass. [sweep](https://github.com/antirez/ds4/blob/main/docs/PERFORMANCE.md) · [CUDA](https://github.com/antirez/ds4/blob/main/docs/CUDA_MULTI_GPU.md) |
| **Arc** | Benchmark server configured **545,000**; quantitative batch sweep uses only ~68-token prompts | A filled 545K request. Earlier long-context quality tests predate a subsequent YaRN correction. [benchmarks](https://github.com/aeonmindai/arc/blob/master/docs/BENCHMARKS.md) |
| **vllm.cpp** | V4 GGUF decode measured on GB10; long-context/two-H100 service not established here | That the example's model is qualified through every production-server route. [benchmarks](https://github.com/localai-org/vllm.cpp/blob/main/docs/benchmarks/dwarfstar-gguf.md) |
| **KTransformers** | 1M configured; ~600K prompt performance described; default example's KV pool is **964,864**, below 1M | A default full-1M admission pass or fit within our host budget. [guide](https://ktransformers.net/en/docs/inference/long-context-deployment) |
| **Fucina** | Dedicated V4 streamed-expert runner | A demonstrated filled large-context H100 serving window. [runner](https://github.com/matteo-grella/fucina/blob/main/apps/deepseek4/README.md) |
| **Sonar/Aphrodite** | V4 model/recipe and vLLM-derived cache infrastructure | Independent two-H100 filled-window measurements. [recipe](https://github.com/dphnAI/sonar/blob/main/recipes/models/deepseek-ai/DeepSeek-V4-Flash.yaml) |
| **Chitu** | V4 model variants/configurations; native Flash listed pro, open Flash-FP8 different footprint | Measured V4 context/concurrency on this pair. [matrix](https://github.com/thu-pacman/chitu/blob/public-main/docs/en/SUPPORTED_MODELS.md) |
| **LightLLM** | Open implementation PR covers compressed attention, state and multi-tier caches | Released model/window qualification. [PR](https://github.com/ModelTC/LightLLM/pull/1593) |
| **RTP-LLM** | Hopper PR reports a **32K prompt** performance case on disaggregated H20s | Current native-FP4 two-H100 fit or 1M performance. [PR](https://github.com/alibaba/rtp-llm/pull/1508) |
| **TensorFold** | Frozen Python/MLX V4 recipe tests through **128K**; 349,184 is an admission estimate. Separate V4.1 fork reports an actual **1,035,020-token prompt** | Current released native V4-0731/H100 support; the two code lines are not interchangeable. [legacy](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/deepseek-v4-flash.md) · [fork](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/README.md) |
| **MLX-LM/exo** | Framework CUDA exists, but V4-Flash model/branch and H100 execution remain unqualified here | A model-specific H100 window from the generic CUDA requirement. [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html) · [V4 PR](https://github.com/ml-explore/mlx-lm/pull/1797) |

The most useful large-window alternative evidence is therefore **EXL3/GGUF's actually filled ~250K cases**, not the mere presence of “1M” in a config. They show genuine operational capability on other hardware. They do not prove a larger usable window than your current 262K deployment, and their quantization changes the comparison.

The second lesson is that **cache capacity and throughput interact**. The ik toolkit trades prefix caching/checkpoints away in its large-window profile; KTransformers disables radix reuse; ExLlamaV3 demonstrates multiple allocated slots but not four concurrently saturated 250K requests. Q2 and Q4 explain why a nominally larger window may worsen your ordinary multi-turn workload.

### Q2. How good is prompt processing and time to first token?

**Cold prefill, cached continuation and queueing must be separated.** TTFT includes admission/queue time, cache lookup/restore, prompt execution, first-token generation and transport. Dividing prompt tokens by TTFT is not automatically pure model prefill speed. DwarfStar explicitly measures newly added intervals at increasing context depth; that is useful continued-prefill evidence but not a cold full-document TTFT. [DwarfStar methodology](https://github.com/antirez/ds4/blob/main/docs/PERFORMANCE.md)

| Reported configuration—not an engine leaderboard | Prefill / prompt processing | Latency/context qualification |
|---|---|---|
| **Your historical vLLM**, 2×H100 NVL | **6,860 tok/s** at the local table's 65,536-depth point | TTFT **9.685 s**, `--no-cache`, dated **2026-07-31**, not current 0.32; benchmark depth is not an independently recounted total prompt length |
| **SGLang**, original Flash, H100 TP8 catalog | Pure prefill rate not reported in the cited row | **205 ms TTFT**, 8,192 input/1,024 output, C1, v0.5.15—not 0731/TP2. [data](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx) |
| **ExLlamaV3 2.04 bpw**, one RTX PRO 6000 | **2,265.18 tok/s** for ~250K filled prompt | Retrieval request **110.774 s end-to-end**, not separately measured TTFT; no thinking/no speculation. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) |
| **llama.cpp IQ2XXS**, same RTX PRO 6000 | **395.19 tok/s** for same reported ~250K retrieval probe | **632.934 s end-to-end**; different quant/representation from EXL3, so not an isolated engine comparison. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) |
| **ik_llama.cpp patched toolkit**, one RTX PRO 6000 + host streaming | **1,359 tok/s at 256K depth** | 512K configured, CPU-resident experts/KV; caching off. [profile](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/README.md) |
| **DwarfStar Q2**, one GB10 Spark | **825.76 at 2K → 822.98 at 64K tok/s** | Continued 2,048-token intervals, not full cold prompts. [baseline](https://github.com/antirez/ds4/blob/main/docs/PERFORMANCE.md) |
| **Arc qtip2b**, one H200 | **171.58 tok/s**, server-instrumented compute at C1 | Only ~68 prompt tokens, TTFT **0.375 s**; not a long-prefill measurement. [benchmarks](https://github.com/aeonmindai/arc/blob/master/docs/BENCHMARKS.md) |
| **KTransformers hybrid**, 4×5090 + large CPU host | Guide reports roughly **590 → 320 tok/s** as context grows | ~600K prefill **20–25 minutes**; host resources and placement unlike ours. [guide](https://ktransformers.net/en/docs/inference/long-context-deployment) |
| **TensorFold frozen Python V4**, M3 Ultra 256 GB | **362 at 32K / 260 at 128K tok/s** | Actual old Mac/MLX path, not CUDA/H100. [recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/deepseek-v4-flash.md) |
| **TensorFold community V4.1 fork**, two Sparks | **2,179 tok/s at 32K; 1,266 at ~1.035M** | Largest prompt TTFT about **13.6 minutes**; different model/branch. [report](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/README.md) |
| **RTP-LLM Hopper PR**, 8×H20 split prefill/decode | No isolated pp rate needed to interpret cited result | **32K TTFT 1.7742 → 1.5880 s**, internal branch optimization, not an engine A/B. [PR](https://github.com/alibaba/rtp-llm/pull/1508) |
| **LMDeploy, TensorRT-LLM, Sonar, Chitu, Fucina, LightLLM, MLX-LM/exo** | No suitable V4/H100-pair prefill benchmark found | Unknown—not zero and not proof the implementation cannot run |

These rows must not be sorted into a numerical engine ranking: some are full cold prompts, one is an incremental interval, others are tiny-prompt or configured-window cases. Even the useful same-Blackwell EXL3/GGUF comparison changes quantization and representation alongside the engine. Its roughly 5.7× prompt-processing difference is a **recipe result**, not proof of an intrinsic 5.7× engine advantage.

For your agentic use, **repeated-prefix TTFT is at least as important as cold prefill**. The same-Hopper report below measures cold/hot prefixes separately. KTransformers' long-context configuration and ik's largest-window profile explicitly give up prefix reuse, which can erase the practical attraction of a larger window when every tool turn resends a growing history. Q4 examines those interactions.

### Q3. How good is generation, both single-user and concurrent?

The strongest directly comparative Hopper evidence found is the community **4×H800 SM90** report. It uses the exact 0731 checkpoint, a 262K configured window, short ~16-token prompts with 200–400 generated tokens, and a separate repeated-prefix scenario. SGLang is an August nightly; the reference is vLLM 0.26 tuned for weeks. vLLM's full speculation settings and raw streaming timing harness are not specified, so this is **reported recipe evidence**, not a perfectly controlled engine-only experiment. [source and conditions](https://github.com/sgl-project/sglang/issues/37348)

| Same 4×H800 node | Decode per stream, C1 | C8 | C32 | Cold-prefix TTFT | Hot-prefix TTFT |
|---|---:|---:|---:|---:|---:|
| **vLLM 0.26 reference** | 149.8 tok/s | **100.6** | **58.6** | **3.588 s** | **0.238 s** |
| **SGLang + DSpark** | **221.0** | 93.2 | 39.0 | 6.470 s | 0.520 s |
| **SGLang target-only TP4** | 144.0 | 77.1 | 35.4 | 5.472 s | 0.523 s |

Within that report, SGLang+DSpark is about **48% faster at C1**, while vLLM's per-stream figure is about **50% higher at C32** than SGLang+DSpark. These ratios describe these reported rows only. They do not predict H100 NVL, two cards, newer Humming kernels, vLLM 0.32 or your traffic. A later H200 comment's FP8-weight-memory arithmetic is internally suspect, so its suggested conversion/topology is not adopted as a fit claim.

Other published generation observations:

| Engine / reported setup | Single-request generation | Concurrency evidence / caveat |
|---|---|---|
| **SGLang H100 TP8 catalog**, original Flash | TPOT **3.19 ms**, equivalent to **~313 tok/s** by reciprocal arithmetic | Original Flash/v0.5.15, not 0731/TP2; the separate 319 input+output tok/s/GPU field is not decode. [data](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx) |
| **SGLang vendor FP8 recipe**, 8×H100 | Reported **39 tok/s/user at C32**, not a C1 result | **1,262 aggregate output tok/s**, mean TTFT 2.463 s, p99 32.221 s. Original Flash/EAGLE, not your native 0731. [vendor](https://lambda.ai/inference-models/deepseek-ai/deepseek-v4-flash) |
| **ExLlamaV3 2.04 bpw**, one RTX PRO 6000 | **95.33 short / 79.25 after ~250K** | Four agent workers completed a 300-case run using four 262K slots; not four simultaneous full-window stress. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) |
| **llama.cpp IQ2XXS**, same RTX PRO 6000 | **78.16 short** | Single-stream result; speculative support exists separately. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) |
| **ik_llama.cpp host-streamed MXFP4**, one RTX PRO 6000 | **16.8 at 256K depth** | Not an all-GPU engine ceiling. Another all-resident low-bit speculative profile reports ~87 tok/s under different conditions. [profiles](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/README.md) |
| **FastLLM**, TP8/SM120 | **99.93**, 128 input/256 output | GPU SKU/interconnect/build date absent; source's 86.57 “total throughput” matches output tokens / full duration, not input+output. [benchmark](https://github.com/ztxz16/fastllm/blob/master/docs/benchmarks/deepseek_v4.md) |
| **DwarfStar Q2**, one Spark | **18.05 at 2K → 13.84 at 64K** | Separate 8×L40S Q4 example reports **~126 aggregate** with 16 sessions; V4 batching does not also use DSpark. [sweep](https://github.com/antirez/ds4/blob/main/docs/PERFORMANCE.md) · [CUDA](https://github.com/antirez/ds4/blob/main/docs/CUDA_MULTI_GPU.md) |
| **Arc qtip2b**, one H200 | **17.99 per-user p50** at C1 | **111.69 aggregate at C256**, but **0.53 per-user p50**; p95 TTFT 58.5 s. [benchmarks](https://github.com/aeonmindai/arc/blob/master/docs/BENCHMARKS.md) |
| **vllm.cpp**, one GB10, paired same-GGUF test | **18.69** with resident weights vs **16.33** for ds4 | About **14.45%** higher in this paired test; no comparison to Python vLLM on this model/H100. [paired test](https://github.com/localai-org/vllm.cpp/blob/main/docs/benchmarks/dwarfstar-gguf.md) |
| **KTransformers**, one 5090 plus CPU experts | Tutorial says **20+ tok/s**, 16K configured | Different host/placement; long-context guide's second request falls back to eager rather than captured batch-1 graphs. [tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md) |
| **Fucina**, CPU-first streamed Q4K | **1.5–3.6 warm tok/s** with 20 GB expert budget | Not CUDA/H100 performance. [runner](https://github.com/matteo-grella/fucina/blob/main/apps/deepseek4/README.md) |
| **TensorFold frozen Python V4**, M3 Ultra | **49.3 chat / 70.8 code**, short greedy fixture; **34 after 128K** | Four streams reported **63 aggregate**, not four times single-user speed. [recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/deepseek-v4-flash.md) |
| **TensorFold community V4.1 fork**, two Sparks | **56.8 prose / 92.2 code** | **140.3 aggregate code at C4**; distinct model/branch, not released native V4/H100 support. [fork](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/README.md) |
| **RTP-LLM open Hopper PR**, 8×H20 | TPOT **224.1 → 28.2 ms**, reciprocal of latter **~35.5 tok/s** | C4, 4+4 PD split, no speculation; an intra-branch optimization rather than superiority to vLLM. [PR](https://github.com/alibaba/rtp-llm/pull/1508) |

Code/prose, reasoning mode, output length, cache state and speculative acceptance all change these results. The TensorFold fork itself reports higher code than prose speed because acceptance differs. A CPU-streaming ik result cannot bound its speed when weights are resident across two cards, just as an 8-GPU fleet aggregate cannot tell you what one interactive user receives.

Finally, older local notes record **400 nonstreamed output tokens in 1.201 s**, about **333 output tokens/s over that request's measured duration**. This is historical orientation, not a fresh vLLM 0.32 decode-stage benchmark; it includes unspecified prompt/transport work. Old first-to-last-SSE-chunk figures of thousands of tokens/s were explicitly rejected. Speculative bursts make stream chunks an unsafe timing unit unless actual generated-token counts and the measurement interval are defined.

### Q4. Which cache, offload and speculative features actually help this model?

**The feature combination matters more than individual checkmarks.** An engine can implement speculation and batching without supporting them together for V4; it can expose CPU offload for weights rather than KV; it can restore a prefix's tokens by recomputing rather than importing all cached GPU state. Those differences change both usable context and turn latency. [DwarfStar serving contracts](https://github.com/antirez/ds4/blob/main/docs/SERVER.md) · [ik placement profiles](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/README.md)

| Path | Speculation / concurrency evidence | Prefix/cache/offload qualification |
|---|---|---|
| **vLLM baseline** | Local record: DSpark k5 and max sequences 6 | Simple CPU tier has completed restores; controlled correctness/latency qualification remains open. It does not enlarge the per-request configured window |
| **SGLang** | Exact-0731 DSpark and concurrent Hopper reports; original MTP and newer DSpark checkpoints must not be mixed | Radix/HiCache and multi-tier paths are documented; version/backend-dependent long-session cache misses/hangs are reported. [cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) · [long-session issue](https://github.com/sgl-project/sglang/issues/35129) |
| **LMDeploy** | Generic MTP support is not a verified V4 DSpark path; current V4 target loader skips `mtp.*` | Generic prefix-cache features are not automatically a V4 hybrid-state reuse qualification. [V4 loader](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/models/deepseek_v4.py) |
| **TensorRT-LLM** | V4 guide documents MTP for matching checkpoints; bundled 0731 DSpark parity not established | Special cache manager is documented. One active-history offload option is explicitly **prototype/not enableable for inference**; cold-page compression is a separate feature. [V4 guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md) · [config source](https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm_args.html) |
| **ExLlamaV3/TabbyAPI** | Dynamic batching documented; pinned no-spec 4-worker agent result; one tested speculative mode regressed quality | FP16 compressed-model cache already supports the cited 250K request; do not assume a further speculative/quantized-cache configuration is equally correct. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md) |
| **ik_llama.cpp** | Model/speculation/TP capabilities exist, but the cited large-window speed is a patched, non-equivalent profile | Host KV and streamed experts can increase capacity; the 512K profile turns prompt caching/checkpoints off. [profiles](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/README.md) |
| **llama.cpp** | Merged V4 MTP/DSpark work; 0731 requires the appropriate DSpark head | Broad placement/server capabilities; correct head and V4-specific cache state must be qualified. [merged PR](https://github.com/ggml-org/llama.cpp/pull/25784) |
| **FastLLM** | V4 guide documents built-in DSpark and two-device TP | Prefix cache and CPU/NUMA/disk expert placement documented; exact SM90 combination and bug fixes unverified here. [guide](https://github.com/ztxz16/fastllm/blob/master/docs/deepseek.md) |
| **DwarfStar/ds4** | Supported multi-GPU Flash layout has native grouped decode; **V4 session batching uses target-only decoding** | Disk prefix state exists; TP cache loading can rebuild prefixes rather than instantly restore all ranks. Single-card CUDA sessions use ordered fallback, not native batching. [server](https://github.com/antirez/ds4/blob/main/docs/SERVER.md) |
| **Arc** | Measured batch curve; speculative gain falls with batch in its own report | V4 batch measurement has paged attention/prefix caching off. TurboQuant has **no V4 head-dim-512 kernel**; expert parallelism is explicitly not implemented, and inherited generic multi-GPU claims are not a V4 execution receipt. [benchmarks](https://github.com/aeonmindai/arc/blob/master/docs/BENCHMARKS.md) · [limits](https://github.com/aeonmindai/arc/blob/master/docs/FLEET.md) |
| **vllm.cpp** | Same-GGUF GB10 decode test; no corresponding current H100 V4 service/speculation result | V4 test uses an example generator; production-server/model-registry reachability and cache/TP support must be checked separately. [benchmark](https://github.com/localai-org/vllm.cpp/blob/main/docs/benchmarks/dwarfstar-gguf.md) |
| **KTransformers** | Tutorial advertises EAGLE/MTP; exact current-0731 drafter handling remains unresolved. Long recipe captures batch one only | Weights offload, not just KV. Long-context recipe requires radix cache off; second concurrent request uses eager execution. [guide](https://ktransformers.net/en/docs/inference/long-context-deployment) · [tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md) |
| **Fucina** | Dedicated V4 MTP sidecar and parity fixtures; no H100 serving throughput result | Streamed expert cache; optional CUDA GEMM offload is not proof the full quantized-model pipeline runs efficiently on both GPUs. [runner](https://github.com/matteo-grella/fucina/blob/main/apps/deepseek4/README.md) |
| **Sonar/Aphrodite** | V4 source/recipe, vLLM-derived batching/speculation infrastructure | Features are plausible through its fork, but no independent speedup/cache-latency measurement against current upstream found. [recipe](https://github.com/dphnAI/sonar/blob/main/recipes/models/deepseek-ai/DeepSeek-V4-Flash.yaml) |
| **Chitu** | V4 configurations exist; matched generation/concurrency data missing | Native Flash marked pro; open FP8 variant has a different footprint. Existing performance table covers other models, not V4. [model matrix](https://github.com/thu-pacman/chitu/blob/public-main/docs/en/SUPPORTED_MODELS.md) · [performance table](https://github.com/thu-pacman/chitu/blob/public-main/docs/en/PERFORMANCE.md) |
| **LightLLM** | Open PR proposes MTP/DSpark, state-safe speculative writes and CUDA graphs | Proposed CPU/disk/PD cache and prefix checkpoint work, not a released throughput/correctness pass. [PR](https://github.com/ModelTC/LightLLM/pull/1593) |
| **RTP-LLM** | Hopper PR's numerical comparison explicitly uses **no speculation** | FP8-resident experts and PD topology change memory/communication substantially; not a direct two-card recipe. [PR](https://github.com/alibaba/rtp-llm/pull/1508) |
| **TensorFold** | Legacy Mac V4 DSpark/concurrency receipts; separate community V4.1 fork has 4-user results | Fork reports prefix reuse and disk-spilled cached prompts; current native release still has a DeepSeek serving gate. [legacy](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/deepseek-v4-flash.md) · [fork](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/README.md) · [current README](https://github.com/ashhart/TensorFold/blob/main/README.md) |
| **MLX-LM/exo** | No current V4-on-H100 speed/speculation qualification in this survey | Framework CUDA eligibility does not settle V4-specific kernels, distributed communication or cache state. [MLX](https://ml-explore.github.io/mlx/build/html/install.html) · [V4 PR](https://github.com/ml-explore/mlx-lm/pull/1797) |

**The TensorRT-LLM distinction is especially important.** Its source describes `enable_kv_cache_offload` for ratio-4 compressed history as a prototype that “currently cannot be enabled for inference.” Separately, NVIDIA documents cold-page quantization, including V4 CSA and preservation of SWA/HCA/compressor state. That separate feature is not proof the prototype is usable, nor a measured H100/0731 latency win. Treat exact cache method, hardware/build support and model-state correctness as separate checks. [configuration](https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm_args.html) · [cold-page compression](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html)

**The small-node SGLang result also explains why more parallelism can hurt.** In that report, adding DP attention without cache-aware session routing reduced prefix hits from 99.2% to 51.6% and increased hot TTFT to 1.873 s. A prefix cache partitioned across ranks needs suitable request affinity or cache transfer. This is configuration-specific evidence, not a prohibition on DP. [report](https://github.com/sgl-project/sglang/issues/37348)

For stability, source-level support and observed transfers are not the end of qualification. SGLang has open long-context DSPARK and cache reports, including an 8×H20 hang near 245K and a separate B200 HiCache stall; do not call both “H100 bugs.” A related old SM90 FlashInfer race was reported clean in newer tests but does not prove every downstream issue fixed. Conversely, your own Simple CPU restores still lack controlled restored-output/performance verification. Fair comparison applies the same correctness standard to the baseline and alternatives. [H20 issue](https://github.com/sgl-project/sglang/issues/33549) · [B200 issue](https://github.com/sgl-project/sglang/issues/32356) · [SM90 report](https://github.com/sgl-project/sglang/issues/33397)

### Q5. How good is each candidate for your workload?

The assessments below prioritize **long prompts, repeated agent turns and several concurrent users**, not the largest aggregate fleet number. They combine Q1's usable-context evidence, Q2's prefill/TTFT, Q3's generation and Q4's feature combinations. They are research priorities, not measured H100 performance rankings.

| Path | Assessment for your deployment |
|---|---|
| **vLLM baseline** | **Keep as the reference.** It is already deployed with the target model, 262K configuration and DSpark. Its current speed and CPU-cache benefit still need a controlled baseline; do not assume historical numbers describe 0.32 |
| **SGLang** | **Best first native-checkpoint challenger.** Most complete relevant evidence across Hopper, exact 0731, speculation and prefix/hierarchical caching. A potential single-user latency win is plausible; the closest reported comparison warns it can lose under concurrency and prefix reuse |
| **ExLlamaV3/TabbyAPI** | **Best converted-weight/context-efficiency experiment.** Pinned 250K and four-worker evidence is unusually useful. Low-bit weights, thinking mode and draft correctness are part of the choice; do not compare its scores or speeds as if it were unchanged native weights |
| **LMDeploy** | **Good compatibility candidate, unproven performance upgrade.** Packed-FP4 Hopper implementation is real, but current V4 DSpark parity and target-hardware benchmarks are missing. Its generic historical throughput claims do not answer this workload |
| **TensorRT-LLM** | **Serious engine, but no demonstrated small-node V4 advantage here.** V4 cache/precision/draft support must be pinned carefully. General large-cluster reputation and eight-GPU examples neither prove superiority nor disqualify two cards |
| **ik_llama.cpp** | **Strong placement/large-model toolkit option.** Real deep-context profiles exist, but cited speeds include host streaming and local patches; all-GPU two-H100 performance could differ substantially. Disabling prompt cache in its largest profile is an important agentic trade-off |
| **llama.cpp** | **Credible portable/GGUF option.** Actual 250K serving and merged DSpark support exist. A selected low-bit artifact was markedly slower on prefill than EXL3 in one matched-host report, but quantization differed, so that is not a universal engine result |
| **DwarfStar/ds4** | **One of the most interesting smaller engines.** Good model-focused documentation, context sweeps and real grouped CUDA serving. Trade-off: V4 session batching and DSpark do not combine in the documented path; exact H100 pair behavior remains open |
| **FastLLM** | **Worth watching/testing, not numerically ranked.** Direct V4/TP/DSpark support is attractive. Its published speed point is short-context SM120 TP8 with incomplete hardware/version metadata; no defensible H100 long-prompt projection |
| **Sonar/Aphrodite** | **Alternative fork, not a demonstrated speed upgrade.** Similar vLLM-derived infrastructure reduces novelty but does not establish equivalent behavior or extra performance. No convincing current-target head-to-head found |
| **Chitu** | **Conditional vendor/project candidate.** Model/edition support warrants interest, but the accessible benchmark table is other-model evidence. Ask for a V4-native, two-H100, long-prompt/concurrent demonstration before accepting performance claims |
| **Arc** | **Capacity/compression research, not a preferred interactive-serving candidate on published results.** Authors themselves identify serious overhead; measured 17.99 tok/s single-user and 0.53 at C256 are far from projected fleet floors. V4 TurboQuant/multi-GPU assumptions must not fill the gap |
| **vllm.cpp** | **Promising implementation experiment, not a proven production replacement.** Its same-GGUF GB10 improvement over ds4 is real reported paired evidence; model entrypoint, long context and H100 multi-GPU serving remain unqualified |
| **KTransformers/KT-Kernel** | **Useful when model capacity is the problem.** Current H100 weight fit makes its CPU-heavy reference path less compelling as a latency upgrade. Large-context guide uses a much larger CPU/memory environment, loses radix reuse and has very slow cold prefill; other placements may behave differently |
| **Fucina** | **Research/CPU-hybrid candidate today.** Clear implementation/parity work, but no convincing full-H100 serving benchmark or large-window concurrency demonstration in this search. Its CPU timings are not a CUDA speed ceiling |
| **TensorFold** | **Interesting watch-list/fork experiment, not your current drop-in.** Legacy V4 Mac results and the V4.1 dual-Spark fork show useful engineering, but neither validates released native V4-0731 on H100. The 2× fork result deserves attention without being transferred across model/hardware |
| **LightLLM** | **Development watch.** Proposed V4 feature coverage is broad; published cache/unit tests in an open PR are not a serving-performance release qualification |
| **RTP-LLM** | **Development watch with concrete Hopper optimization evidence.** Its H20 FP8/PD branch is meaningful work, but an internal ~8× TPOT improvement from a slow baseline does not make it 8× faster than vLLM or fit two H100s |
| **MLX-LM/exo** | **Portability/development watch.** Do not dismiss Linux CUDA, but no exact V4/H100/performance case was established. Current model branch and distributed kernels need evidence before comparison with production CUDA servers |

A few rankings change from a hardware-only survey. **LMDeploy and TensorRT-LLM remain legitimate, but lose priority relative to SGLang when current DSpark/prefix behavior matters.** **ExLlamaV3 rises** because it has an actual deep-request and concurrent-agent receipt, provided you accept quantized weights. Arc's low-bit fit, by contrast, does not make it a performance leader.

For memory-pressure or offline document-processing workloads, the priorities can reverse: a CPU/SSD hybrid that finishes a request otherwise impossible is useful even at low tokens/sec. That is a different objective from preserving fast interactive turns when the native model already fits. None of these conclusions requires assuming that missing data proves incompatibility.

## Comparison: scenario-oriented shortlist

| Your objective | First paths to investigate | Reason / main caveat |
|---|---|---|
| Preserve native checkpoint and improve C1 latency | **SGLang**, with vLLM as control | Relevant Hopper/DSpark evidence; no guaranteed concurrency win |
| Serve several agent users with growing reused prefixes | **Current vLLM vs SGLang** | Measure warm TTFT and per-user tails; cache affinity and hybrid-state correctness matter |
| Fit useful long windows with more GPU headroom | **ExLlamaV3/TabbyAPI**, then suitable GGUF paths | Actual 250K evidence, but converted weights and no same-target H100 comparison |
| Explore another native-weight kernel implementation | **LMDeploy**, **TensorRT-LLM** | Plausible kernels; first close exact-0731 draft/cache gaps and TP2 memory accounting |
| Specialized small-engine CUDA serving | **DwarfStar**, **FastLLM** | Real model paths; concurrency/speculation interactions and H100 behavior unverified |
| CPU/SSD capacity beyond GPU limits | **ik_llama.cpp**, **KTransformers**, **Fucina** | Placement flexibility; host quotas/bandwidth and cold-prefill latency can dominate |
| Follow an ambitious new engine/fork | **TensorFold V4.1 fork**, **vllm.cpp**, **LightLLM/RTP development** | Interesting reported progress, but separate branch/model/hardware and release gates |

The table intentionally does **not** declare one engine fastest. The best reported single-user competitor is not the same as the best concurrent/prefix-serving configuration in the nearest Hopper comparison. Likewise, a smaller weight artifact can create more context headroom while changing output quality and kernel behavior.

The most decision-useful distinction is therefore **native-model service optimization versus representation/placement changes**. Treat the latter as full recipe alternatives, not free engine swaps. That preserves the meaning of a comparison even when memory, quantization and context availability are the main reasons to consider them.

## Contradictions & Debates

1. **SGLang wins / vLLM wins.** Both appear in the same H800 report: SGLang+DSpark leads C1, while tuned vLLM leads C8/C32 and hot-prefix latency. **Resolution:** compare workload/latency objectives and versions; do not select the most favorable row as engine truth. [report](https://github.com/sgl-project/sglang/issues/37348)
2. **“1M context” versus real requests.** KTransformers configures 1M but its default example pool is ~965K; Arc configures 545K but measures tiny prompts; ExLlama's four-slot 1M pool is four 262K windows. **Resolution:** retain configured, admitted, filled and aggregate capacities separately. [KT](https://ktransformers.net/en/docs/inference/long-context-deployment) · [Arc](https://github.com/aeonmindai/arc/blob/master/docs/BENCHMARKS.md) · [EXL3](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md)
3. **Arc projected 4.1K tok/s/GPU versus measured 111.69 aggregate.** The former is a bandwidth-floor projection, not a serving run. The authors explicitly retract a previously claimed measured TurboQuant context multiplier. **Resolution:** use the measured serving curve; no V4 TurboQuant capacity benefit is established. [limits](https://github.com/aeonmindai/arc/blob/master/docs/FLEET.md)
4. **TensorFold release cannot serve DeepSeek versus a 2× dual-Spark report.** The latter uses a community fork of frozen Python 0.6.6 plus a dedicated V4.1 backend, not current native release support. **Resolution:** both can be true; identify model, fork and commit. [current upstream](https://github.com/ashhart/TensorFold/blob/main/README.md) · [fork provenance](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/README.md)
5. **TensorRT-LLM exposes offload, therefore the V4 option works.** The specific active-history flag's own source says it cannot currently be enabled for inference; cold-page compression is a different supported mechanism. **Resolution:** a config field is not a runtime pass. [source](https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm_args.html) · [separate compression](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html)
6. **Speculation nearly doubles ExLlama speed, therefore enable it.** The cited build/mode also changes outputs and loses on its small quality suite. **Resolution:** retain the no-spec result as the qualified recipe in that report; investigate exact draft correctness, not a general prohibition on speculative decoding. [evaluation](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md)
7. **Vendor says H100 must use FP8.** Lambda's example says native FP4 requires Blackwell, but newer SGLang docs provide Hopper W4A16/W4A8 paths and the local vLLM deployment already uses packed experts. **Resolution:** quote Lambda's benchmark for its FP8 recipe only, not its obsolete engine-wide limitation. [vendor](https://lambda.ai/inference-models/deepseek-ai/deepseek-v4-flash) · [current Hopper paths](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)
8. **Huge throughput numbers versus useful token delivery.** SGLang's combined input+output/GPU field, FastLLM's ambiguously named “total” rate, streamed draft bursts and Arc kernel floors measure different things. **Resolution:** preserve metric definitions and label every reciprocal/ratio as derived; no fabricated unified leaderboard.

## Uncertainties & Gaps

- **No current two-H100 A/B exists in this research.** No engine was installed or executed. The 95% confidence applies to evidence classification, not a fastest-engine claim or independent verification of every author's numbers.
- **Current vLLM 0.32 lacks a controlled speed baseline here.** Capacity/offload observations are dated 2026-10-18; the local prefill/completion numbers are older and version-specific. No fresh production health or memory probe was performed.
- **Most numeric results come from a single project/author.** Pinned hashes, raw-run links and honest methodology make them more useful, but not independent reproductions or broad reliability guarantees. Community reports retain that credibility tier.
- **Window evidence is heterogeneous.** Some rows show a configured limit or an admission estimate; others an actual request; others a shared pool. These are not interchangeable.
- **One filled needle does not establish comprehensive long-context quality.** Nor does a small coding suite or GSM8K result prove identical quality under low-bit weights, another reasoning mode, or speculative decoding.
- **The H800 comparison is close, not exact.** Different GPU count, interconnect, versions and tuning; baseline speculation/timing details are incomplete. It supports workload sensitivity, not a numerical prediction for your pair.
- **SM90 tuning advice can carry bad memory arithmetic.** One comment in that thread claims FP8 V4 weights total 75 GB across four GPUs. That is not accepted as a fit fact or used to authorize conversion; actual converted tensor residency must be measured.
- **Host quotas constrain hybrid portability.** A 256GB/large-many-core reference host is not your 192GiB-limited, 16-CPU container; the 100GB offload budget is not a memory-availability measurement. A host recommendation is also not a measured process RSS.
- **Current-model draft support matters.** Original V4 MTP, bundled 0731 DSpark and a tutorial generically saying “MTP” cannot be assumed equivalent. Some inspected paths skip or do not document the current draft head.
- **Cache features need combination-specific verification.** Unknown means unknown, not unsupported. Published generic prefix caching/CPU offload may not include V4's request-private SWA/compressor state or speculation.
- **Evidence is bounded and dated.** The 20-query search found useful reports but cannot establish that no additional benchmark exists anywhere. Main-branch docs and open PRs may change after 2026-10-20.

## Recommendations

### Primary recommendation

**Retain vLLM as the reference and make SGLang the first candidate for a future controlled comparison.** It is the strongest native-model alternative for the exact feature mix, but the available Hopper evidence warns against assuming a concurrency or cache-latency improvement. Establish a current vLLM 0.32 baseline before assigning an alternative a speedup.

For a meaningful future comparison, keep the checkpoint/representation, reasoning mode, output cap, speculative mode, per-request context, actual prompt depth, cache state and concurrency explicit. Record cold and repeated-prefix TTFT p50/p95, per-user generation/TPOT, aggregate output throughput, queueing, preemptions, memory peaks and output correctness. This is an evidence requirement, **not a benchmark executed or deployment authorized by this report**.

### Alternatives

- **ExLlamaV3/TabbyAPI** if you accept a converted low-bit checkpoint: strongest practical large-window/concurrent-agent receipt found. Validate the intended quality/reasoning mode and drafter, rather than transferring another quant's test score.
- **LMDeploy / TensorRT-LLM** if investigating alternative native-weight kernels is worthwhile: first pin current-0731 draft/cache support and two-rank memory behavior. Neither is currently a documented speed win on your appliance.
- **DwarfStar / FastLLM / ik_llama.cpp** for further specialized-engine investigation. They have real implementations and useful measurements, but materially different speculation/batching/placement conditions.
- **TensorFold's community V4.1 fork** as a separate watch item. Its reported improvements are interesting enough to track; do not replace the current V4/H100 evidence gap with a different model's Spark result.

### Not recommended as a performance decision today

Do not switch based on “1M”, a generic engine-wide multiplier, an 8-GPU aggregate, or an unmeasured bandwidth floor. Do not prefer CPU/SSD-heavy reference configurations purely for speed while your native model already fits; they may be valuable for capacity, but the documented cold-prefill and prefix-cache trade-offs are substantial. Do not buy a commercial/pro variant or allocate additional hardware on the strength of the missing V4 measurements. No infrastructure allocation or production change is proposed here.

## Methodology

- **Depth/type:** exhaustive comparative research; 20 web queries across seven search/retrieval rounds, followed by direct docs/source/issue inspection.
- **Scope:** 19 comparison rows: the current vLLM reference plus 18 alternative paths. TensorFold's upstream legacy/native and community-fork evidence are explicitly separated within one path; MLX-LM/exo is a backend-related group, not a claim that they are identical engines.
- **Source hierarchy:** exact model/build/hardware records first; other-Hopper reports second; different-hardware same-model results as orientation; different-model and kernel/theoretical claims only when clearly labelled. No website summary was promoted over a contradicting primary document.
- **Evidence extraction:** metric definitions, hardware count/architecture, model/quant, context setting versus actual prompt, output length, concurrency, cache state, speculation, build/date and limitations. Missing fields remain null in the companion JSON.
- **Derived quantities:** reciprocal TPOT, simple same-report ratios and output-tokens/full-duration rates are arithmetic, not newly measured throughput. Combined input+output is never labelled prefill or decode.
- **Cross-referencing:** Q1's context and placement findings constrain Q2/Q3 timing interpretation; Q4 checks whether desired speed features coexist; Q5 ranks investigation value rather than inventing an unsupported performance winner.
- **Confidence:** research checkpoint approved report synthesis at **95% documentary-classification confidence** (subquestions 96/96/96/95/93). Confidence in precise cross-hardware speed predictions is deliberately not supplied.
- **Breadth:** direct source retrieval exceeded the usual small source-planning band because every shortlisted engine required model-specific and often mutually conditional context/cache/speculation checks. The 20-query discovery budget was retained.
- **Local work only:** research Markdown and a sanitized structured ledger; no model download, dependency installation, GPU test, inference load, production edit, commit or push. Prior compatibility research and deployment documents are unchanged.

## Sources

Inspection date is 2026-10-20. Publication/test dates are stated in the analysis or ledger when reported; a mutable main-branch page is not assigned an invented publication date. Primary project/vendor records are first-party claims; community results are attributed measurements, not independent reproduction by this research.

| # | Source | URL | Date/provenance | Credibility |
|---:|---|---|---|---|
| 1 | Hopper comparison | https://github.com/sgl-project/sglang/issues/37348 | 2026-09-01; checked 2026-10-20 | Tier 3 community report / primary state |
| 2 | ExLlamaV3 evaluation | https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md | 2026-08-09; checked 2026-10-20 | Tier 3 community report / primary state |
| 3 | LMDeploy V4 | https://github.com/InternLM/lmdeploy/pull/4554 | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 4 | TensorRT-LLM V4 | https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 5 | DwarfStar | https://github.com/antirez/ds4/blob/main/docs/PERFORMANCE.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 6 | FastLLM | https://github.com/ztxz16/fastllm/blob/master/docs/benchmarks/deepseek_v4.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 7 | ik toolkit | https://github.com/daimonionnn/ik-llama-toolkit/blob/main/README.md | 2026-09-04; checked 2026-10-20 | Tier 3 community report / primary state |
| 8 | Arc measurements | https://github.com/aeonmindai/arc/blob/master/docs/FLEET.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 9 | KTransformers long context | https://ktransformers.net/en/docs/inference/long-context-deployment | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 10 | TensorFold community fork | https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/README.md | 2026-10-17, 2026-10-18; checked 2026-10-20 | Tier 3 community report / primary state |
| 11 | DwarfStar serving | https://github.com/antirez/ds4/blob/main/docs/SERVER.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 12 | LMDeploy source | https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/models/deepseek_v4.py | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 13 | Arc | https://github.com/aeonmindai/arc/blob/master/docs/BENCHMARKS.md | 2026-08-17; checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 14 | SGLang data definition | https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 15 | docs | https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 16 | test config | https://github.com/InternLM/lmdeploy/blob/main/autotest/configs/deepseek-ai/DeepSeek-V4-Flash-0731.yml | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 17 | guide | https://github.com/ztxz16/fastllm/blob/master/docs/deepseek.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 18 | CUDA | https://github.com/antirez/ds4/blob/main/docs/CUDA_MULTI_GPU.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 19 | benchmarks | https://github.com/localai-org/vllm.cpp/blob/main/docs/benchmarks/dwarfstar-gguf.md | 2026-08-04, 2026-08-05; checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 20 | runner | https://github.com/matteo-grella/fucina/blob/main/apps/deepseek4/README.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 21 | recipe | https://github.com/dphnAI/sonar/blob/main/recipes/models/deepseek-ai/DeepSeek-V4-Flash.yaml | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 22 | matrix | https://github.com/thu-pacman/chitu/blob/public-main/docs/en/SUPPORTED_MODELS.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 23 | PR | https://github.com/ModelTC/LightLLM/pull/1593 | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 24 | PR | https://github.com/alibaba/rtp-llm/pull/1508 | 2026-10-13; checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 25 | legacy | https://github.com/ashhart/TensorFold/blob/main/docs/recipes/deepseek-v4-flash.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 26 | MLX installation | https://ml-explore.github.io/mlx/build/html/install.html | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 27 | V4 PR | https://github.com/ml-explore/mlx-lm/pull/1797 | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 28 | vendor | https://lambda.ai/inference-models/deepseek-ai/deepseek-v4-flash | Checked 2026-10-20 | Tier 1 vendor self-report |
| 29 | tutorial | https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 30 | long-session issue | https://github.com/sgl-project/sglang/issues/35129 | Checked 2026-10-20 | Tier 3 community report / primary state |
| 31 | config source | https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm_args.html | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 32 | merged PR | https://github.com/ggml-org/llama.cpp/pull/25784 | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 33 | performance table | https://github.com/thu-pacman/chitu/blob/public-main/docs/en/PERFORMANCE.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 34 | current README | https://github.com/ashhart/TensorFold/blob/main/README.md | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 35 | cold-page compression | https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html | Checked 2026-10-20 | Tier 1 project documentation/code/self-report |
| 36 | H20 issue | https://github.com/sgl-project/sglang/issues/33549 | Checked 2026-10-20 | Tier 3 community report / primary state |
| 37 | B200 issue | https://github.com/sgl-project/sglang/issues/32356 | Checked 2026-10-20 | Tier 3 community report / primary state |
| 38 | SM90 report | https://github.com/sgl-project/sglang/issues/33397 | Checked 2026-10-20 | Tier 3 community report / primary state |
