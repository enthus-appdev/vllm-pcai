# Open vLLM PRs for PCAI: DeepSeek-V4-Flash-0731 on TP2 H100 NVL

**Date:** 2026-10-23 (UTC snapshot)

**Depth:** deep, approved by requesting session

**Confidence:** 88% in applicability/ranking, **not** in predicted deployment speedups

**Evidence:** 44 primary-source items inspected across four research rounds; no serving benchmarks run here.

**Scope:** This is a dated research snapshot. PR states, rankings and patch-stack checks describe the inspection point, not the current image or a deployment qualification. See the [README vendored-patch inventory](README.md#why-this-image-exists) for the current carried fixes. Public upstream source URLs are retained; no deployment addresses or credentials are included.

## Executive Summary

**#58074 is the strongest capacity candidate, but not yet a low-risk production carry.** It replaces oversized CUDA C128 compressor history with bounded circular state and reports +42.6% KV capacity elsewhere; its complete diff applies to both v0.32.0 and our patched source. The next priorities are #60078 (our actual SimpleCPUOffload load path, but an outstanding reviewer identifies lost copy/compute overlap), #55654 (V4-specific DSpark prefill work reduction, requiring native compilation), #61461 (small Hopper decode optimization), and #53592 (real V4 HiSparse support, currently unsuitable to carry).

There is **no verified ready-to-carry V4 KV-sharding solution compatible with our exact DSpark configuration**. #59916 explicitly rejects speculative decoding; #44573 is a broad, stale native-DCP port. Do not bump just for #60692 or #61023: their relevant implementation is already in v0.32.0. A newer base becomes more interesting for merged #54163, #63251, #63165 and Humming dependency updates, with important qualifications below.

## Key Findings

1. **Capacity without changing checkpoint quantization:** #58074 changes compressor-state allocation, not expert weights. It explicitly reports a SimpleCPUOffload evaluation, making it much closer to our setup than generic MLA cache work. Its +42.6% is an upstream claim, not our expected gain. [PR](https://github.com/vllm-project/vllm/pull/58074)
2. **HiSparse support exists as open work, not a usable toggle today:** #53592 handles V4 C4 geometry while other groups stay resident. Author reports saturation regressions/allocator failure; independent comments report startup failures. [PR](https://github.com/vllm-project/vllm/pull/53592)
3. **DSpark prefill has a genuinely V4-specific optimization:** #55654 removes discarded Q projection, dummy Q insertion and redundant KV normalization. H20/V4-0731 evidence is unusually relevant, but changes native CUDA symbols and cannot be installed as Python-only edits over the current image. [PR](https://github.com/vllm-project/vllm/pull/55654)
4. **Generic CPU offload is not SimpleCPUOffload:** #61961, #47891 and #54743 must not be sold as immediate fixes for our connector. [Sync-load PR](https://github.com/vllm-project/vllm/pull/61961), [lookup PR](https://github.com/vllm-project/vllm/pull/47891), [group-scoping PR](https://github.com/vllm-project/vllm/pull/54743)
5. **Hopper MoE throughput wins can be a single-user regression:** #61875 reports high-concurrency gains but low-concurrency regressions, and needs newer Humming prerequisites. [PR](https://github.com/vllm-project/vllm/pull/61875)
6. **Long-context upgrade safety matters more than headline throughput:** merged #63251 addresses V2 drafter rotary-cache overruns, but can cap target context for non-extendable drafters. Verify resolved draft configuration before adopting it. [PR](https://github.com/vllm-project/vllm/pull/63251)

## Ranked Comparison: OPEN PRs

All rows below were OPEN at inspection. Dates are latest `updatedAt` UTC dates, not authorship dates. `PASS/PASS` means full upstream diff passes `git apply --check` against clean v0.32.0 / v0.32.0 plus all four existing patches. `FAIL/FAIL` means at least one hunk fails in each. It does not mean the change is impossible to backport. `NT` means not checked because not a carry candidate. Sizes include tests. “Carry now” means a narrow candidate for the next normal validated image change, **not** a claim it is deployment-qualified.

| Rank | PR / title | Update; size | Change and exact-config applicability | Benefit and principal risk | Apply clean/patched | Verdict |
|---:|---|---|---|---|---|---|
| 1 | [#58074 Use circular buffer for DeepSeek V4 C128 state](https://github.com/vllm-project/vllm/pull/58074) | Oct 23; +841/-112 | CUDA V4 C128 state becomes one bounded ring per request; includes speculative lookahead. TP2, V2 and fp8 V4 are relevant; C4 unchanged. | Author: 335,514→478,421 KV tokens; SimpleCPUOffload GSM8K 0.955. Actual gain at max-seqs=6 and block=256 unknown. CuTe/Triton state semantics are substantial correctness risk; no approving review. | **PASS/PASS** | **Watch — first capacity candidate to qualify** |
| 2 | [#60078 Order SimpleCPUOffload loads after pending compute reads](https://github.com/vllm-project/vllm/pull/60078) | Oct 22; +103/-8 | Exact connector; adds load-side compute event to protect reused GPU blocks. Hardware-independent mechanism applies to sm_90/DSpark. | Correctness, not speed. Independent worker/DMA reproductions support hazard; reviewer says event is recorded too late, serializing with current forward. Outstanding change request; failing checks. | **PASS/PASS** | **Watch — wait for corrected barrier placement** |
| 3 | [#55654 DSpark KV-only context insert and fused kv_norm](https://github.com/vllm-project/vllm/pull/55654) | Oct 12; +1073/-23 | Specifically **V4, not V4.1**; MXFP4/FP8 checkpoint unchanged. One stacked KV-only GEMM plus fused normalization/RoPE/insert on packed fp8 cache. | Author H20 TP4 V4-0731: context write kernel 4.7–4.9x, engine write ~4.2x. These are sub-operations, not TTFT or serving throughput. Native CUDA build needed; stable-libtorch source paths absent in release. | **FAIL/FAIL** | **Watch — strongest prefill-oriented kernel candidate** |
| 4 | [#61461 Tune fused mHC post + pre-norm for SM90 at up to 8 tokens](https://github.com/vllm-project/vllm/pull/61461) | Oct 21; +58/-16; draft | V4 hidden=4096; DSpark k=5/C1 means six target rows, in tuned band. C2 can be outside band. No weight requantization. | H20 V4-shape microbenchmark: six rows 10.85→10.24 µs; V4.1 E2E ~0.9% output gain. V4 tested only at op level; summation order changes; draft, no approval. | **PASS/PASS** | **Watch — modest low-concurrency decode experiment** |
| 5 | [#53592 HiSparse: add DeepSeek V4 support](https://github.com/vllm-project/vllm/pull/53592) | Oct 21; +705/-82 | Actual V4 C4 hot/host layout, preserves split FP8 values/scales; non-C4 stays on GPU. Separate HiSparse operating mode, not proof it composes with our SimpleCPUOffload tier. | Potential long-context capacity breakthrough. Author saturation regression/allocator failure, invalid prior instrumentation; startup failures reported on H20 and another architecture. Native code and layout changes. | **FAIL/FAIL** | **Watch — not production-ready** |
| 6 | [#59916 --dcp-gather: gather-based DCP (+ DSv4)](https://github.com/vllm-project/vllm/pull/59916) | Oct 12; +483/-2; draft | V2 yes, V4 TP2/DCP2 evaluated on B300; **explicitly rejects any speculative_config** and downgrades full graphs to PIECEWISE. | Can shard persistent KV; needs per-layer scratch/gathers. Requires abandoning DSpark; Hopper/NVLink and 1M cost unverified. | **FAIL/FAIL** | **Watch — alternative no-DSpark branch, not exact-config carry** |
| 7 | [#58291 Attach V4 request tools to existing system/developer messages](https://github.com/vllm-project/vllm/pull/58291) | Sep 23; +64/-8 | Python V4 tokenizer; fixes developer-first tool prompt placement. Applies beside our #46257 patch; no runtime/kernel change. | Correct reference ordering when a developer message carries instructions. Upstream author reports 45 tests passed; no human approval and failing/cancelled checks. No benefit if clients always use system-first. | **PASS/PASS** | **Carry now, conditional on developer-role traffic and tokenizer tests** |
| 8 | [#61824 MRV2 DSpark local argmax reduction](https://github.com/vllm-project/vllm/pull/61824) | Oct 19; +172/-0; draft | Greedy DSpark TP2 is a direct conceptual match: exchange per-rank score/token winners instead of full vocabulary. Touches V4 drafter and V2 speculator. | Less draft communication; does not inherently improve acceptance. **No benchmark validation yet**; tie-breaking, sharded Markov bias and padding need tests. | **PASS/PASS** | **Watch** |
| 9 | [#58780 Stream V4/V3.2 string tool arguments incrementally](https://github.com/vllm-project/vllm/pull/58780) | Oct 16; +243/-1 | V4 parser; long code/command arguments can stream before parameter close. Separate from existing encoder patches. | Better incremental UI/tool payload delivery, not earlier execution of an incomplete tool. Reviewer flags quadratic regex; failing CI and no approval. | **PASS/PASS** | **Watch** |
| 10 | [#52996 Separate DSpark scheduler input budgets](https://github.com/vllm-project/vllm/pull/52996) | Sep 28; +50/-3 | Separate target prefill and fixed draft-query budgets; V2 DSpark relevant. | Could improve chunk budget utilization, but with 4096 tokens and 1–2 requests fixed query overhead is small. CPU regression evidence only; scheduler backport conflicts. | **FAIL/FAIL** | **Watch, low priority** |
| 11 | [#44573 Add DeepSeek-V4 DCP decode support](https://github.com/vllm-project/vllm/pull/44573) | Aug 19; +1583/-166 | Native V4 sparse decode/compressor/index mapping and merge changes; TP2/DCP2 reported. | KV sharding potential, but broad/stale source port; reported tests use different MoE/EP configuration, no proof of our V2+DSpark+offload combination. | **FAIL/FAIL** | **Watch, not a production patch stack** |
| 12 | [#47891 Exclude DSpark draft group from OffloadingConnector lookup poisoning](https://github.com/vllm-project/vllm/pull/47891) | Oct 13; +569/-8 | Generic OffloadingConnector plus core hit reconciliation, **not our SimpleCPU implementation**. Modifies same config file as #48023 but already fails clean tag too. | Relevant symptom if switching connectors; author ~82.9% hit-rate result is not transferable. Core changes can affect unrelated methods. | **FAIL/FAIL** | **Watch only for connector migration investigation** |
| 13 | [#61875 Declare MXFP8 activations; run block FP8 on Hopper](https://github.com/vllm-project/vllm/pull/61875) | Oct 23; +44/-2 | V4 MXFP4 experts + Humming SM90; needs #62751 and Humming 0.1.18. Changes activation execution format, not checkpoint weights. | Author TP4 high-concurrency gains; at c4 TP4 **-2%**, EP **-6.5%** output throughput. Exact c1/TP2 unmeasured. Needs explicit owner acceptance if no-requant policy also forbids runtime activation-format changes. | **FAIL/FAIL** | **Skip for current single-user workload; watch upstream** |
| 14 | [#62958 DSpark int8-W2 fused Markov walk on Hopper](https://github.com/vllm-project/vllm/pull/62958) | Oct 22; +1747/-1 | SM90 and V4 vocab=129280/k=5 shape included in kernel measurements. Introduces opt-in **int8 quantization of W2**. | H100 microbenchmark 308→40 µs for that shape; E2E measured on Qwen, not V4. Native extension and draft acceptance changes. Violates literal no-requantization requirement. | **FAIL/FAIL** | **Skip under current policy** |
| 15 | [#58016 Skip unused checkpoint tensors before reading](https://github.com/vllm-project/vllm/pull/58016) | Oct 7; +100/-4 | DSpark V4 default safetensors loader; no proof of RunAI streamer path benefit. Our #48023 exists specifically for streamer/shared-checkpoint handling. | Author V100/Turing-fork boot 10:25→6:03; does not establish our startup gain. Loader and DSpark test conflicts. | **FAIL/FAIL** | **Watch; confirm actual load format first** |
| 16 | [#56184 Handle empty/stray V4 DSML fragments](https://github.com/vllm-project/vllm/pull/56184) | Sep 20; +502/-0 | V4 parser and delegating reasoning/tool pass; affects malformed generation recovery. | Reduces leaks/content loss; reviewer questions preserving content after empty tool wrapper versus reference behavior. Broad semantic change, overlaps #57051. | **PASS/PASS** | **Watch; do not stack competing recovery policies** |
| 17 | [#57051 Drop orphan V4 DSML structural closers](https://github.com/vllm-project/vllm/pull/57051) | Sep 15; +117/-1 | Narrower parser leakage fix with skip-tool-parsing pass guard. | Useful if orphan closers persist after our encoder fix; full diff conflict is V4.1 test hunk, not proof V4 runtime is incompatible. | **FAIL/FAIL** | **Watch** |
| 18 | [#53984 Skip unregistered V4 DSpark expert weights](https://github.com/vllm-project/vllm/pull/53984) | Oct 23; +49/-0 | Loader guard; independent H20 V4-0731+DSpark startup validation reported. | Prevents a particular missing-scale KeyError. Already-serving image has no evidence of this failure; no steady-state performance benefit. | **PASS/PASS** | **Skip unless reproducing loader error** |
| 19 | [#61961 Load small CPU-tier hits synchronously](https://github.com/vllm-project/vllm/pull/61961) | Oct 23; +336/-30 | **OffloadingConnector**, not SimpleCPUOffload; opt-in byte budget. | Avoids scheduler-step delay for short hits. Earlier token-budget revision benchmarked; latest byte-budget/separate-stream revision not E2E validated. | NT | **Skip for exact connector; watch as alternative** |
| 20 | [#59265 Preserve packed layouts for layer-wise KV cache](https://github.com/vllm-project/vllm/pull/59265) | Oct 22; +1218/-46; draft | Stacked recipe/layout series, three prerequisites bundled. | Not a compact CPU-slot or V4 offload-hit certification; five-line incremental feature masks a large stack. | NT | **Skip current image** |
| 21 | [#61277 HiSparse FlashInfer SM90 sparse MLA backend](https://github.com/vllm-project/vllm/pull/61277) | Oct 21; +434/-42; draft | Generic NoPE sparse backend, depends on #61243; **not V4's custom FlashMLA attention**. | H200/GLM accuracy repair does not enable V4. | NT | **Skip as a V4 fix** |
| 22 | [#58540 V4 Hopper FP8 FlashInfer moe_ep mega backend](https://github.com/vllm-project/vllm/pull/58540) | Oct 22; +319/-49; draft | Hardware title matches, but empty purpose/results and no verification of official mixed MXFP4 checkpoint path. | No usable speed/accuracy evidence; cannot infer benefit from sm_90 label. | NT | **Skip pending evidence** |
| 23 | [#60320 Compute windowed draft context for trailing rows only](https://github.com/vllm-project/vllm/pull/60320) | Oct 16; +71/-1 | Explicit gate requires **no KV connector**; our SimpleCPUOffload excludes it. | Author V4.1 TTFT gains ≤1.3%, confidence intervals include zero; not an exact-config optimization. | NT | **Skip** |

### What the comparison means

The major distinction is between reducing **persistent state allocation** (#58074), moving selected KV to a host hot-buffer system (#53592), and sharding KV across devices (#59916/#44573). These are not interchangeable capacity multipliers. #58074 is the only leading capacity option whose full diff presently applies without a port and whose body explicitly includes SimpleCPUOffload validation; it still changes state interpretation and deserves exact long-context correctness qualification.

For single-user latency, smaller kernels are not automatically better priorities than avoiding re-prefill. #55654 attacks a real V4 draft-context sub-operation, whereas #61824 cuts greedy draft communication and #61461 trims a small mHC operation. None provides an evidence-backed forecast for our measured 9.44-second cold TTFT. Likewise, #61875 is a clear example where high-throughput tuning can make low-concurrency use worse.

## Detailed Analysis

### 1. KV capacity: first qualify bounded state, do not blindly enable DCP/HiSparse

The official V4-0731 config specifies one KV head, 64 attention heads, head dimension 512, hidden size 4096 and max positions 1,048,576. This confirms the importance of avoiding replicated persistent allocations, but does not establish that every cache component will halve under DCP. [Official config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json)

The v0.32.0 source rejects V4 in `vllm/v1/hisparse/layout.py`; V4's custom sparse backend subclasses `AttentionBackend` directly, not the generic V3.2 `FlashMLASparseBackend`. Its `get_impl_cls()` is intentionally unimplemented because attention runs through `DeepseekV4Attention`. Therefore generic sparse-MLA/SM90 fixes are not sufficient evidence of V4 applicability. This also explains why #59319's generic mixed-batch workspace savings and #63104's FA4 grouped prefill should not be carried as V4 optimizations. [Release source](https://github.com/vllm-project/vllm/tree/v0.32.0), [generic workspace PR](https://github.com/vllm-project/vllm/pull/59319), [FA4 PR](https://github.com/vllm-project/vllm/pull/63104)

#58074 reports capacity relief by replacing C128 history's paged state with a bounded ring, not by eliminating repeated history copies. An upstream commenter made the latter claim and the author corrected it. At our max-seqs=6, the percentage gain may differ substantially from the reported benchmark. This connects directly to offload compatibility: a bounded non-prefix-cacheable state group must not poison hit reconciliation or be blindly exported as normal attention KV.

### 2. Hopper prefill and MXFP4: distinguish model path and workload

#55654 is unusually well targeted: V4-0731, Hopper-class H20 and packed uint8 fp8-DS-MLA cache. It retains official FP8 projections/MXFP4 experts and removes unused draft Q work. However, it adds native functions in a stable-libtorch tree absent from v0.32.0. Carrying only its Python file would reference missing operators; a proper native backport or newer matching build is needed. Its 4.2x engine *context-write* number is not a full prefill speedup.

#61461 is smaller and applies cleanly, but accelerates decode-sized mHC rather than 4096-token prefill chunks. #61875 targets MoE activation format selection and has a documented low-concurrency regression; our workload argues against carrying it just to chase c128 benchmarks. No verified V4/H100/TP2/1M end-to-end prefill win was found that is both ready and mechanically installable into the current image.

### 3. DSpark acceptance and CPU offload: correctness before draft count tuning

Local argmax (#61824) is a natural fit for greedy TP2 but lacks performance validation. It should preserve selection semantics rather than increase acceptance. #52996 fixes budget accounting, not the poor late-position acceptance curve. The int8 fused Markov walk (#62958) is specifically disallowed by the literal no-requant requirement, even though target verification protects final sampling semantics.

The existing connector makes #60078 more relevant than generic OffloadingConnector hit-path PRs. Its full review identifies the key trade-off: the proposed event waits on the current forward instead of only earlier reads, potentially sacrificing overlap. Independent DMA tests confirm a race and the barrier mechanism, but do not settle the optimal production event placement. This is a **watch** correctness fix, not a TTFT optimization to ship unreviewed.

### 4. Parser and startup: narrow fixes are easier, but traffic/loader must match

#58291 is the narrowest conditional carry: developer-first messages get the reference instructions-then-tools layout, and it coexists mechanically with #46257. Our existing #50686 already addresses split assistant messages; malformed-output recovery PRs are complementary, not proof the encoder patch is obsolete. #56184 and #57051 overlap and should not be stacked without an explicit policy decision on content following empty/malformed tool blocks.

#58780 improves argument streaming granularity but carries a reviewer-identified regex-cost issue. #58016 only filters tensors before reading in the default safetensors loader. Its boot-time evidence does not establish RunAI streaming support; do not extrapolate that result to our startup path merely because both use DSpark.

### 5. Merged-to-main since v0.32.0: bump reasons and false positives

| PR | Merged UTC | What a newer base adds / exact-config significance | Assessment |
|---|---|---|---|
| [#63251 Keep drafters within rotary cache on V2](https://github.com/vllm-project/vllm/pull/63251) | Oct 23 17:06 | Extends cache-only RoPE configurations where safe; otherwise caps V2 target context at draft limit. Relevant safety class after moving to 1M. | **Strong reason to inspect a newer base**, but confirm resolved DSpark config. Official target already has 1M and YaRN; this is not proof our current draft overruns. A bump could unexpectedly reduce advertised context if draft limit is lower. |
| [#54163 DFlash/DSpark no longer drop last prefix-cache block](https://github.com/vllm-project/vllm/pull/54163) | Oct 23 00:03 | Narrows EAGLE trailing-block policy; DSpark no longer unnecessarily drops a reusable block. | **Relevant prefix-cache correctness/efficiency reason.** Largest reports are hybrid/Mamba cases, not V4; do not claim equivalent full-reprefill elimination here. Full diff does not cleanly apply to patched release. |
| [#63165 SHM readers poll notify socket while busy](https://github.com/vllm-project/vllm/pull/63165) | Oct 23 04:58 | Bounds ZMQ notify-command backlog under sustained load; TP2 and KV-connector response readers relevant. | **Useful operational fix**, not prefill speed. Full diff passes on patched release; distinct file from our queue-size patch. Could be carried separately rather than justify a whole bump alone. |
| [#62816 Humming 0.1.18](https://github.com/vllm-project/vllm/pull/62816) | Oct 22 19:41 | Upgrades release's 0.1.17 and removes DeviceInfo workaround; prerequisite for current Hopper work. | **Dependency-coherent bump is preferable to Python-only cherry-picks.** Does not prove TP2 V4 speedup. |
| [#62751 Honor Humming MXFP4 activation override](https://github.com/vllm-project/vllm/pull/62751) | Oct 23 15:09 | Shared input-schema helper and explicit override correctness; prerequisite for #61875. | Useful infrastructure, **not** an automatic win under current defaults; avoid activation changes without policy approval. |
| [#61615 Small-head FlashMLA prefill on SM90](https://github.com/vllm-project/vllm/pull/61615) | Oct 22 07:50 | New FlashMLA pin and actual TP-local heads for **V4.1**. Changed model file is V4.1-only. | **Not a V4-0731 bump justification.** |
| [#62875 Rust DeepSeek DSML markers by token identity](https://github.com/vllm-project/vllm/pull/62875) | Oct 22 04:31 | Rust parser distinguishes dedicated DSML token from ordinary text. | **Not current Python parser fix**; only relevant with Rust frontend adoption. |

**Already in v0.32.0, not bump reasons:**

- [#60692 DSpark/DFlash metadata-build skipping](https://github.com/vllm-project/vllm/pull/60692), merged Oct 12. Compared release files with the merge commit: draft speculator, base speculator, draft cudagraph, target-dependent speculator, sparse-SWA and V4.1 FlashInfer files are identical. The remaining generic Triton file differs due to other changes. The PR's own greedy TP4/GB200 table shows essentially unchanged throughput, not a hidden large latency win.
- [#61023 Sparse-MLA prefill workspace factor](https://github.com/vllm-project/vllm/pull/61023), merged Oct 12. **All four production files identical** between release and merge commit; default factor resolves to two above 256K. V4 reads the shared indexer sizing helper, but its own FlashMLA gather workspace is separately implemented. Do not claim generic GLM's reclaimed GiB for V4.
- [#58857 Stream V4/V4.1 VL language weights](https://github.com/vllm-project/vllm/pull/58857), merged before release timestamp, concerns VL wrapper loading; not established for our text-only V4 path.

## Patch Applicability and Local-Patch Conflicts

Checks ran **only** in `/tmp/pcai-pr-research/vllm-check`, a scratch shared clone. The checked tag resolves to `5df63a7203cdb9328d824a27b682e77c7b9928be` (commit timestamp Oct 17 18:57:10 UTC); inspected main snapshot was `224b2f54f19d44bfb96edfe16c6343c7495853d3`. Source checkout is shallow: a failed merge-base ancestry check is **not evidence of feature absence**. Content comparisons above override that invalid inference.

Applied existing patches in Dockerfile order in scratch: `46257-deepseek-generation-controls-v0.30.patch`, `48023-spec-draft-inherit-model-weights.patch`, `mq-worker-response-honour-chunk-bytes.patch`, `50686-merge-consecutive-assistant-messages.patch`. **All four apply successfully.** Each candidate was checked independently, not stacked with other candidate PRs.

- All 20 initially checked upstream diffs had the **same pass/fail result before and after our patches**. No candidate was newly broken solely by a local patch.
- #58291 shares `deepseek_v4.py` with #46257 but passes with the actual patch applied.
- #47891/#62958 share `speculative.py` with #48023, yet already fail on clean release. Those failures cannot be blamed solely on our patch.
- #58074/#60078/#61461/#61824 do not touch our four patched files.
- #55654/#53592 fail in part because their native `csrc/libtorch_stable/*` files are absent in release. These need more than a context-only patch rebase.
- #57051 fails a V4.1 test hunk; #59916 fails V2 attention/model-runner hunks. Do not equate full-diff rejection with a proven semantic incompatibility.
- #56184/#57051/#58780 alter common parser surfaces; pairwise candidate stacking and behavior were not tested.
- Additional merged checks on **patched** tag: #63165 PASS; #54163 FAIL (`speculative.py`); #63251 FAIL (docs and `speculative.py`).

Scratch evidence: numbered PR metadata JSON and diffs plus `apply-results.json` under `/tmp/pcai-pr-research/`. These are temporary inspection artifacts, not a committed patch proposal. Report contains no deployment identifiers, credentials or network addresses.

## Contradictions & Debates

- **“HiSparse PR is not draft” vs author warns it remains draft:** GitHub API currently reports non-draft; body explicitly withholds readiness for performance/allocator reasons. The operational evidence, not the UI flag, determines the watch verdict.
- **“Circular state speeds history copying” vs actual allocation mechanism:** author correction says prior blocks were preallocated already; assess #58074 as capacity relief, not a claimed copy-elimination speedup.
- **“A load barrier is correct” vs “this event placement is too late”:** independent tests support the missing ordering edge, while review challenges overlap and hook-test coverage. Both can be true; no current carry recommendation until placement is resolved.
- **“Merged but not ancestor” vs present source:** shallow release history invalidates naive ancestry conclusions. Direct production-file comparisons establish #60692/#61023 presence.

## Uncertainties & Gaps

- No cluster access, image build, CUDA compilation, GPU tests, model loading or benchmark requests were performed. Every numerical speed/capacity result above is an upstream report, not independently reproduced here.
- Mechanical patch applicability is not runtime compatibility, CI approval, correctness or permission to deploy. No leading performance/capacity PR is qualified for production by this research alone.
- Actual selected MoE backend, resolved draft RoPE/max-position configuration and current CPU-tier hit metrics were not introspected. Official target config has 1M positions and YaRN; exact runtime drafter inheritance still needs checking before #63251 adoption.
- Official config confirms `dspark_block_size=5`, vocabulary 129280, hidden 4096 and FP8 block scales; it does not establish each PR's assumptions about runtime projection layouts.
- No verified safe V4 HiSparse+DCP+DSpark+SimpleCPUOffload combination was found. Capacity gains must not be multiplied together speculatively.
- GitHub CI rollups contain automation/meta/documentation checks as well as tests. Success counts are not equivalent to a full GPU suite passing; failures are not assumed to be code defects without logs.

## Recommendations

**Primary:** keep the current base for now; prioritize exact-config qualification of **#58074**, then a properly ported/native-built **#55654**. Track **#60078** for reviewer-approved barrier placement. The only narrow immediate open-PR carry recommendation is **#58291 if developer-role traffic is actually present**, with the existing encoder tripwires and new tokenizer tests retained.

**Alternative:** evaluate a newer coherent base for merged **#54163/#63251/#63165** and Humming 0.1.18 rather than assembling their evolving dependencies manually. Verify that it still advertises and safely serves 1M with the resolved DSpark drafter. This is a research recommendation, not an executed bump.

**Not recommended:** current #53592 in production; #59916 while retaining DSpark; int8 W2 under no-requant policy; carrying high-concurrency Humming activation changes into c1/c2 service without matched benchmarks; generic GLM/V4.1 sparse-MLA changes presented as V4-0731 improvements.

For eventual qualification, preserve official weights and compare fixed cold/cached 69K prompts, large-context extension and prefix reuse, c1/c2 decode, DSpark positional acceptance, CPU-tier reload correctness, and near-1M admission/OOM behavior. These tests were **not run** here.

## Captured Review / CI Snapshot

Counters include meta checks; they are not GPU-test totals. All candidates still require review (`REVIEW_REQUIRED`); historical individual reviews can include change requests.

| PR | Head checked | Draft flag | Review decision | Check conclusions |
|---|---|---|---|---|
| #58074 | `f8bb0ba87141` | False | REVIEW_REQUIRED | SKIPPED: 2, SUCCESS: 7 |
| #60078 | `520208b140db` | False | REVIEW_REQUIRED | FAILURE: 3, SKIPPED: 6, SUCCESS: 6 |
| #55654 | `2c580b880d18` | False | REVIEW_REQUIRED | FAILURE: 1, SKIPPED: 3, SUCCESS: 5 |
| #61461 | `dcbb047275eb` | True | REVIEW_REQUIRED | SKIPPED: 4, SUCCESS: 9 |
| #53592 | `d22f72c16eea` | False | REVIEW_REQUIRED | CANCELLED: 2, SKIPPED: 4, SUCCESS: 12 |
| #59916 | `298fd3755c37` | True | REVIEW_REQUIRED | SKIPPED: 3, SUCCESS: 7 |
| #58291 | `bc57dd196eb5` | False | REVIEW_REQUIRED | CANCELLED: 4, FAILURE: 2, SKIPPED: 10, SUCCESS: 7 |
| #61824 | `532f49bcd33d` | True | REVIEW_REQUIRED | SKIPPED: 3, SUCCESS: 9 |
| #58780 | `3062cb2fa043` | False | REVIEW_REQUIRED | FAILURE: 3, SKIPPED: 7, SUCCESS: 4 |
| #52996 | `0c4b9f9d17ce` | False | REVIEW_REQUIRED | ACTION_REQUIRED: 1, CANCELLED: 2, FAILURE: 2, SKIPPED: 6, SUCCESS: 5 |
| #44573 | `29642c21fa03` | False | REVIEW_REQUIRED | SKIPPED: 2, SUCCESS: 6 |
| #47891 | `a9eb2db3460f` | False | REVIEW_REQUIRED | FAILURE: 1, SKIPPED: 3, SUCCESS: 4 |
| #61875 | `383e48f408e7` | False | APPROVED | FAILURE: 3, SKIPPED: 1, SUCCESS: 162 |
| #62958 | `19af173d3af7` | False | REVIEW_REQUIRED | CANCELLED: 8, FAILURE: 2, SKIPPED: 13, SUCCESS: 7 |
| #58016 | `bb1263c986d2` | False | REVIEW_REQUIRED | FAILURE: 1, SKIPPED: 3, SUCCESS: 5 |
| #56184 | `5aa88d5e7ea1` | False | REVIEW_REQUIRED | FAILURE: 1, SKIPPED: 2, SUCCESS: 5 |
| #57051 | `67920f920234` | False | REVIEW_REQUIRED | CANCELLED: 6, FAILURE: 2, SKIPPED: 13, SUCCESS: 7 |
| #53984 | `69a7fb3665c3` | False | REVIEW_REQUIRED | CANCELLED: 6, FAILURE: 2, SKIPPED: 8, SUCCESS: 7 |
| #61961 | `f03405748951` | False | REVIEW_REQUIRED | SKIPPED: 6, SUCCESS: 8 |
| #59265 | `2202c5c9ce7a` | True | REVIEW_REQUIRED | CANCELLED: 2, FAILURE: 2, SKIPPED: 4, SUCCESS: 4 |
| #61277 | `d63de7c5d57f` | True | REVIEW_REQUIRED | SKIPPED: 10, SUCCESS: 9 |
| #58540 | `1f7d1771277e` | True | REVIEW_REQUIRED | SKIPPED: 2, SUCCESS: 7 |
| #60320 | `69a504902ad6` | False | REVIEW_REQUIRED | SKIPPED: 8, SUCCESS: 11 |

## Methodology

Four search rounds combined varied web searches with GitHub repository searches, full PR metadata/diffs/reviews, official model configuration and release source inspection. Followed capacity → cache layout → speculative/offload constraints, and claimed speedup → actual benchmark shape → our low-concurrency relevance. Confidence by question: capacity 90%, Hopper paths 85%, DSpark/offload 90%, parser/startup 85%, merged/compatibility 90%; overall checkpoint 88%. Five questions answered; runtime efficacy remains explicitly unverified. No delegation, public posting, pushes or builds occurred.

## Sources

All PR rows and merged rows above link directly to primary-source PR pages (body, files, checks, review and discussion); dates in those tables are the captured update/merge dates. Credibility is Tier 1 for source code/state, but **author benchmark claims are not independent validation**.

| Source group | URLs / identification | Accessed | Credibility |
|---|---|---|---|
| Official model configuration | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json | 2026-10-23 | Tier 1 |
| Release source | https://github.com/vllm-project/vllm/tree/v0.32.0 | 2026-10-23 | Tier 1 |
| Ranked open candidates | PR pages individually linked in ranks 1–23; #58074, #60078, #55654, #61461, #53592, #59916, #58291, #61824, #58780, #52996, #44573, #47891, #61875, #62958, #58016, #56184, #57051, #53984, #61961, #59265, #61277, #58540, #60320 | 2026-10-23 | Tier 1 code/state; reported measurements |
| Additional offload screening | https://github.com/vllm-project/vllm/pull/54743 | 2026-10-23 | Tier 1 |
| Generic backend exclusions | https://github.com/vllm-project/vllm/pull/59319 ; https://github.com/vllm-project/vllm/pull/63104 | 2026-10-23 | Tier 1 |
| Merged upgrade screening | PR pages individually linked above: #63251, #54163, #63165, #62816, #62751, #61615, #62875 | 2026-10-23 | Tier 1 |
| Already-in-release / temporal controls | https://github.com/vllm-project/vllm/pull/60692 ; https://github.com/vllm-project/vllm/pull/61023 ; https://github.com/vllm-project/vllm/pull/58857 | 2026-10-23 | Tier 1 |
| Local compatibility evidence | `/tmp/pcai-pr-research/apply-results.json`, numbered `.diff`/`.json` files; local Dockerfile and four patches | 2026-10-23 | Direct local observation |

---

# Addendum: Non-KV long-prefill memory, revised priority

**Date:** 2026-10-23. **Depth:** deep, revised four-question scope approved. **Confidence:** 88% in source-based conclusions, not a serving qualification. Findings were sent to the requesting session before this local append. Sources: pinned v0.32.0 `5df63a7203cd`, pinned DeepGEMM `1e1842a83369`, and 34 primary-source/code items across three rounds. No runtime configuration, image or cluster change was made for this research.

## Executive Summary / revised decision

The binding constraint is **non-KV peak memory**, not simply KV capacity. Expandable segments are not a safe standalone toggle with the configured connector; enabling CuMem avoids that startup guard but also changes weights/workspace allocation and does not by itself protect ordinary CUDA-graph inputs from legacy custom-allreduce IPC incompatibility. The owner ruled out that experiment.

The indexer logits budget is a **logical per-matrix chunking budget**, not a guarantee of peak physical allocation: the native kernel pads its output, and previous-subchunk logits remain referenced while the next matrix is allocated. A smaller cap can reduce the long-context peak, but for an unsplit 280 MiB matrix split into two pieces, the two-live sum can remain 280 MiB. Upstream #63047 independently identifies this lifetime defect; #58068 addresses growing-shape fragmentation but **does not fix the lifetime**.

The two candidate configurations are (A) max length 786432, batched tokens 4096, logits cap 128 MiB, utilization 0.95; (B) max length 1048576, batched tokens 2048, logits cap 128 MiB assumed, fixed KV 7.125 GiB. A third 1M/4096 column below isolates the context-dependent comparison requested. Neither source-derived table certifies the goal of at least 1 GiB real driver-free slack with at least 7.0 GiB KV.

## Q1. Allocator safety and CuMem without sleep

### What enabling CuMem actually does

`GPUWorker._maybe_get_memory_pool_context` uses CuMem whenever `model_config.enable_cumem_allocator` is true on CUDA. Callers wrap **weights**, workspace-manager scratch, **KV**, and KV-initialization **runtime** state. It is not a KV-only switch. CuMem's `use_memory_pool` temporarily disables expandable segments and restores the **complete previous allocator configuration** on exit, avoiding accidental loss of other options. Dynamic allocations outside these contexts can still use PyTorch expandable segments. Persistent shared workspace allocations use the workspace callback and stay in CuMem. [Worker](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_worker.py), [CuMem](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/device_allocator/cumem.py), [allocator helpers](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/device_allocator/alloc_conf.py)

The relevant v0.32.0 CuMem source does **not** have an unconditional expandable-segments assertion in `use_memory_pool`: it has the temporary-disable/restore implementation. The explicit worker assertion is that the weights pool has zero existing usage when first entered. Do not apply an older assertion-based diagnosis to this tag.

### CUDA graphs and collectives remain separate concerns

`use_cumem_cudagraph_pool` is true only with sleep-mode graph offloading, sleep mode enabled, the CuMem sleep backend, non-NONE graph mode and a CUDA-like platform. **`--enable-cumem-allocator` alone, without sleep mode, does not select a CuMem graph pool.** FULL_AND_PIECEWISE and breakable eager regions do not change this predicate. [Config](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/vllm.py), [graph routing](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/compilation/cudagraph_pool.py)

The communicator sets `register_graph_buffers = not use_cumem_cudagraph_pool`; ordinary capture can therefore still use legacy graph-buffer registration. `CustomAllreduce.register_graph_buffers` calls the compiled `get_graph_buffer_ipc_meta`, whose native implementation calls `cudaIpcGetMemHandle`. VMM-backed activation pointers are not automatically made legacy-IPC-compatible by allocating KV through CuMem. The proposed fixes [#51903](https://github.com/vllm-project/vllm/pull/51903), [#55135](https://github.com/vllm-project/vllm/pull/55135), [#56193](https://github.com/vllm-project/vllm/pull/56193) are **closed unmerged**, not available fixes in this release. [Communicator](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/distributed/device_communicators/cuda_communicator.py), [custom AR](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/distributed/device_communicators/custom_all_reduce.py), [native registration](https://github.com/vllm-project/vllm/blob/v0.32.0/csrc/custom_all_reduce.cuh)

`VLLM_ALLREDUCE_USE_SYMM_MEM` defaults to 1; Hopper/TP2 supports a separate 64 MiB symmetric scratch buffer. `VLLM_USE_NCCL_SYMM_MEM` defaults to 0 and is a different switch. Symmetric scratch does not prove all captured input pointers are safe for the legacy registration path. Exact selected collective routing must be observed before any allocator qualification; no such qualification was performed here.

SimpleCPUOffload's CPU tier registers **host** storage with `cudaHostRegister` and builds DMA copies from GPU virtual base pointers. It does not register device memory with RDMA. Stable CuMem KV pointers and pinned-host DMA are compatible in principle, but this code reading is not a proof of the complete graph/offload combination. [DMA source](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/simple_kv_offload/cuda_mem_ops.py), [worker](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/simple_kv_offload/worker.py)

### Defaults, overrides, profiling and cost

No automatic global enable of expandable segments was found in the inspected release paths. CuMem temporarily toggles it; worker weight-loading and auto-sizing profile scopes temporarily set `max_split_size_mb=20` while preserving/restoring the previous configuration. The connector guard checks the legacy CUDA environment variable; using a different allocator environment-variable spelling to bypass that guard is **not** a safety fix.

With explicit `kv_cache_memory_bytes`, the worker still calls `profile_run` for compilation/workspace setup, then returns the requested KV capacity. It bypasses the automatic `memory_profiling` block and its scoped split limit, and does not use `gpu_memory_utilization` to derive that KV size. **Fixed KV does not imply no profiling forward, but it does imply no automatic safety resizing.** Workspace savings can become headroom instead of being spent on more auto-sized KV, as the owner noted.

CuMem pool callbacks, physical-page allocation and first-use JIT/capture can change startup behavior; disabling custom allreduce as an additional workaround could change latency. No credible matched startup/throughput number for our exact setup was found. Therefore no percentage cost or safe production allocator recipe is asserted. [#48243](https://github.com/vllm-project/vllm/pull/48243) provides a connector opt-out proposal, but its evidence is mainly integrated-GPU behavior and it does not solve graph IPC.

## Q2. Runtime cap, profile mismatch and cost of reducing it

V4 uses the shared `DeepseekV32IndexerMetadataBuilder` with compression ratio 4 for its indexer. The builder divides the cap by four and ensures `M × N × 4 <= budget`, splitting the query dimension for a single over-budget request. The dummy path allocates **exactly** `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB × 2^20` uint8 bytes alongside gather scratch. The setting therefore affects the real V4 path, not only an unrelated generic backend. [Builder](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/mla/indexer.py), [runtime/profile op](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/sparse_attn_indexer.py)

### Physical allocation is larger than the logical cap

For Hopper and 64 index heads, pinned DeepGEMM uses `block_q=2`, `block_kv=256` and 1024-byte row-stride alignment. With default `max_seqlen_k=0`, physical fp32 storage is:

```text
align(M, 2) × align(N + 256, 256) × 4 bytes
```

It returns an `[M,N]` view over that storage. The planner budgets the view, not the padded storage. At N=262144 (1M/C4), a logical 512 MiB chunk physically allocates 512.5 MiB, and a logical 128 MiB chunk allocates 128.125 MiB. [Pinned DeepGEMM allocation](https://github.com/vllm-project/DeepGEMM/blob/1e1842a833699298f7afc02eefb2ed168fff6938/csrc/apis/attention.hpp)

The Python `for chunk` loop does not delete `logits` before evaluating the next `logits = fp8_fp4_mqa_logits(...)`. The previous tensor is still referenced while the RHS allocates its replacement. A CPU object-lifetime probe confirms the language semantics (peak two live objects); CUDA peak sizes remain an analytical inference, independently corroborated by upstream [#63047](https://github.com/vllm-project/vllm/pull/63047). Thus worst-case dense logits can approach **two padded matrices**, whereas the dummy profile reserves one unpadded cap.

### 256 MiB and 128 MiB: what is and is not known

At 1M/C4, 512 MiB permits 512 query rows/call; 256 MiB permits 256; 128 MiB permits 128. For a global 4096-row query chunk that is 8, 16 or 32 score/top-k pairs, before TP row-shard intersection. Smaller caps may require more launches and more KV reads; continuation chunks can reuse the K-gather workspace. **No matched H100/V4 TTFT measurement was found**, so doubling calls is not a claim that total prefill time doubles.

At 70,000 input tokens (N=17500), a 4096-row matrix physically uses 280 MiB. A 256 MiB cap splits it into 3834 and 262 rows: physical allocations approximately 262.090 and 17.910 MiB. Because they can coexist, their sum is still 280 MiB. Thus lowering the cap alone may not reduce the peak of precisely that one original chunk. The reported failing request was approximately 70K; these exact 70,000-token numbers illustrate the mechanism rather than reproduce the reported 292 MiB request.

For the final two candidate configurations at cap 128 MiB:

| Quantity | A: 768K / B4096 | B: 1M / B2048 | Comparison: 1M / B4096 |
|---|---:|---:|---:|
| C4 keys N | 196608 | 262144 | 262144 |
| Full-cap logical query rows | 170 | 128 | 128 |
| Max one padded logits matrix | 127.666016 MiB | 128.125 MiB | 128.125 MiB |
| Two consecutive full matrices | **255.332031 MiB** | **256.25 MiB** | **256.25 MiB** |
| Calls for full global query chunk | 25 | 16 | 32 |

TP prefill-row sharding narrows the rows processed per rank when its gate is satisfied (CUDA, TP>1, DCP1, no PCP, no NCCL symmetric-memory path, no batch-invariant mode, and mixed graph mode not FULL). It can reduce actual local calls/allocations; the table conservatively evaluates unsharded full chunks. It must not be described as KV sharding.

## Q3. Per-buffer bounds for the revised candidate configurations

**Units are MiB per GPU unless otherwise specified. These rows are NOT additive.** Shared scratch gets reused across attention/indexer/MoE; graph views, dynamic activation liveness and metadata-builder multiplicity determine a real peak. Values derive from the release source, not nvidia-smi or a profiler run.

| Buffer / capacity | A: 768K, 4096 rows | B: 1M, 2048 rows | 1M, 4096 rows | Allocation / caveat |
|---|---:|---:|---:|---|
| FlashMLA C4 shared scratch warmup reservation, including top-k indices/lens | **794.516** | **1037.508** | **1050.516** | `4 × (ceil(L/4)+128+B) ×512×2` plus `B×(640×4+4)`; four-request reservation retained |
| Actual one-request C4 bf16 K-gather region | 196.124 | 258.124 | 260.124 | `(floor(L/4)+B+127)×512×2`; view inside shared scratch, not additional allocation |
| Indexer FP8 K-gather request | 396 | 528 | 528 | `(16L/4)×132`; aliases larger shared scratch; factor2 |
| Indexer MXFP4 K-gather alternative | 204 | 272 | 272 | `(16L/4)×68`; **not** the Hopper FP8 path assumed above |
| Humming BF16 TP2 indexed MoE scratch estimate | 240 | 120 | 240 | Conditional: local intermediate1024, topk6, hidden4096; ping-pong60 MiB/1024 rows; shared scratch, not additive; backend/padding/schema must match |
| Padded attention Q `[B,64,512]` bf16 | 256 | 128 | 256 | Independent dynamic tensor; native real-head input can overlap briefly |
| Attention output `[B,64,512]` bf16 | 256 | 128 | 256 | Separate from Q and shared scratch; caller-owned output for breakable capture |
| Unpadded real-head Q `[B,32,512]` bf16 | 128 | 64 | 128 | Input projection output; lifetime can overlap padded Q creation, not necessarily full indexer scoring |
| Indexer projected Q `[B,64,128]` bf16 | 64 | 32 | 64 | Before FP8 quantization; typically not retained throughout scoring |
| Indexer quantized Q | 32 | 16 | 32 | FP8; weights/scales add small row-linear storage |
| One hc_mult4 bf16 residual/reconstruction or persistent MTP buffer | **128** | **64** | **128** | `[B,4,4096]`; multiple differently lived instances exist; do not multiply by43 layers |
| Three reduced target aux states for DSpark | 96 | 48 | 96 | `3×B×4096×2`; reconstruction buffers can overlap them |
| Shared target top-k `[B,512]` int32 | 8 | 4 | 8 | Persistent, not per indexer layer; draft has another buffer |
| C128 metadata: global-decode + prefill index buffers | **192** | **128** | **256** | Two `[B,align(ceil(L/128),128)]` int32 buffers **per C128 metadata builder**, not per layer |
| C4 indexer expanded block-table buffer | approximately48 | approximately32 | approximately64 | `B×ceil(L/256)×4` per builder if plan retains block256; use actual cache-plan width if changed |
| C128 combined-index scratch request | 98 | 65 | 130 | `B×(ceil(L/128)+128)×4`; shared scratch, well below C4 warmup reservation |
| Dense indexer logits, two-live bound, cap128 | 255.332 | 256.25 | 256.25 | Independent of workspace-manager scratch; padded native storage |

Source anchors: [V4 attention buffer allocation](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/models/deepseek_v4/attention.py), [V4 FlashMLA prefill](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/models/deepseek_v4/nvidia/flashmla.py), [C128 metadata](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/models/deepseek_v4/sparse_mla.py), [model residual/aux buffers](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/models/deepseek_v4/nvidia/model.py), [Humming workspace metadata](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py), [workspace manager](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/workspace.py).

### Why the generic workspace factor is not the large V4 knob

Above 256K, `sparse_mla_prefill_workspace_factor` resolves to2. The shared indexer helper allocates `16×L` rows, and V4 divides by compression ratio4. However V4's custom FlashMLA attention directly reserves a four-request gather shape, independently of the generic `FlashMLASparseImpl` workspace factor. Lowering the helper does **not** remove the approximately0.8–1GiB C4 workspace above. Since indexer scratch is already below that shared maximum, reducing its bound alone may reclaim **zero physical shared-workspace bytes**. `WorkspaceManager` allocates the maximum requested byte buffer per lane and suballocates views; it does not allocate every logical scratch request independently.

The FlashMLA four-request scratch is **already budgeted**, not a late allocation: `attn_metadata is None` reserves it during `profile_run`, which runs inside `memory_profiling` for utilization-based sizing. V2 locks the workspace after capture; a later oversized request raises the workspace assertion rather than silently growing it into an OOM. With utilization-based sizing, this reservation reduces the selected KV pool; it must not be counted again as an unexplained loss of slack. With fixed KV, the operator must budget it explicitly.

The checkpoint's compression entries43–45 (DSpark/MTP layers) are0, and V4 DSpark constructs only those additional SWA-only decoder layers. Thus there is **no draft C128 cache** and one target C128 metadata builder is the expected configuration without DBO/group splitting. This is a source/config expectation, not an observed runtime inventory. The C128 metadata pair in the table should not be multiplied by20 target layers or duplicated for the drafter.

V2 DSpark config requests two workspace lanes; target and draft CUDA graphs can retain views concurrently. The target is expected to dominate the large C4 gather. Draft SWA-only warmup gather at4096 is roughly16.5MiB plus metadata, but actual draft workspace can instead be dominated by its MoE request; do not add another full1GiB merely because there are two lanes.

### Implications for the two choices

Candidate A lowers the context-dependent target workspace by approximately243MiB versus candidate B, but doubles row-linear Q/output/hc/aux and metadata buffers. Therefore “768K means less memory” does not by itself establish a smaller overall peak than1M/2048. At equal4096 rows, lowering1M to768K reduces the large shared scratch by256MiB, the C128 metadata pair by64MiB per builder, the expanded indexer table by16MiB per builder, and some context-sized rotary/index metadata; it does not reduce a single row-linear activation.

The table omits exact graph-private retention, allocator-bin fragmentation, CuTe/DeepGEMM planner caches, selected MoE schema padding, all static weight-side buffers and actual metadata-builder counts. These omissions prevent a precise sum or a guarantee of1GiB slack. Real slack means CUDA driver-free memory during the worst step, not `reserved - allocated` bytes that may be stranded in unusable segments.

For utilization-based sizing, the startup reserve outside the requested budget is `initial free device memory - utilization × total device memory`. With 91.45 GiB initially free of 93.09 GiB total, utilization 0.97 leaves about 1.16 GiB and 0.95 leaves about 3 GiB. Do not infer over-budget KV from the post-capture suggested KV size versus the current KV allocation: that diagnostic accounts for the graph estimate and actual pool separately and subtracts an additional margin. The selected 0.95 candidate preserves roughly 3 GiB outside the requested budget; runtime high-watermark measurements are still required.

## Q4. Upstream peak-memory / fragmentation candidates

Full-diff `git apply --check` below used only `/tmp/pcai-image-source-check` at v0.32.0 plus current image patches. It does not imply runtime validation or authorize carrying an unreviewed subset.

| PR | State at inspection | Actual relevance | Release apply result / recommendation |
|---|---|---|---|
| [#58068 Fixed-width sparse-indexer logits](https://github.com/vllm-project/vllm/pull/58068) | OPEN; head `1631c22e88f5` | Directly relevant shared V4/SM90 dense path; Python forwards existing native `max_seqlen_k`; opt in on discrete Hopper with `VLLM_SPARSE_INDEXER_FIXED_LOGITS_WIDTH=1`. Stabilizes sizes, not a lifetime fix. | Full diff FAIL: GLM-specific hunk and shared-builder call context. Relevant production sections can be ported without new CUDA operators. Owner requested separate draft experiment, no dispatch. |
| [#63047 Budget prefill memory startup misses](https://github.com/vllm-project/vllm/pull/63047) | OPEN; head `54feb1b6a049` | Shared loop adds `del logits`; shared builder sizes expanded table by actual max decode rows, while other changes are V4.1 Engram-specific. With6 seqs/k5, decode row bound36 could replace B4096/2048 table rows. | Full diff FAIL in common op/V4.1 model. **Strongest narrowly scoped peak-reduction follow-up to investigate**, but subset port/test not yet approved; GPU validation pending upstream. |
| [#60493 Reserve indexer metadata before KV sizing](https://github.com/vllm-project/vllm/pull/60493) | OPEN | Budgets late expanded-table allocation; does not inherently reduce it. Fixed KV is preserved, so it cannot manufacture extra slack. | FAIL; release lacks new planner module. Prefer understanding/shrinking the actual buffer to a broad planner backport. |
| [#62860 Reserve decode indexer logits](https://github.com/vllm-project/vllm/pull/62860) | OPEN | Profiling accuracy for decode/mixed eager logits, not the failed long-prefill allocation. | FAIL common op context. At our small6-seq bound, quantify before prioritizing. |
| [#63038 Budget compressed width](https://github.com/vllm-project/vllm/pull/63038) | OPEN | Fewer chunks for multiple prefills; no reduction of c1 max dense matrix. Depends conceptually on newer compressed-logits API. | **PASS** mechanically. Not an answer to the single-user peak constraint. |
| [#60618 Single-launch fused scoring+top-k](https://github.com/vllm-project/vllm/pull/60618) | OPEN/WIP | Avoids full logits, but source explicitly gates SM100 and adds native CUDA/C++ operators. | FAIL; **skip Hopper**. No H100 no-materialization path established. |
| [#48726 LiteTopK/LiteDSA](https://github.com/vllm-project/vllm/pull/48726) | OPEN | SM100 tcgen05 kernels, native build and substantial workspace; not a Hopper path. | Not apply-tested; **skip for this hardware**. |
| [#62240 BF16 logits + DeepSelect](https://github.com/vllm-project/vllm/pull/62240) | OPEN | Removes SM100 BF16-to-FP32 copy; SM90 currently returns FP32. | FAIL; **not a Hopper memory fix**. |
| [#55573 Gather before hc_head](https://github.com/vllm-project/vllm/pull/55573) | OPEN | Reduces discarded final-head work, eager-only gate; does not shrink main hc residual during attention. | FAIL V2 runner hunk. Secondary, not a material long-prefill memory solution. |
| [#45061 Optimize V4 prefill chunk planning](https://github.com/vllm-project/vllm/pull/45061) | MERGED Jun15 | Bounds request packing by workspace area; dynamic chunk planning already present in release source. | Already present behavior, not a new bump reason. |
| [#48243 Connector allocator opt-out](https://github.com/vllm-project/vllm/pull/48243) | OPEN | Bypasses conservative connector guard on operator assertion; does not repair custom-allreduce graph IPC. | FAIL; owner ruled out allocator experiment. |

### Fixed-width caveats for the authorized draft experiment

The requested backport is [draft image PR #55](https://github.com/enthus-appdev/vllm-pcai/pull/55), branch `claude/v4-indexer-fixed-width-logits`, head `b8a6fc992c0fa8bb4a9ab388e537a4bea1ece7b2`. It is not part of the current image. The production-only Python backport pins SHA256 `967eda4bd76a3f94cbdfccaebec7e9bbd27b86cb8be225ab5c2b666924cffb86`; isolated patch application, CPU tripwires and patched-source parsing provide source-level checks, not serving/GPU qualification.

Fixed width is **ceil(max_model_len/4)** on this V4 indexer. At1M/cap128 it forces128-row subchunks from the first prompt; at768K/cap128 it permits170 rows. The native fixed-width allocation avoids the default additional256 key columns, but still rounds query rows to2 and keeps the two-live behavior. For early short contexts this may materially **increase** allocation and launch count relative to the dynamic path. Upstream defaults it to integrated GPUs for exactly that unmeasured short-prompt trade-off; the Hopper experiment must remain explicitly opt-in, with old behavior available via0/unset.

## C128 state accounting

The image carries the C128 circular-state patch through [image PR #54](https://github.com/enthus-appdev/vllm-pcai/pull/54). The calculations below describe source geometry, not GPU qualification or a guaranteed increase in free HBM.

At k5, the ring consumes20MiB/request/rank across20 layers, max6=120MiB/rank (240MiB summed TP2). Legacy pages contain8×1024 FP32 values, aligned to32832bytes; each additional retained row across20 layers costs82080bytes/rank, but legacy sliding history is **bounded**, not1M times that number. For1024-token one-batch admission, `ceil((127+1024)/8)+1=145` pages/layer gives90.802002MiB versus20MiB ring, a70.802002MiB raw bound reduction/request/rank. Two-batch2048 bound gives170.958252MiB versus20MiB, a150.958252MiB reduction. These are before pool packing and not exact freed HBM. At a six-row decode bound, ring raw storage is larger than legacy. These state-allocation bounds do not establish the peak of the separate indexer/activation path.

## Recommendations and validation limits

1. Keep allocator experiments off the production path as directed. CuMem plus expandable segments is not a source-proven safe complete recipe for this configuration.
2. Treat cap128 as approximately256MiB of potentially live padded dense logits on stock release, not128MiB peak. Evaluate fixed-width as a fragmentation/shape-reuse experiment with possible early-prompt regressions, not as an unconditional memory reduction.
3. The most promising separate narrow code investigation is the **shared logits lifetime and actual-decode-row metadata sizing** in#63047. The current request authorizes only research and the#58068 draft, so no#63047 subset was applied.
4. The owner's selected ladder-test candidate is **768K /4096 /utilization0.95 /cap128**; the alternative remains1M /2048 /fixedKV7.125GiB. Compare using fixed weights/quantization, capture actual workspace/metadata-builder counts and driver-free high-watermark, and require at least1GiB measured free slack. Do not infer the goal from startup profiling or sum this nonadditive table.

**Unverified:** throughput penalty, total peak, builder multiplicity, exact chosen MoE schema and graph retention. Code inspections, isolated planner execution and object-lifetime simulation are not CUDA validation. Research evidence lives under `/tmp/pcai-memory-research/` (`analyze_source.py`, `estimates.json`, pinned native header, PR metadata/diffs). No cluster access, model requests, deploys or image dispatches occurred for this addendum.

### Additional sources (all accessed2026-10-23)

Every source linked in this addendum is a Tier1 upstream repository/code/PR record. Native allocation and release code are authoritative for implementation; upstream performance claims remain author reports. Direct local observations are the planner/byte arithmetic/lifetime script and isolated apply checks. Key source inventory: release config, worker, CuMem/alloc-conf, graph routing, custom AR Python/native code, communicator/symmetric helpers, simple offload DMA/worker, shared indexer builder/runtime, V4 attention/FlashMLA/sparse metadata/model/DSpark sources, workspace manager/Humming expert implementation, pinned DeepGEMM attention API, and PRs#58068/#63047/#60493/#62860/#63038/#60618/#48726/#62240/#55573/#45061/#48243/#51903/#55135/#56193. Explicit URLs are provided at each discussion/table entry above.
