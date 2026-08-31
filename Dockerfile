# PCAI can't mount volumes, so the chat templates are baked in. Details: README.
#
# DeepSeek-V4-Flash-Vision-Exp support is based on the 2026-08-31 nightly plus the implementation
# linked from vllm#54561. Keep the base at that implementation's nearest published ancestor so the
# vendored patch remains reviewable; its build-time tripwire must fail on incompatible bumps.
#
# Back on a nightly, reluctantly: the DSV4 KV-capacity work all landed after the v0.26.0 branch cut.
# vllm#48993 (packed KV group overlays: per-block cost sum(groups) -> max(groups)) and vllm#48317
# (get_max_concurrency_for_kv_cache_config counted only ONE group's page size, so every concurrency
# figure we ever recorded was overstated) are the reasons; #48957/#49486/#50004 ride along.
# v0.26.1rc0 has the first two but is a git tag only — no image is published.
#
# vllm#50298 (~1.88x kernel) shipped an early-return warmup branch in
# models/deepseek_v4/nvidia/flashmla.py asserting `self.topk_indices_buffer is not None`. A DSpark
# drafter has no indexer buffer, so profile_run died with a bare AssertionError AFTER the full
# ~32 min weight load (vllm#50615). We pinned below it until vllm#50693 fixed it; that patch is now
# vendored, so the pin is lifted and #50298/#50312/#49236/#48047 ride along.
# The fix only helps because our draft layers are SWA-only: compress_ratios has 46 entries for 43
# hidden layers and the trailing three (the MTP/DSpark layers) are 0, so `swa_only` is True and the
# assert is never reached. A checkpoint whose draft layers compress would still trip it.
#
# ⚠ Nightly tags are pruned (~2 weeks). If a rebuild fails on an unresolvable FROM, that is why —
# move to the first release tag that is a superset, do not silently pick a newer nightly.
#
# Bumping is not a date comparison: vLLM cuts release branches, so a later tag can MISS commits.
# `gh api repos/vllm-project/vllm/compare/<current>...<target> --jq .status` should say "ahead".
# If it says "diverged", check whether the behind-by commits are backports that exist on main under
# different SHAs (they usually are) before treating it as a blocker.
FROM vllm/vllm-openai:nightly-44fe2a392b71d52a8d72faf2f8278834379482c9

# Exact four-commit patch series from upstream vllm#54566 at head
# 93274397143dfc0135b6b0672859ffcab6e725a4. This PR changes both Python and the compiled
# topk_softplus_sqrt operator, so patching site-packages alone would create an ABI mismatch.
# Rebuild vLLM from the pinned base source, then install it over the stock wheel.
COPY patches/54566-deepseek-v4-vision.patch /tmp/54566.patch
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends cmake cuda-nvrtc-dev-13-0 git ninja-build; \
    rm -rf /var/lib/apt/lists/*; \
    test "$(sha256sum /tmp/54566.patch | cut -d' ' -f1)" = "c1205b5d1f6798d7d5dbbcef7d192bebe45a032291c8855b7c174750c342d86f"; \
    git clone --filter=blob:none https://github.com/vllm-project/vllm.git /tmp/vllm-src; \
    git -C /tmp/vllm-src checkout 44fe2a392b71d52a8d72faf2f8278834379482c9; \
    git -C /tmp/vllm-src apply --check /tmp/54566.patch; \
    git -C /tmp/vllm-src apply /tmp/54566.patch; \
    VLLM_DIR="$(cd / && python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n 1)"; \
    cd /tmp/vllm-src; \
    cmake -S . -B /tmp/vllm-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DCMAKE_JOB_POOL_COMPILE=compile \
      -DCMAKE_JOB_POOLS=compile=4 \
      -DVLLM_PYTHON_EXECUTABLE="$(command -v python3)" \
      -DVLLM_PYTHON_PATH="$(python3 -c 'import sys; print(":".join(sys.path))')" \
      -DVLLM_TARGET_DEVICE=cuda; \
    cmake --build /tmp/vllm-build --target _C_stable_libtorch -j 4; \
    cp -a vllm/. "$VLLM_DIR/"; \
    SO="$(find /tmp/vllm-build -type f -name '_C_stable_libtorch*.so' -print -quit)"; \
    test -n "$SO"; \
    cp "$SO" "$VLLM_DIR/_C_stable_libtorch.abi3.so"; \
    rm -rf /tmp/vllm-src /tmp/vllm-build /tmp/54566.patch

# Qwen's enhanced template is baked (not upstream); Gemma uses vLLM's in-image template — serve with
#   --chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja
COPY chat-template-fix/chat-template/*.jinja /templates/

# PCAI has no shell and no pod logs, so expose vLLM's collect_env over the serving port as
# GET /collect_env. No route auth of its own — the LB's per-isvc bearer is the only gate, same
# as every other route here.
COPY diag/collect_env_route.py /tmp/collect_env_route.py
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; \
    cat /tmp/collect_env_route.py >> "$VLLM_DIR/entrypoints/openai/api_server.py"; \
    rm -f /tmp/collect_env_route.py; \
    python3 -c "import inspect; import vllm.entrypoints.openai.api_server as m; from vllm.collect_env import get_pretty_env_info; assert hasattr(m, '_pcai_collect_env_wrapper'); assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in inspect.signature(m.build_app, follow_wrapped=False).parameters.values()), 'wrapped build_app must stay variadic'; print('collect_env route baked OK')"

# Tripwire: --reasoning-parser/--tool-call-parser deepseek_v4 depend on the base providing these;
# a bump that drops or renames them fails HERE. DSML_PARAM_CLOSE guards the streaming close-tag
# leak fix. Does NOT prove streaming correctness — needs GPU.
RUN python3 - <<'PY'
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
r = ReasoningParserManager.get_reasoning_parser("deepseek_v4")
t = ToolParserManager.get_tool_parser("deepseek_v4")
assert r.__name__ == "DeepSeekV4ParserReasoningAdapter", r
assert t.__name__ == "DeepSeekV4EngineToolParser", t
from vllm.parser.deepseek_v4 import deepseek_v4_config, DSML_PARAM_CLOSE
import vllm.parser.deepseek_v32  # noqa: F401
assert deepseek_v4_config(thinking=True).initial_state.name == "REASONING"
assert deepseek_v4_config(thinking=False).initial_state.name == "CONTENT"
assert DSML_PARAM_CLOSE == "</｜DSML｜parameter>", DSML_PARAM_CLOSE
print("deepseek_v4/v32 engine parsers OK:", r.__module__, "/", t.__module__)
PY

# The tokenizer was replaced upstream after vllm#46257. Its old add_generation_prompt /
# continue_final_message patch no longer applies; renderer-level behavior must be exercised through
# /v1/chat/completions in the GPU acceptance test rather than against the removed encoder API.
RUN python3 - <<'PY'
from vllm.tokenizers.deepseek_v4_encoding import encode_messages
out = encode_messages(
    [{"role": "user", "content": "hi"}],
    thinking_mode="thinking",
    reasoning_effort="max",
)
assert "hi" in out
print("deepseek_v4 encoder max-reasoning path OK")
PY

# vllm#48748 (a reply that never emits </think> leaves the EOS token in reasoning_content) is IN
# THIS BASE, so the vendored patch is gone. The tripwire stays as a regression check on the base:
# the symbols must still exist (so an upstream rename fails here instead of passing vacuously) and
# the DROP_TERMINAL branch must not gate on skip_tool_parsing. Source-level, not behavioral —
# driving the parser needs vLLM's test fixtures (MockTokenizer et al), absent from the runtime image.
RUN python3 - <<'PY'
import inspect, re
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
cls_src = inspect.getsource(StreamingParserEngine)
fn_src = inspect.getsource(StreamingParserEngine._on_terminal)
assert "skip_tool_parsing" in cls_src, "symbol gone — re-check whether vllm#48748 still applies"
assert "DROP_TERMINAL" in fn_src, fn_src
branch = re.search(r"if[^:]*DROP_TERMINAL[^:]*:", fn_src, re.S)
assert branch, fn_src
assert "skip_tool_parsing" not in branch.group(0), branch.group(0)
print("EOS-in-reasoning fix (vllm#48748) present in base OK")
PY

# Qwen's DFlash drafter mixes sliding + full attention, which only the V2 model runner can do
# (multiple KV groups). vLLM auto-forces V2 for it via this helper, which is why no serve-arg or
# env change is needed; a base that loses it would leave V1 selected and fail at boot instead.
# Tripwire: does NOT prove acceptance — a mis-wired drafter still serves CORRECT text, just slower
# than no speculation at all. Verify on Grafana acceptance (~28-30%), not on output looking fine.
RUN python3 - <<'PY'
from vllm.config.vllm import VllmConfig
from vllm.config.speculative import SpeculativeConfig
assert hasattr(VllmConfig, "_dflash_needs_multi_kv_group")
# #48787: lets the target run fp8 KV while the drafter stays BF16 (untested here, see docs).
assert "kv_cache_dtype" in SpeculativeConfig.__annotations__, SpeculativeConfig.__annotations__
import vllm.model_executor.models.qwen3_dflash  # noqa: F401
print("hybrid-SWA DFlash prereqs OK")
PY

# DSpark needs a checkpoint that ships the drafter (deepseek-ai/DeepSeek-V4-Flash-0731; the separate
# -DSpark repo it replaced is retired) + cudagraphs (no --enforce-eager). num_speculative_tokens must
# be >= config.json's dspark_block_size (5) — below it the block drafter emits GARBLED text, not just
# lower acceptance. Leave draft_sample_method default — probabilistic degrades into loops.
# Tripwire: a bump that drops DSpark fails HERE. Does NOT prove the Hopper kernels lower or that
# acceptance is good — both need GPU + the checkpoint.
RUN python3 - <<'PY'
import vllm
from vllm.config.speculative import SpeculativeMethod
assert "dspark" in repr(SpeculativeMethod), repr(SpeculativeMethod)
import vllm.models.deepseek_v4.nvidia.dspark  # noqa: F401
import vllm.v1.worker.gpu.spec_decode.dspark.speculator  # noqa: F401
print("DSpark OK:", vllm.__version__)
PY

# Exact-PR tripwire: verifies the model and dedicated processor from vllm#54566 survived all
# subsequent PCAI overlays.
RUN python3 - <<'PY'
from transformers.processing_utils import ProcessorMixin
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.deepseek_v4.common.mm_preprocess import (
    DeepseekV4VLProcessingInfo,
    DeepseekV4VLProcessor,
)
from vllm.models.deepseek_v4.nvidia.vl_model import (
    DeepseekV4ForConditionalGeneration,
)
arch = "DeepseekV4ForConditionalGeneration"
assert arch in ModelRegistry.get_supported_archs(), arch
assert issubclass(DeepseekV4VLProcessor, ProcessorMixin)
assert DeepseekV4VLProcessingInfo.get_hf_processor.__annotations__["return"] is DeepseekV4VLProcessor
print("exact vllm#54566 Vision model and processor registered OK")
PY

# With --load-format runai_streamer and an s3:// model, ModelConfig rewrites `model` to a local
# config-only cache dir (json/py/model, never safetensors) and keeps the URL in `model_weights`.
# Drafters that live INSIDE the target checkpoint (DSpark, MTP) are built from the rewritten path
# and inherit an empty `model_weights`, so the drafter load dies with "Cannot find any safetensors
# model weights" AFTER the ~16 min target stream. Upstream vllm#48023 (fixes vllm#42060); vendored
# because it is still open. Drop once the base carries it — the apply below will fail loudly.
COPY patches/48023-spec-draft-inherit-model-weights.patch /tmp/spec-draft-weights.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/spec-draft-weights.patch; \
    else patch -p1 -d "$SITE" < /tmp/spec-draft-weights.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/spec-draft-weights.patch

# Build-time tripwire: behavioral (pure config, no GPU/network). Asserts BOTH directions — a shared
# checkpoint inherits the object-storage URL, and a drafter that resolved its own is not clobbered.
RUN python3 - <<'PY'
from unittest.mock import MagicMock, patch
from vllm.config import ParallelConfig
from vllm.config.speculative import SpeculativeConfig

S3, CACHE = "s3://llm-model-cache-std-01/DeepSeek-V4-Flash-0731/", "/root/.cache/vllm/assets/model_streamer/abcd1234"

def build(draft_weights, draft_model):
    with patch("vllm.config.speculative.ModelConfig") as mc:
        d = MagicMock(); d.model = draft_model; d.model_weights = draft_weights
        d.hf_config.model_type = "deepseek_mtp"; d.hf_config.n_predict = None; d.max_model_len = 4096
        mc.return_value = d
        t = MagicMock(); t.model = CACHE; t.model_weights = S3
        t.hf_text_config.model_type = "deepseek_v3"; t.quantization = None; t.max_model_len = 4096
        try:
            SpeculativeConfig(method="mtp", num_speculative_tokens=1,
                              target_model_config=t, target_parallel_config=ParallelConfig())
        except Exception as e:
            # These mocks track SpeculativeConfig.__post_init__; a base bump can require more
            # attributes. That is a stale-tripwire failure, NOT evidence the patch is unneeded.
            raise SystemExit(f"tripwire mocks no longer match this base ({type(e).__name__}: {e})"
                             " — update the mocks, do NOT drop the patch") from e
        return d.model_weights

assert build("", CACHE) == S3, "shared-checkpoint drafter did not inherit model_weights"
assert build("s3://other/drafter/", CACHE) == "s3://other/drafter/", "clobbered a drafter's own weights"
print("spec-draft model_weights inheritance OK")
PY

# PCAI fixes /dev/shm at 64 MiB and exposes no way to change it (that needs an emptyDir volume,
# which PCAI forbids). vllm#48879 added check_shm_free_space, which now refuses to over-commit —
# correct, but fatal here: MessageQueue's rpc_broadcast_mq honours VLLM_MQ_MAX_CHUNK_BYTES_MB while
# worker_response_mq (created once PER WORKER) hardcodes the 24 MiB default, so TP=2 needs
# 2 x 240 MiB no matter what the env var says. Not filed upstream yet; the env var plainly intends
# to bound shm usage, so one call site ignoring it is a bug.
COPY patches/mq-worker-response-honour-chunk-bytes.patch /tmp/mq-chunk.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/mq-chunk.patch; \
    else patch -p1 -d "$SITE" < /tmp/mq-chunk.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/mq-chunk.patch

# 1 MiB x 10 chunks: rpc 10 MiB + one 10 MiB response queue per worker = 30 MiB at TP=2, inside the
# 64 MiB PCAI allows. The 24 MiB default exists for grammar bitmasks at 1024 requests; we serve
# max-num-seqs 4 (~32 KiB), and an oversized message degrades to a local socket rather than failing.
# Override per-deployment if a model ever needs bigger messages AND has the shm for them.
ENV VLLM_MQ_MAX_CHUNK_BYTES_MB=1

# Tripwire: both queue call sites must honour the env var, or a small-/dev/shm host dies at boot
# after the (very long) weight load. Source-level — constructing a real MessageQueue needs shm.
RUN python3 - <<'PY'
import inspect, re
from vllm.v1.executor import multiproc_executor as m
src = inspect.getsource(m)
sites = re.findall(r"MessageQueue\((?!\s*\)).*?\)", src, re.S)
assert sites, "no MessageQueue(...) construction found — upstream refactor, re-check the patch"
bare = [s for s in sites if "max_chunk_bytes" not in s]
assert not bare, f"MessageQueue built without max_chunk_bytes: {bare}"
import vllm.envs as envs
assert envs.VLLM_MQ_MAX_CHUNK_BYTES_MB == 1, envs.VLLM_MQ_MAX_CHUNK_BYTES_MB
print("all MessageQueue sites honour VLLM_MQ_MAX_CHUNK_BYTES_MB OK")
PY

# vllm#50693 is in this base. Keep the source tripwire: a DSpark drafter has no indexer buffer, so
# the SWA-only warmup path must be selected before that buffer is touched.
RUN python3 - <<'PY'
import inspect
from vllm.models.deepseek_v4.nvidia import flashmla
src = inspect.getsource(flashmla)
marker = "if attn_metadata is None:"
assert marker in src, "warmup branch gone — re-read forward_mqa before trusting this patch"
warmup = "\n".join(src[src.index(marker):].split("\n")[:35])
guard = warmup.find("if swa_only:")
assert guard != -1, "swa_only guard missing from the warmup branch -> vllm#50693 not applied"
first_assert = warmup.find("assert self.topk_indices_buffer is not None")
assert first_assert == -1 or first_assert > guard, (
    "warmup branch asserts on topk_indices_buffer BEFORE the swa_only guard: a DSpark drafter "
    "would die in profile_run after the full weight load (vllm#50615)"
)
print("flashmla warmup branch is DSpark-safe OK")
PY

# Clients that replay one assistant turn as two consecutive assistant messages (a content-only
# preamble, then a tool_calls + reasoning message) get a malformed prompt: the encoder emits the
# Assistant transition only after user/developer messages, so the second message is glued on with a
# stray <｜end▁of▁sentence｜> mid-turn and its reasoning renders as free text closed by an orphan
# </think>. The model imitates the unbalanced markup and emits stray </｜DSML｜tool_calls> or drops a
# quote inside a tool-call header, which the parser then reads as a garbage tool name.
COPY patches/50686-merge-consecutive-assistant-messages.patch /tmp/dsv4-consec.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/dsv4-consec.patch; \
    else patch -p1 -d "$SITE" < /tmp/dsv4-consec.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/dsv4-consec.patch

# Tripwire: functional, not source-level — the encoder is dependency-free, so we can render a split
# turn and assert the prompt is well-formed. Catches both a dropped patch and an upstream rewrite
# that reintroduces the bug.
RUN python3 - <<'PY'
from vllm.tokenizers import deepseek_v4_encoding as enc

msgs = [
    {"role": "user", "content": "check server status"},
    {"role": "assistant", "content": "Let me check the server status."},
    {"role": "assistant", "reasoning": "The user wants a health check.",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "uptime"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "up 3 days"},
]
out = enc.encode_messages(msgs, thinking_mode="thinking", drop_thinking=False)

canonical = ("<｜Assistant｜><think>The user wants a health check.</think>"
             "Let me check the server status.\n\n<｜DSML｜tool_calls>")
assert canonical in out, f"split assistant turn not merged (vllm#50686):\n{out}"
# Only the trailing generation prompt may leave a <think> unclosed.
assert out.count("<think>") - 1 == out.count("</think>"), f"unbalanced think tags:\n{out}"
# A merged turn ends once; a stray mid-turn EOS is the original bug.
assert out.count("<｜end▁of▁sentence｜>") == 1, f"stray mid-turn EOS:\n{out}"
print("deepseek_v4 encoder merges split assistant turns OK")
PY
