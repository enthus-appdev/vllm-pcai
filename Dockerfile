# PCAI can't mount volumes, so the chat templates are baked in. Details: README.
#
# Back on a nightly, reluctantly: the DSV4 KV-capacity work all landed after the v0.26.0 branch cut.
# vllm#48993 (packed KV group overlays: per-block cost sum(groups) -> max(groups)) and vllm#48317
# (get_max_concurrency_for_kv_cache_config counted only ONE group's page size, so every concurrency
# figure we ever recorded was overstated) are the reasons; #48957/#49486/#50004 ride along.
# v0.26.1rc0 has the first two but is a git tag only — no image is published.
#
# ⚠ Pinned to the 07-29 nightly ON PURPOSE, not the newest. vllm#50298 (merged 07-30) added an
# early-return warmup branch to models/deepseek_v4/nvidia/flashmla.py that asserts
# `self.topk_indices_buffer is not None`. A DSpark drafter has no indexer buffer, so profile_run
# dies with a bare AssertionError AFTER the full ~32 min weight load. Verified: the assert is
# absent here and in v0.26.0, and #50298 is the only commit touching that file in between.
# Cost of stopping short: #50298 (~1.88x kernel) and #50312 (448 MiB) — both landed 07-30 with
# the bug. Re-test them once it is fixed upstream.
#
# ⚠ Nightly tags are pruned (~2 weeks). If a rebuild fails on an unresolvable FROM, that is why —
# move to the first release tag that is a superset, do not silently pick a newer nightly.
#
# Bumping is not a date comparison: vLLM cuts release branches, so a later tag can MISS commits.
# `gh api repos/vllm-project/vllm/compare/<current>...<target> --jq .status` should say "ahead".
# If it says "diverged", check whether the behind-by commits are backports that exist on main under
# different SHAs (they usually are) before treating it as a blocker.
FROM vllm/vllm-openai:nightly-6f91edf96d3f3272945809c04702380053bff4de

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

# The deepseek_v4/v32 tokenizer-mode encoders silently ignore add_generation_prompt +
# continue_final_message without this. Vendored; drop once upstream (vllm#46257) lands in the base.
COPY patches/deepseek-add-gen-prompt-on-nightly.patch /tmp/dsv4-genprompt.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/dsv4-genprompt.patch; \
    else patch -p1 -d "$SITE" < /tmp/dsv4-genprompt.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/dsv4-genprompt.patch

# Build-time tripwire: behavioral (pure templating, no GPU) — verifies both params are honored.
RUN python3 - <<'PY'
from vllm.tokenizers.deepseek_v4_encoding import (
    encode_messages, ASSISTANT_SP_TOKEN as A, eos_token as EOS, thinking_end_token as ET,
)
u = [{"role": "user", "content": "hi"}]
a = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
assert encode_messages(u, thinking_mode="chat", add_generation_prompt=True).endswith(A + ET)
assert A not in encode_messages(u, thinking_mode="chat", add_generation_prompt=False)
assert encode_messages(a, thinking_mode="chat", add_generation_prompt=True).endswith(A + ET)
c = encode_messages(a, thinking_mode="chat", continue_final_message=True)
assert c.endswith("yo") and not c.endswith(EOS) and A in c
print("deepseek add_generation_prompt / continue_final_message honored OK")
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
