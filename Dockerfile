# PCAI can't mount volumes, so the chat templates are baked in. Details: README.
#
# vLLM v0.34.0 (commit 62d4bc3c7e33), pinned by digest.
FROM vllm/vllm-openai:v0.34.0@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1

# jq for the pcai-inference chart's pre-start init scripts: the storage check and
# file staging run in this image as POSIX sh + jq (no Python in the chart).
RUN apt-get update && apt-get install -y --no-install-recommends jq && rm -rf /var/lib/apt/lists/*

# Qwen's enhanced template is baked (not upstream); Gemma uses vLLM's in-image template — serve with
#   --chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja
COPY chat-template-fix/chat-template/*.jinja /templates/

# PCAI has no shell and no pod logs, so expose vLLM's collect_env over the serving port as
# GET /collect_env. No route auth of its own — the LB's per-isvc bearer is the only gate, same
# as every other route here.
COPY diag/collect_env_route.py /tmp/collect_env_route.py
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; \
    cat /tmp/collect_env_route.py >> "$VLLM_DIR/entrypoints/launchers/app.py"; \
    rm -f /tmp/collect_env_route.py

# Build the actual ASGI app: importing a wrapper on the deprecated shim did not register a route.
RUN python3 - <<'PY'
import inspect
from argparse import Namespace
from fastapi.testclient import TestClient
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.common.cli_args import FrontendArgs
from unittest.mock import patch

# Frontend defaults avoid instantiating GPU-dependent engine config on the CI runner.
args = Namespace(**vars(FrontendArgs(root_path="")))
app = build_app(args, supported_tasks=("generate",))
route = next(route for route in app.routes if getattr(route, "path", None) == "/collect_env")
assert not inspect.iscoroutinefunction(route.endpoint)
with patch("vllm.collect_env.get_pretty_env_info", return_value="pcai-env-test"):
    response = TestClient(app).get("/collect_env")
assert response.status_code == 200, response.text
assert response.text == "pcai-env-test", response.text
assert response.headers["content-type"].startswith("text/plain")
print("collect_env route registered and serves diagnostics OK")
PY

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
# continue_final_message without this. Exact source diff from rebased vllm#46257 head
# 3dd748819f25fa64250258cdb059c5b0e4487563; it also applies cleanly to v0.30.0.
COPY patches/46257-deepseek-generation-controls-v0.30.patch /tmp/dsv4-genprompt.patch
RUN set -eux; \
    test "$(sha256sum /tmp/dsv4-genprompt.patch | cut -d' ' -f1)" = "238e8f7b17debf344c87d1ee001a38df6ecbc397df8ee770fbe26f74a108774d"; \
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; SITE="$(dirname "$VLLM_DIR")"; \
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

# vllm#48748 is upstream in v0.30.0. Keep the source-level regression tripwire: terminal drops
# must happen independently of tool parsing so EOS cannot leak into reasoning_content.
RUN python3 - <<'PY'
import inspect, re
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
cls_src = inspect.getsource(StreamingParserEngine)
fn_src = inspect.getsource(StreamingParserEngine._on_terminal)
assert "skip_tool_parsing" in cls_src, "symbol gone — re-check vllm#48748"
assert "DROP_TERMINAL" in fn_src, fn_src
branch = re.search(r"if[^:]*DROP_TERMINAL[^:]*:", fn_src, re.S)
assert branch, fn_src
assert "skip_tool_parsing" not in branch.group(0), branch.group(0)
print("EOS-in-reasoning fix (vllm#48748) present upstream")
PY

# Qwen's DFlash drafter needs mixed full/SWA cache support and a draft KV dtype override.
# v0.34.0 uses the GPU V2 runner; the old per-model V2 selection helper is gone.
RUN python3 - <<'PY'
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.qwen3_dflash import dflash_has_any_sliding
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig

assert "kv_cache_dtype" in SpeculativeConfig.__annotations__
assert DFlashSpeculatorConfig.base_method == "dflash"
assert callable(dflash_has_any_sliding)
print("hybrid-SWA DFlash prerequisites OK")
PY

# vllm#52923 prevents OffloadingConnector store scheduling from outrunning either allocated GPU
# chunks or available offload keys. This is the exact invariant that crashed the v0.26.0 engine.
RUN python3 - <<'PY'
import inspect
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import RequestOffloadState
src = inspect.getsource(RequestOffloadState.storable_chunks)
assert "num_allocated_chunks" in src, src
assert "num_keyed_chunks" in src, src
assert "return min(num_chunks, num_allocated_chunks, num_keyed_chunks)" in src, src
print("OffloadingConnector chunk bounds fix (vllm#52923) present")
PY

# DSpark needs the DSpark checkpoint (deepseek-ai/DeepSeek-V4-Flash-DSpark) + cudagraphs (no
# --enforce-eager); leave draft_sample_method default — probabilistic degrades into loops.
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
# Preserve the object-storage weight URL for a drafter sharing the target checkpoint
# (vllm#48023). v0.34.0 handles implicit MTP upstream but still defaults DeepSeek
# DSpark to the rewritten config-only path. The common constructor keeps the guard
# for implicit DSpark and explicit shared-path drafts without clobbering other drafts.
COPY patches/48023-spec-draft-inherit-model-weights-v034.patch /tmp/spec-draft-weights.patch
RUN set -eux; \
    test "$(sha256sum /tmp/spec-draft-weights.patch | cut -d' ' -f1)" = "bcb475b185aae8c032302b8270087be4cdf9ef014dde27008f97929d43551f14"; \
    SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    patch --batch --forward --fuzz=0 -p1 -d "$SITE" < /tmp/spec-draft-weights.patch; \
    rm -f /tmp/spec-draft-weights.patch

RUN python3 - <<'PY'
from types import SimpleNamespace as Config
from vllm.v1.worker.gpu.spec_decode.speculator_config import _inherit_shared_draft_weights

target = Config(model="/config-only", model_weights="s3://example/target/")
for path, weights, expected in (
    (target.model, "", target.model_weights),
    (target.model, "s3://example/drafter/", "s3://example/drafter/"),
    ("/other", "", ""),
):
    draft = Config(model=path, model_weights=weights)
    _inherit_shared_draft_weights(target, draft)
    assert draft.model_weights == expected
print("shared drafter retains target weight source without clobbering independent drafts")
PY

# Make worker_response_mq (one per worker) honour VLLM_MQ_MAX_CHUNK_BYTES_MB like rpc_broadcast_mq,
# so the env var bounds /dev/shm usage. Small-/dev/shm deployments must set it; see README.
COPY patches/mq-worker-response-honour-chunk-bytes.patch /tmp/mq-chunk.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/mq-chunk.patch; \
    else patch -p1 -d "$SITE" < /tmp/mq-chunk.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/mq-chunk.patch

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
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; SITE="$(dirname "$VLLM_DIR")"; \
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

# CUDA V4 C128 history uses one bounded ring per request, including draft lookahead.
# Rebased from vllm#58074 head f8bb0ba871415a779de33fadcb40ab2ba5acbc09;
# preserve the compressor method split and stream-capture guard.
# Python CuTe DSL/Triton JIT only. Requalify the carry on GPU after a base update.
COPY patches/58074-deepseek-v4-c128-circular-state-v033.patch /tmp/runtime-fix.patch
RUN set -eux; \
    test "$(sha256sum /tmp/runtime-fix.patch | cut -d' ' -f1)" = "ac31e8167e3410e64d9177a7b83e2fda2dffc4090bb53f35ba23a54e1c99bfae"; \
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/runtime-fix.patch; \
    else patch -p1 -d "$SITE" < /tmp/runtime-fix.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/runtime-fix.patch

# Count physical held blocks, not null-padded table positions, for SWA admission.
# vllm#60045 production-only diff, head 843db1462a066076d407bb628098140da4455be5.
# GPU serving remains unqualified.
COPY patches/60045-swa-admission-held-blocks.patch /tmp/offload-fix.patch
RUN set -eux; \
    test "$(sha256sum /tmp/offload-fix.patch | cut -d' ' -f1)" = "d2e0787a02594e4582ebdad78c92d928fd3f19dfc3bed6d98b192f6cc4c729a4"; \
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/offload-fix.patch; \
    else patch -p1 -d "$SITE" < /tmp/offload-fix.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/offload-fix.patch

# Order SimpleCPUOffload loads after queued compute reads of the destination blocks.
# Needed for async scheduling with the host KV tier (speculative steps still reading a block).
# vllm#60078 production-only diff, head 520208b140dbca471a76c752598b6fff2f8a7092.
# GPU serving remains unqualified.
COPY patches/60078-simple-offload-load-after-compute.patch /tmp/offload-fix.patch
RUN set -eux; \
    test "$(sha256sum /tmp/offload-fix.patch | cut -d' ' -f1)" = "6efcda2d6eead265dca1c14b1c27175809b5e523226ff15ee10860c4fcd624dc"; \
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/offload-fix.patch; \
    else patch -p1 -d "$SITE" < /tmp/offload-fix.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/offload-fix.patch

# Plugin: '--cpu-offload-params ... embed_tokens' puts DeepSeek V4's embedding table in pinned
# host memory through vLLM's UVA offloader (vLLM offloads layers and towers, never the embedding).
# Inactive unless that parameter is requested.
COPY plugins/pcai_embed_offload /tmp/pcai_embed_offload
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; SITE="$(dirname "$VLLM_DIR")"; \
    cp -r /tmp/pcai_embed_offload/. "$SITE/"; \
    rm -rf /tmp/pcai_embed_offload

# C1c: retain one earlier semantic SWA boundary without restoring block dropping.
COPY patches/dspark-swa-prefix-fallback-v033.patch /tmp/prefix-fallback.patch
RUN set -eux; \
    test "$(sha256sum /tmp/prefix-fallback.patch | cut -d' ' -f1)" = "83e748d8821e8a715a3fe3acd2e0cb0e43f6b75ed60853853a9e5a29e0400512"; \
    VLLM_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"; SITE="$(dirname "$VLLM_DIR")"; \
    patch --batch --forward --fuzz=0 -p1 -d "$SITE" < /tmp/prefix-fallback.patch; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/prefix-fallback.patch

# The base now skips the residual allocation for DSpark while retaining MTP.
RUN python3 - <<'PY'
from types import SimpleNamespace
from vllm.models.deepseek_v4.common.spec_decode import needs_mtp_hidden_buffer

for dspark, expected in ((True, False), (False, True)):
    config = SimpleNamespace(use_dspark=lambda: dspark, use_eagle=lambda: True,
                             uses_draft_model=lambda: False)
    assert needs_mtp_hidden_buffer(config) is expected
assert not needs_mtp_hidden_buffer(None)
print("stock DeepSeek DSpark/MTP residual allocation policy OK")
PY

# Humming 0.1.19: store fused-E8M0 MoE scales as 4-bit nibbles (two per byte). On SM90 with
# FP8 activations the rebased scales are 1..12, so this is lossless and halves resident
# scale memory. Opt-in with HUMMING_USE_PACKED_BS=1; off, kernels behave as before.
COPY patches/humming-packed-bs.patch /tmp/humming.patch
RUN set -eux; \
    test "$(sha256sum /tmp/humming.patch | cut -d' ' -f1)" = "623d2dd7cf22d30ca4be81e13fc296be1379bb8b6d6543a872b67e8d7142c798"; \
    HUMMING_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("humming").origin))')"; \
    test "$(python3 -c 'import importlib.metadata as m; print(m.version("humming-kernels"))')" = "0.1.19"; \
    patch --batch --forward --fuzz=0 -p1 -d "$HUMMING_DIR" < /tmp/humming.patch; \
    find "$HUMMING_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/humming.patch

# v0.34 preserves checkpoint-sized storage in replace_parameter, so nibble
# packing alone no longer refunds VRAM. Rebind only packed MoE scale storage,
# retaining Parameter identity/loader aliases; refuse unsupported direct reload.
COPY patches/humming-packed-bs-storage-v034.patch /tmp/humming-storage.patch
COPY tools/check_humming_packed_storage.py /tmp/check_humming_packed_storage.py
RUN set -eux; \
    test "$(sha256sum /tmp/humming-storage.patch | cut -d' ' -f1)" = "38206ccd55f93a0079bad588e67bc96c08bb9ed6f91e86566811f0ac3ae169f3"; \
    SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    patch --batch --forward --fuzz=0 -p1 -d "$SITE" < /tmp/humming-storage.patch; \
    python3 /tmp/check_humming_packed_storage.py; \
    rm -f /tmp/humming-storage.patch /tmp/check_humming_packed_storage.py

# Packed scales also require native launcher metadata and half-width validation.
COPY patches/humming-packed-bs-launcher.patch /tmp/humming-launcher.patch
COPY tools/build_humming_launcher.py /tmp/build_humming_launcher.py
RUN set -eux; \
    test "$(sha256sum /tmp/humming-launcher.patch | cut -d' ' -f1)" = "4c0ca83172f59a0953f2ad2d2b699a486167679dd384d8c99f65d9324d0e1ae0"; \
    HUMMING_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("humming").origin))')"; \
    patch --batch --forward --fuzz=0 -p1 -d "$HUMMING_DIR" < /tmp/humming-launcher.patch; \
    MAX_JOBS=1 OMP_NUM_THREADS=1 python3 /tmp/build_humming_launcher.py; \
    rm -f /tmp/humming-launcher.patch /tmp/build_humming_launcher.py
