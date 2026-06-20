# PCAI can't mount volumes, so the chat templates are baked in. Details: README.
#
# Base is a cu129 NIGHTLY, not a release: needed for the engine streaming parsers (qwen3/gemma4
# tool-calling + gemma4 reasoning channels) and DFlash, none of which are in v0.23.0. Pinned by
# commit; bump deliberately (Dependabot won't track a nightly SHA tag).
FROM vllm/vllm-openai:cu129-nightly-b4c80ec0fd19c13a53d89623bb5957cd5cd631bb

# Qwen's enhanced template is baked (not upstream); Gemma uses vLLM's in-image template — serve with
#   --chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja
COPY chat-template-fix/chat-template/*.jinja /templates/

# vLLM PR #40898 (DFlash sliding-window attention) baked in — replaces the separate :pr-8-swa-dflash image.
# Pure-Python overlay (no CUDA recompile): the DFlash drafter's SWA layers get correct windowed attention
# → deep-position acceptance at high k (~5-10% wall-clock on Qwen). Inert for MTP models (Gemma). Drop this
# block once #40898 merges upstream and the nightly base carries it. ⚠️ A clean apply is NOT proof of
# correctness — an earlier hand-port applied fine yet gave 6% acceptance; validate the GPU per-position
# curve. Keep --attention-backend flash_attn (FlashInfer + #40898 crashes "Window left ...", vLLM #39995).
COPY patches/40898-on-nightly.patch /tmp/40898.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/40898.patch; \
    else patch -p1 -d "$SITE" < /tmp/40898.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/40898.patch

# Build-time tripwire: a broken/partial apply fails HERE (does NOT catch the 6% semantic bug — needs GPU).
RUN python3 - <<'PY'
import vllm
import vllm.v1.spec_decode.dflash, vllm.v1.spec_decode.llm_base_proposer
import vllm.model_executor.models.qwen3_dflash
import vllm.v1.worker.gpu_model_runner, vllm.v1.core.sched.scheduler, vllm.config.speculative
print("DFlash+SWA(#40898) overlay import OK:", vllm.__version__)
PY

# PCAI has no shell and no pod logs, so expose vLLM's collect_env over the serving port as
# GET /collect_env. No route auth of its own — the LB's per-isvc bearer is the only gate, same
# as every other route here.
COPY diag/collect_env_route.py /tmp/collect_env_route.py
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; \
    cat /tmp/collect_env_route.py >> "$VLLM_DIR/entrypoints/openai/api_server.py"; \
    rm -f /tmp/collect_env_route.py; \
    python3 -c "import inspect; import vllm.entrypoints.openai.api_server as m; from vllm.collect_env import get_pretty_env_info; assert hasattr(m, '_pcai_collect_env_wrapper'); assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in inspect.signature(m.build_app, follow_wrapped=False).parameters.values()), 'wrapped build_app must stay variadic'; print('collect_env route baked OK')"

# DeepSeek V4 + V3.2 parsers ported to the streaming parser engine — vendored from upstream
# vLLM PR #45877 (https://github.com/vllm-project/vllm/pull/45877), rebased onto this base
# (only registered_adapters.py needed re-resolving — this base predates the GLM parser). Replaces
# the legacy single-token-delta reasoning/tool parsers that leak </think> and mishandle tool calls
# under MTP / spec decoding (vLLM #43933). One state machine for reasoning AND DSML tool calls.
# Serve unchanged: --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4. Pure-Python
# overlay (no recompile). Plain #45877 now — the DeepSeek-only non-streaming BOS strip that used to
# live here is replaced by the general engine fix in 46225-on-nightly.patch (applied below).
# ⚠️ #45877 is an open draft — re-verify on bump; drop this block once it lands in the base nightly.
COPY patches/deepseek-v4-45877-on-nightly.patch /tmp/dsv4-45877.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/dsv4-45877.patch; \
    else patch -p1 -d "$SITE" < /tmp/dsv4-45877.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/dsv4-45877.patch

# Build-time tripwire: a broken/partial apply fails HERE (does NOT prove streaming correctness — needs GPU).
RUN python3 - <<'PY'
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
r = ReasoningParserManager.get_reasoning_parser("deepseek_v4")
t = ToolParserManager.get_tool_parser("deepseek_v4")
assert r.__name__ == "DeepSeekV4ParserReasoningAdapter", r
assert t.__name__ == "DeepSeekV4EngineToolParser", t
from vllm.parser.deepseek_v4 import deepseek_v4_config
import vllm.parser.deepseek_v32  # V3.2 sibling ported in the same PR
assert deepseek_v4_config(thinking=True).initial_state.name == "REASONING"
assert deepseek_v4_config(thinking=False).initial_state.name == "CONTENT"
print("deepseek_v4/v32 engine parsers baked OK:", r.__module__, "/", t.__module__)
PY

# vLLM PR #46225 (https://github.com/vllm-project/vllm/pull/46225) — the general upstream fix for
# the non-streaming special-token leak (the DeepSeek BOS leak is one instance). The engine scanner
# drops bos/eos/pad by token id, but non-streaming extract_reasoning fed an empty id list so the
# drop never fired. This threads the model output token ids through DelegatingParser.parse ->
# extract_reasoning -> ParserEngine._feed (only for engine_based_streaming parsers), so the scanner
# drops by id in non-streaming too — replacing the old DeepSeek-only string-strip. Pure-Python
# overlay. ⚠️ Open PR (changes-requested iteration) — re-verify on bump; drop once it lands in base.
COPY patches/46225-on-nightly.patch /tmp/46225.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/46225.patch; \
    else patch -p1 -d "$SITE" < /tmp/46225.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/46225.patch

# Build-time tripwire: confirms the wire-through applied (a broken apply fails HERE). Runtime BOS
# drop still needs the live model — re-run the thinking=false repro after deploy to confirm 0%.
RUN python3 - <<'PY'
import inspect
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.abstract_parser import DelegatingParser
assert "model_output_token_ids" in inspect.signature(ParserEngine.extract_reasoning).parameters
assert "model_output_token_ids" in inspect.signature(DelegatingParser.extract_reasoning).parameters
print("46225 non-streaming token-id wire-through baked OK")
PY
