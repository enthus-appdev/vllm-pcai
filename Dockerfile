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

# DeepSeek V4 reasoning on the streaming parser engine (chunk-size-invariant), replacing the
# legacy single-token-delta reasoning parser that leaks </think> and raw reasoning into content
# when MTP / speculative decoding emits multi-token deltas (vllm#43933; the #45413 engine the
# nightly already carries was never wired to DeepSeek). Reasoning only — DeepSeek tool calls stay
# on the `deepseek_v4` TOOL parser (different token format). Serve unchanged: --reasoning-parser
# deepseek_v4. Pure-Python overlay (no recompile). Drop once an engine deepseek parser lands upstream.
COPY deepseek/deepseek_v4.py /tmp/dsv4/deepseek_v4.py
COPY deepseek/deepseek_v4_engine_reasoning_parser.py /tmp/dsv4/deepseek_v4_engine_reasoning_parser.py
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; \
    cp /tmp/dsv4/deepseek_v4.py "$VLLM_DIR/parser/deepseek_v4.py"; \
    cp /tmp/dsv4/deepseek_v4_engine_reasoning_parser.py "$VLLM_DIR/reasoning/deepseek_v4_engine_reasoning_parser.py"; \
    printf '\n\n# vllm-pcai overlay: route deepseek_v4 reasoning through the streaming parser engine.\nReasoningParserManager.register_lazy_module(\n    "deepseek_v4",\n    "vllm.reasoning.deepseek_v4_engine_reasoning_parser",\n    "DeepSeekV4ParserReasoningAdapter",\n)\n' >> "$VLLM_DIR/reasoning/__init__.py"; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -rf /tmp/dsv4

# Build-time tripwire: a broken overlay fails HERE (does NOT prove streaming correctness — needs GPU).
RUN python3 - <<'PY'
from vllm.reasoning import ReasoningParserManager
cls = ReasoningParserManager.get_reasoning_parser("deepseek_v4")
assert cls.__name__ == "DeepSeekV4ParserReasoningAdapter", cls
from vllm.parser.deepseek_v4 import deepseek_v4_config
assert deepseek_v4_config(thinking=True).initial_state.name == "REASONING"
assert deepseek_v4_config(thinking=False).initial_state.name == "CONTENT"
print("deepseek_v4 engine reasoning parser baked OK:", cls.__module__)
PY
