# PCAI can't mount volumes, so the chat templates are baked in. Details: README.
#
# Base is a cu129 NIGHTLY, not a release: needed for the engine streaming parsers (qwen3/gemma4
# tool-calling + gemma4 reasoning channels), DFlash, and the DeepSeek-V4 DSpark prereqs (warmup
# modules + sparse_swa), none of which are in v0.23.0. Pinned by commit; bump deliberately
# (Dependabot won't track a nightly SHA tag).
FROM vllm/vllm-openai:cu129-nightly-09663abde0f50944a8d5ea30120666024b503faa

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

# DeepSeek V4 + V3.2 parsers ported to the streaming parser engine — vendored from upstream
# vLLM PR #45877 (https://github.com/vllm-project/vllm/pull/45877), rebased onto the base nightly.
# Replaces the legacy single-token-delta reasoning/tool parsers that leak </think> and mishandle tool calls
# under MTP / spec decoding (vLLM #43933). One state machine for reasoning AND DSML tool calls.
# Serve unchanged: --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4. Pure-Python
# overlay (no recompile). Plain #45877: the non-streaming special-token-leak fix formerly vendored
# here as #46225 now ships IN the base nightly (upstream #46875, merged 2026-06-30) — not carried here.
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

# Make the deepseek_v4/v32 tokenizer-mode encoders honor add_generation_prompt +
# continue_final_message (they otherwise silently ignore both). Rationale + evidence in the PR.
# Vendored overlay; drop once upstreamed into the base nightly.
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

# DeepSeek-V4 DSpark spec decode (vLLM PR #46995, https://github.com/vllm-project/vllm/pull/46995) is
# now IN the base nightly (merged 2026-07-01) — vendored patch dropped. Serve unchanged:
# --speculative-config {"method":"dspark","num_speculative_tokens":5} + cudagraphs (no --enforce-eager);
# needs the DSpark checkpoint deepseek-ai/DeepSeek-V4-Flash-DSpark. Leave draft_sample_method default
# (probabilistic degrades into loops).
# Sanity tripwire: a base bump that silently drops DSpark fails HERE (does NOT prove the Hopper kernels
# lower or that acceptance is good — both need GPU + the checkpoint).
RUN python3 - <<'PY'
import vllm
from vllm.config.speculative import SpeculativeMethod
assert "dspark" in repr(SpeculativeMethod), repr(SpeculativeMethod)
import vllm.models.deepseek_v4.nvidia.dspark  # noqa: F401  (model-side draft module)
import vllm.v1.worker.gpu.spec_decode.dspark.speculator  # noqa: F401  (worker-side speculator)
print("DSpark(#46995) native in base OK:", vllm.__version__)
PY
