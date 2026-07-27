# PCAI can't mount volumes, so the chat templates are baked in. Details: README.
#
# v0.26.0 is the first RELEASE carrying everything we previously needed nightlies for: the engine
# streaming parsers (qwen3/gemma4/deepseek_v4), DFlash incl. hybrid SWA+full drafters (vllm#47914 —
# z-lab/Qwen3.6-27B-DFlash is 4-of-5 sliding, so this is required, not optional), and DeepSeek-V4
# DSpark. Before bumping, verify the target tag is a superset of this one — vLLM cuts release
# branches, so a later tag can MISS commits present here (compare/<sha>...<tag> must say "ahead").
FROM vllm/vllm-openai:v0.26.0

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

# Without this, a reply that never emits </think> leaves the EOS token in reasoning_content
# (generation ends in the parser's REASONING state). vllm#48748 merged 2026-07-22 but landed
# AFTER the v0.26.0 branch cut, so the release tag does not have it. Drop at v0.27.0 — the
# apply below will fail loudly once the base carries it.
COPY patches/48748-eos-reasoning-leak-on-v0.26.0.patch /tmp/dsv4-eos.patch
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; SITE="$(dirname "$VLLM_DIR")"; \
    if command -v git >/dev/null 2>&1; then git -C "$SITE" apply -p1 --verbose /tmp/dsv4-eos.patch; \
    else patch -p1 -d "$SITE" < /tmp/dsv4-eos.patch; fi; \
    find "$VLLM_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    rm -f /tmp/dsv4-eos.patch

# Tripwire asserts BOTH directions: the symbols must still exist (so an upstream rename fails
# here instead of passing vacuously), and the DROP_TERMINAL branch must no longer gate on
# skip_tool_parsing. Source-level, not behavioral — driving the parser needs vLLM's test
# fixtures (MockTokenizer et al), which are not in the runtime image.
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
print("EOS-in-reasoning fix (vllm#48748) applied OK")
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
