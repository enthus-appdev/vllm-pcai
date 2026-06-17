# PCAI can't mount volumes, so the chat templates are baked in. Details: README.
#
# Base is a cu129 NIGHTLY (not a release) on purpose: it carries the streaming ParserEngine
# (vLLM #45413 + #45588). For Qwen that keeps DFlash's large multi-token drafts from corrupting
# streaming tool calls in opencode; #45588 is also the engine-based gemma4 parser — the fix for
# Gemma's streaming tool-call leak. The v0.23.0 release has only the legacy parsers. The nightly
# still ships DFlash core (#43445) + qwen3_dflash, and is ahead of the v0.23.0 tag.
# Pinned by commit; bump deliberately (Dependabot won't track a nightly SHA tag).
FROM vllm/vllm-openai:cu129-nightly-4c626633159887b0f2c962058c17c78f1434556d

# All chat templates baked to /templates/: Qwen enhanced (submodule) + Gemma PR#118 (gemma-template/).
# Serve with e.g. --chat-template /templates/qwen3.6-enhanced.jinja  or  /templates/gemma-4-31b-pr118.jinja
COPY chat-template-fix/chat-template/*.jinja /templates/
COPY gemma-template/*.jinja /templates/

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
