# PCAI can't mount volumes, so the enhanced chat template is baked in. Details: README.
#
# Base is a cu129 NIGHTLY (not a release) on purpose: it carries the streaming ParserEngine
# (vLLM #45413 + #45588), so DFlash's large multi-token drafts don't corrupt streaming tool
# calls in opencode — the legacy parser in the v0.23.0 release does not have this. The nightly
# still ships DFlash core (#43445) + qwen3_dflash, and is ahead of the v0.23.0 tag.
# Pinned by commit; bump deliberately (Dependabot won't track a nightly SHA tag).
FROM vllm/vllm-openai:cu129-nightly-6607a80dabfa03932515808895b016d2666b0a55

# Serve with: --chat-template /templates/qwen3.6-enhanced.jinja
COPY chat-template-fix/chat-template/*.jinja /templates/
