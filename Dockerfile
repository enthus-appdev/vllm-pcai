# PCAI can't mount volumes, so the enhanced chat template is baked in. Details: README.
FROM vllm/vllm-openai:v0.23.0-cu129-ubuntu2404

# Serve with: --chat-template /templates/qwen3.6-enhanced.jinja
COPY chat-template-fix/chat-template/*.jinja /templates/
