# Custom vLLM image for HPE PCAI (which cannot mount volumes).
#
# The stock vLLM chat templates are ALREADY inside the base image at
# /vllm-workspace/examples/*.jinja (vLLM's own Dockerfile does `COPY examples examples`),
# so there is no need to add them.
#
# This image only adds the ENHANCED Qwen3.5/3.6 chat templates (allanchan339 fix),
# which are NOT in the base image. They harden the 27B template: proper </think>
# handling before tool calls, hidden historical reasoning, and XML tool-call
# formatting that avoids premature stop tokens.
#
# Reference at runtime, e.g.:  --chat-template /templates/qwen3.6-enhanced.jinja
FROM vllm/vllm-openai:v0.23.0-cu129-ubuntu2404

COPY chat-template-fix/chat-template/*.jinja /templates/
