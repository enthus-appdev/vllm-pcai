# Custom vLLM image with chat templates baked in.
#
# HPE PCAI cannot mount volumes through its UI, so the chat templates must
# live INSIDE the image. They are referenced at runtime via
#   --chat-template /templates/<file>.jinja
#
# The templates come from the pinned vLLM submodule (./vllm @ v0.22.0), so the
# base image tag and the templates always match. Bump the submodule to upgrade.
FROM vllm/vllm-openai:v0.22.0

# Copy the chat templates from the pinned vLLM submodule into the image.
COPY vllm/examples/*.jinja /templates/

# Entrypoint is inherited from the base image (`vllm serve ...`).
