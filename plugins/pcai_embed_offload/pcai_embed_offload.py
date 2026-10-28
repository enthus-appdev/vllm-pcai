"""vLLM plugin: let `--cpu-offload-params embed_tokens` place DeepSeek V4's embedding table in host memory.

vLLM routes decoder layers (`make_layers`) and multimodal towers through the weight offloader, but
never the token embedding. With `--offload-backend uva --cpu-offload-params ... embed_tokens`, this
plugin hands `DeepseekV4Model.embed_tokens` to the same UVA offloader right after construction, so
the table lives in pinned host memory and is read in place over PCIe (a few KB per token).

Inactive unless the configured offloader is UVA and its parameter filter names `embed_tokens`.
"""
import logging

logger = logging.getLogger("vllm.pcai_embed_offload")
SEGMENT = "embed_tokens"


def maybe_offload_embedding(model, offloader) -> bool:
    """Offload `model.embed_tokens` if `offloader` is UVA and asks for embed_tokens."""
    if not getattr(offloader, "supports_tower_offload", False):  # UVA is the only offloader with this
        return False
    if SEGMENT not in (getattr(offloader, "cpu_offload_params", None) or set()):
        return False
    embed = getattr(model, SEGMENT, None)
    if getattr(embed, "weight", None) is None:
        return False
    offloader.wrap_modules(iter([embed]), prefix=SEGMENT)
    done = bool(getattr(embed.weight, "_vllm_is_uva_offloaded", False))
    logger.info("pcai_embed_offload: embed_tokens %s %s offloaded=%s",
                tuple(embed.weight.shape), embed.weight.dtype, done)
    return done


def register():
    try:
        from vllm.models.deepseek_v4.nvidia import model as v4
    except Exception as exc:  # non-CUDA platforms or a refactor: stay inert
        logger.debug("pcai_embed_offload: DeepSeek V4 model not importable: %s", exc)
        return
    cls = v4.DeepseekV4Model
    if getattr(cls, "_pcai_embed_offload", False):
        return
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        from vllm.model_executor.offloader import get_offloader
        maybe_offload_embedding(self, get_offloader())

    cls.__init__ = __init__
    cls._pcai_embed_offload = True
