"""PCAI EAQ plugin. Each part is inactive unless its environment variable is set.

PCAI_ENGRAM_DIR      read DeepSeek V4.1 Engram rows from staged files on disk.
PCAI_EXL3_DENSE=bf16 load EXL3-quantized dense layers of a mixed pack as BF16.
PCAI_EXL3_DIAG=1     check every fused-MoE expert pointer after load (diagnostics).

The fused-MoE physical K fix is always applied when vllm-exl3 is installed.
Every patch first checks that the vLLM/vllm-exl3 internals it replaces are unchanged
(pcai_eaq.compat); ``check_compat()`` runs those checks without patching anything.
"""


def check_compat():
    """Verify every patched internal exists with the expected signature; patches nothing."""
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.models.deepseek_v41.common import engram
    from vllm_exl3 import exl3

    from .compat import check_vllm, require
    from .runtime import ENGRAM_INIT

    check_vllm()
    require(engram.Engram, "__init__", ENGRAM_INIT)
    require(engram.Engram, "_create_embedding", ("self", "layout", "layer_hash_index"))
    require(exl3, "build_exl3_fused_state", ("layer", "inners"))
    require(exl3.Exl3Config, "_matches_non_routed_exl3")
    require(DefaultModelLoader, "get_all_weights", ("self", "model_config", "model"))


def register():
    import os

    if os.environ.get("PCAI_ENGRAM_DIR"):
        from .runtime import install
        install()
    dense = os.environ.get("PCAI_EXL3_DENSE", "")
    if dense:
        if dense != "bf16":
            raise ValueError(f"PCAI_EXL3_DENSE must be 'bf16', not {dense!r}")
        from .exl3_dense import install
        install()
    try:
        import vllm_exl3  # noqa: F401
    except ImportError:
        return
    from .exl3_moe import install as install_moe
    install_moe(diag=os.environ.get("PCAI_EXL3_DIAG") == "1")
