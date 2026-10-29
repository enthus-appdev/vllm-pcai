"""PCAI EAQ plugin. Each part is inactive unless its environment variable is set.

PCAI_ENGRAM_DIR      read DeepSeek V4.1 Engram rows from staged files on disk.
PCAI_EXL3_DENSE=bf16 load EXL3-quantized dense layers of a mixed pack as BF16.
PCAI_EXL3_DIAG=1     check every fused-MoE expert pointer after load (diagnostics).

The fused-MoE physical K fix is always applied when vllm-exl3 is installed.
"""


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
