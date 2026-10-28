"""PCAI EAQ plugin. Each part is inactive unless its environment variable is set.

PCAI_ENGRAM_DIR      read DeepSeek V4.1 Engram rows from staged files on disk.
PCAI_EXL3_DENSE=bf16 load EXL3-quantized dense layers of a mixed pack as BF16.
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
