"""Fused EXL3 MoE state: physical K fix, plus opt-in pointer checks.

K fix: for a layer whose experts share one K, vllm-exl3 sets the fused launch's
``_exl3_k_tuple`` from the config-level ``bits`` (default 4 when a pack, such as
EAQ, declares no global ``bits``). Its physical-K compat corrects ``_exl3_k`` but
not the tuple, so the kernel walks K3 trellises with a K4 stride and reads past
each expert, an illegal access once the experts live in host memory. After the
state is built, the tuple is set from the loaded trellis widths; ``_exl3_k`` too when
all three projections share one K (it is only meaningful for uniform layers).

Checks (PCAI_EXL3_DIAG=1): every trellis/suh/svh pointer is looked up with the
CUDA driver: memory type and the allocation it falls in. Pageable host pointers
or tensors overrunning their allocation are logged as errors with layer and
expert. One summary line per layer is logged.
"""
from __future__ import annotations

import ctypes

from .compat import check_vllm, require

CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12
MEMORY_TYPES = {1: "host", 2: "device", 3: "array", 4: "unified"}
_cuda = None


def _driver():
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
        _cuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        _cuda.cuPointerGetAttribute.restype = ctypes.c_int
    return _cuda


def pointer_info(ptr: int):
    """(memory type, range start, range size).

    Memory type is None for memory CUDA does not know; start and size are None when the
    driver cannot report the allocation range.
    """
    cuda = _driver()
    kind = ctypes.c_uint32()
    start = ctypes.c_uint64()
    size = ctypes.c_size_t()
    if cuda.cuPointerGetAttribute(ctypes.byref(kind), CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr):
        return None, None, None
    if (cuda.cuPointerGetAttribute(ctypes.byref(start), CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, ptr)
            or cuda.cuPointerGetAttribute(ctypes.byref(size), CU_POINTER_ATTRIBUTE_RANGE_SIZE, ptr)):
        return MEMORY_TYPES.get(kind.value, str(kind.value)), None, None
    return MEMORY_TYPES.get(kind.value, str(kind.value)), start.value, size.value


def check_layer(layer, inners):
    """Validate every expert tensor the fused kernel will dereference; returns a summary."""
    kinds = {}
    problems = []
    for e, pack in enumerate(inners):
        for proj in ("gate", "up", "down"):
            linear = pack.get(proj)
            if linear is None:
                continue
            for attr in ("trellis", "suh", "svh"):
                t = getattr(linear, attr)
                ptr, nbytes = int(t.data_ptr()), int(t.numel()) * int(t.element_size())
                kind, start, size = pointer_info(ptr)
                kinds[(attr, kind)] = kinds.get((attr, kind), 0) + 1
                if kind is None:
                    problems.append(f"expert {e} {proj}.{attr}: pointer unknown to CUDA (pageable host?)")
                elif start is None:
                    problems.append(f"expert {e} {proj}.{attr}: allocation range unavailable")
                elif ptr + nbytes > start + size:
                    problems.append(f"expert {e} {proj}.{attr}: {nbytes} bytes overrun allocation by "
                                    f"{ptr + nbytes - start - size}")
                if not t.is_contiguous():
                    problems.append(f"expert {e} {proj}.{attr}: not contiguous, shape {tuple(t.shape)}")
    first = inners[0] if inners else {}
    summary = {
        "layer": getattr(layer, "layer_name", None) or getattr(layer, "prefix", "?"),
        "experts": len(inners),
        "k": getattr(layer, "_exl3_k", None), "k_tuple": getattr(layer, "_exl3_k_tuple", None),
        "mixedk": getattr(layer, "_exl3_mixedk_unified", None),
        "codebook_flags": getattr(layer, "_exl3_codebook_flags", None),
        "trellis_shapes": {p: tuple(first[p].trellis.shape) for p in ("gate", "up", "down") if first.get(p) is not None},
        "temps": [tuple(t.shape) for t in (getattr(layer, "_exl3_fused_temps", None) or ())],
        "memory": {f"{a}:{k}": n for (a, k), n in sorted(kinds.items(), key=str)},
    }
    return summary, problems


def physical_k_tuple(inners):
    """The (gate, up, down) K every expert shares, or None if they differ."""
    tuples = set()
    for pack in inners:
        up = pack["up"]
        gate = pack.get("gate") or up
        tuples.add(tuple(int(getattr(m, "trellis").shape[-1]) // 16 for m in (gate, up, pack["down"])))
    return next(iter(tuples)) if len(tuples) == 1 else None


def install(diag=False):
    from vllm.logger import init_logger
    from vllm_exl3 import exl3 as plugin

    logger = init_logger("vllm." + __name__)
    original = getattr(plugin, "build_exl3_fused_state", None)
    if getattr(original, "_pcai_wrapped", False):
        return
    check_vllm()
    require(plugin, "build_exl3_fused_state", ("layer", "inners"))

    def build_exl3_fused_state(layer, inners):
        result = original(layer, inners)
        if not getattr(layer, "_exl3_mixedk_unified", False):
            kt = physical_k_tuple(inners)
            if kt is not None and getattr(layer, "_exl3_k_tuple", None) != kt:
                logger.info_once("PCAI EXL3: fused K tuple %s -> physical %s",
                                 getattr(layer, "_exl3_k_tuple", None), kt)
                layer._exl3_k_tuple = kt
                if kt[0] == kt[1] == kt[2]:
                    layer._exl3_k = kt[0]
        if diag:
            summary, problems = check_layer(layer, inners)
            logger.info("PCAI_EXL3_DIAG %s", summary)
            if problems:
                logger.error("PCAI_EXL3_DIAG %s: %d invalid expert pointers, first: %s",
                             summary["layer"], len(problems), problems[:5])
        return result

    build_exl3_fused_state._pcai_wrapped = True
    plugin.build_exl3_fused_state = build_exl3_fused_state
