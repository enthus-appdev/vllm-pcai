"""Load the EXL3-quantized dense layers of a mixed EXL3 pack as BF16 weights.

Packs such as EAQ quantize attention, shared experts, Engram projections and the
output head with EXL3, not only the routed experts. vLLM's DeepSeek V4.1 model has
no EXL3 path for those layers, so they are reconstructed once at load time with
ExLlamaV3's own reference reconstruction and handed to vLLM as ordinary BF16
``<module>.weight`` tensors. Routed experts are untouched and stay on the EXL3
kernels.

``wo_a`` is stored as one EXL3 tensor per output group (``wo_a.slice.<g>``); the
groups are the consecutive row blocks of vLLM's fused ``wo_a`` weight, as in
ExLlamaV3's own DeepSeek V4 module.
"""
from __future__ import annotations

import re

PARTS = ("trellis", "suh", "svh", "mul1", "mcg")
MUL1_MULTIPLIER = 0x83DCD12D
MCG_MULTIPLIER = 0xCBAC1FED
_ROUTED = re.compile(r"(?:^|\.)experts\.\d+(?:\.|$)")
_SLICE = re.compile(r"^(?P<base>.+)\.slice\.(?P<group>\d+)$")


def dense_modules(tensor_storage):
    """EXL3 modules to reconstruct: not routed experts, not draft (``mtp.``) layers.

    Returns {module: frozenset(part names)} from the pack's tensor_storage ledger.
    """
    modules = {}
    for module, spec in (tensor_storage or {}).items():
        if spec.get("quant_format") != "exl3":
            continue
        if _ROUTED.search(module) or module.startswith("mtp."):
            continue
        parts = frozenset(name.rsplit(".", 1)[1] for name in spec.get("stored_tensors", {}))
        if not {"trellis", "suh", "svh"} <= parts or not parts <= set(PARTS):
            raise ValueError(f"unexpected EXL3 tensor set for {module}: {sorted(parts)}")
        modules[module] = parts
    return modules


def tensor_storage(model_dir, quant_config):
    """The per-tensor ledger: embedded in config.json, else the sidecar file."""
    import json
    import os

    ledger = (quant_config or {}).get("tensor_storage")
    if isinstance(ledger, dict) and ledger:
        return ledger
    path = os.path.join(model_dir, "quantization_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        sidecar = json.load(f)
    return sidecar.get("tensor_storage") or sidecar.get("quantization_config", {}).get("tensor_storage") or {}


def _group_targets(modules):
    """Map fused targets to their ordered slices; plain modules map to themselves."""
    groups = {}
    for module in modules:
        match = _SLICE.match(module)
        if match:
            groups.setdefault(match["base"], {})[int(match["group"])] = module
    targets = {}
    for base, slices in groups.items():
        order = sorted(slices)
        if order != list(range(len(order))):
            raise ValueError(f"{base}: non-contiguous slices {order}")
        targets[base] = [slices[g] for g in order]
    for module in modules:
        if not _SLICE.match(module):
            targets[module] = [module]
    return targets


def reconstruct_exl3(parts):
    """Full (out_features, in_features) BF16 weight on the current CUDA device."""
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3

    device = torch.device("cuda", torch.cuda.current_device())
    trellis = parts["trellis"].to(device)
    suh = parts["suh"].to(device, torch.float16)
    svh = parts["svh"].to(device, torch.float16)
    in_features, out_features = suh.numel(), svh.numel()
    if tuple(trellis.shape[:2]) != (in_features // 16, out_features // 16):
        raise ValueError(f"trellis {tuple(trellis.shape)} does not match {in_features}x{out_features}")
    codebooks = {}
    for name, constant in (("mul1", MUL1_MULTIPLIER), ("mcg", MCG_MULTIPLIER)):
        if name in parts:
            if int(parts[name].reshape(-1)[0].item()) & 0xFFFFFFFF != constant:
                raise ValueError(f"unsupported {name} codebook multiplier")
            codebooks[name] = parts[name].to(device)
    linear = LinearEXL3(None, in_features, out_features, suh=suh, svh=svh,
                        trellis=trellis, **codebooks)
    weight = linear.get_weight_tensor()  # (in_features, out_features), fp16
    if not torch.isfinite(weight).all():
        raise ValueError("non-finite reconstructed EXL3 weight")
    return weight.t().contiguous().to(torch.bfloat16)


def dequantize_stream(weights, modules, reconstruct=reconstruct_exl3):
    """Replace the EXL3 parts of ``modules`` in a (name, tensor) stream with BF16 weights.

    Everything else passes through unchanged and in order. A fused target is emitted
    as soon as all of its slices are complete; leftovers are an error.
    """
    import torch

    targets = _group_targets(modules)
    owner = {module: target for target, members in targets.items() for module in members}
    pending = {}
    done = {}
    for name, tensor in weights:
        module, _, part = name.rpartition(".")
        target = owner.get(module)
        if target is None or part not in PARTS:
            yield name, tensor
            continue
        pending.setdefault(module, {})[part] = tensor
        if set(pending[module]) != modules[module]:
            continue
        done[module] = reconstruct(pending.pop(module))
        members = targets[target]
        if all(member in done for member in members):
            pieces = [done.pop(member) for member in members]
            yield target + ".weight", pieces[0] if len(pieces) == 1 else torch.cat(pieces, 0)
    missing = sorted(set(pending) | set(done))
    if missing:
        raise ValueError(f"incomplete EXL3 modules after loading: {missing[:8]}")


def install():
    """Route dense layers to BF16 and dequantize their EXL3 tensors while loading."""
    from vllm.logger import init_logger
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm_exl3 import exl3 as plugin

    logger = init_logger("vllm." + __name__)
    if getattr(DefaultModelLoader, "_pcai_exl3_dense", False):
        return
    # Dense linears and the LM head become ordinary unquantized (BF16) layers.
    plugin.Exl3Config._matches_non_routed_exl3 = lambda self, prefix: False

    original = DefaultModelLoader.get_all_weights

    def get_all_weights(self, model_config, model):
        weights = original(self, model_config, model)
        quant = getattr(model_config.hf_config, "quantization_config", None) or {}
        if quant.get("quant_method") != "exl3":
            return weights
        modules = dense_modules(tensor_storage(model_config.model, quant))
        if not modules:
            return weights
        logger.info("Reconstructing %d EXL3 dense modules as BF16 at load time", len(modules))
        return dequantize_stream(weights, modules)

    DefaultModelLoader.get_all_weights = get_all_weights
    DefaultModelLoader._pcai_exl3_dense = True
