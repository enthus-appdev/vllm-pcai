"""Compare EAQ's reconstructed dense weights with the original V4.1 weights on one GPU.

Runs the plugin's own load-time path (dense_modules + dequantize_stream with the
ExLlamaV3 reconstruction) over real EAQ tensors and checks each emitted weight
against the source checkpoint: FP8 E4M3 with UE8M0 32x32 block scales, or BF16.
A transposed matrix or misordered wo_a group gives near-zero similarity; correct
reconstruction leaves only quantization error.
"""
import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.environ.get("PCAI_EAQ_SRC", "/checks"))
from exl3_dense import dense_modules, dequantize_stream, tensor_storage  # noqa: E402

EAQ = Path(os.environ["EAQ_ROOT"])
ORIG = Path(os.environ["ORIG_ROOT"])
TARGETS = [
    "layers.0.attn.wq_a", "layers.0.attn.wkv", "layers.0.attn.wq_b", "layers.0.attn.wo_a",
    "layers.0.attn.wo_b", "layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w2",
    "layers.0.ffn.shared_experts.w3", "layers.21.attn.wq_b", "layers.21.attn.wo_a",
    "layers.21.ffn.shared_experts.w1", "layers.21.ffn.shared_experts.w2",
    "layers.1.engram.wkv", "layers.14.engram.wkv", "head",
]


def tensors(root, names):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = {}
    for name in names:
        by_file.setdefault(index[name], []).append(name)
    for file, wanted in by_file.items():
        with safe_open(str(root / file), framework="pt") as f:
            for name in wanted:
                yield name, f.get_tensor(name)


def original(target):
    names = [target + ".weight"]
    index = json.loads((ORIG / "model.safetensors.index.json").read_text())["weight_map"]
    if target + ".scale" in index:
        names.append(target + ".scale")
    got = dict(tensors(ORIG, names))
    w = got[target + ".weight"].cuda()
    if target + ".scale" not in got:
        return w.float()
    exponent = got[target + ".scale"].cuda().view(torch.uint8).float() - 127
    scale = torch.exp2(exponent).repeat_interleave(32, 0).repeat_interleave(32, 1)
    return w.float() * scale[: w.shape[0], : w.shape[1]]


def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    rel = ((a - b).norm() / b.norm()).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return rel, cos


def main():
    torch.cuda.set_device(0)
    config = json.loads((EAQ / "config.json").read_text())
    storage = tensor_storage(str(EAQ), config.get("quantization_config"))
    selected = {m: p for m, p in dense_modules(storage).items()
                if any(m == t or m.startswith(t + ".slice.") for t in TARGETS)}
    names = [f"{m}.{p}" for m, parts in selected.items() for p in parts]
    failures = []
    seen = set()
    for target, weight in dequantize_stream(tensors(EAQ, names), selected):
        module = target[: -len(".weight")]
        seen.add(module)
        bits = sorted({storage[m]["bits_per_weight"] for m in selected
                       if m == module or m.startswith(module + ".slice.")})
        ref = original(module)
        result = {"module": module, "shape": list(weight.shape), "bits": bits,
                  "dtype": str(weight.dtype)}
        if tuple(weight.shape) != tuple(ref.shape):
            result["error"] = f"shape {tuple(weight.shape)} != original {tuple(ref.shape)}"
            failures.append(module)
        else:
            result["rel_err"], result["cos"] = compare(weight, ref)
            if module.endswith(".wo_a"):
                groups = weight.shape[0] // 1024
                w, r = weight.view(groups, 1024, -1), ref.view(groups, 1024, -1)
                result["group_cos"] = [round(compare(w[g], r[g])[1], 4) for g in range(groups)]
                result["shifted_group_cos"] = round(compare(w[0], r[1])[1], 4)
            # 4-bit EXL3 is ~9% RMS error; a layout mistake is ~100% or worse.
            if result["cos"] < 0.97 or result["rel_err"] > 0.25:
                failures.append(module)
        print(json.dumps(result), flush=True)
        del weight, ref
        torch.cuda.empty_cache()
    missing = sorted(set(TARGETS) - seen)
    failures += missing
    print(json.dumps({"dense_verify": "FAIL" if failures else "PASS", "failures": failures,
                      "missing": missing}), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
