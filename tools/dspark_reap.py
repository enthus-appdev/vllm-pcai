#!/usr/bin/env python3
"""Score DeepSeek V4 DSpark drafter experts from a router capture (REAP).

Reads the files written by VLLM_PCAI_DSPARK_CAPTURE_DIR (per draft layer: MoE
input hidden states, input ids, selected experts, normalized gate weights) and
the checkpoint's draft experts, and computes REAP mean saliency per expert:

    S_j = mean over tokens routed to j of  g_j(x) * ||f_j(x)||_2

f_j is the expert's clamped-SwiGLU FFN on the captured input, with weights
dequantized exactly as vLLM does. Writes saliency.json and kept-<N>.json files
({"layers": {"0": [ids], ...}}) for VLLM_PCAI_DSPARK_KEPT_EXPERTS, and prints
each kept file to stdout.

As a capture check it re-derives routing from the captured inputs and the
checkpoint router, and reports agreement with the experts the server chose.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Checkpoint:
    def __init__(self, model_dir):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        self.cfg = cfg.get("text_config", cfg) | {
            k: v for k, v in cfg.items() if k not in ("text_config", "vision_config")
        }
        self.handles = {}

    def get(self, name):
        shard = self.map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.dir, shard), "pt")
        return self.handles[shard].get_tensor(name)


def dequant_mxfp4(packed, scale):
    from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
        _dequant_mxfp4_ue8m0_gran32,
    )

    return _dequant_mxfp4_ue8m0_gran32(packed, scale).float()


def load_capture(directory, layer):
    files = sorted(glob.glob(os.path.join(directory, f"L{layer}-*.pt")))
    if not files:
        raise SystemExit(f"no capture files for draft layer {layer} in {directory}")
    parts = [torch.load(f, map_location="cpu") for f in files]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}, len(files)


def router_agreement(cap, ckpt, layer, device, image_lo):
    """Fraction of captured top-k ids re-derived from checkpoint router weights."""
    w = ckpt.get(f"mtp.{layer}.ffn.gate.weight").to(device, torch.float32)
    bias = ckpt.get(f"mtp.{layer}.ffn.gate.bias").to(device, torch.float32)
    bias_vl_name = f"mtp.{layer}.ffn.gate.bias_vl"
    bias_vl = ckpt.get(bias_vl_name).to(device, torch.float32) if bias_vl_name in ckpt.map else None
    hits = total = 0
    top_k = cap["topk_ids"].shape[1]
    for start in range(0, cap["hidden_states"].shape[0], 4096):
        x = cap["hidden_states"][start : start + 4096].to(device, torch.float32)
        scores = torch.sqrt(F.softplus(x @ w.T))
        sel_bias = bias.expand_as(scores)
        if bias_vl is not None and "input_ids" in cap:
            ids = cap["input_ids"][start : start + 4096].to(device)
            is_image = ((ids >= image_lo) & (ids < image_lo + 5)).unsqueeze(1)
            sel_bias = torch.where(is_image, bias_vl.expand_as(scores), sel_bias)
        mine = torch.topk(scores + sel_bias, top_k, dim=1).indices.sort(dim=1).values.cpu()
        theirs = cap["topk_ids"][start : start + 4096].long().sort(dim=1).values
        hits += (mine == theirs).all(dim=1).sum().item()
        total += mine.shape[0]
    return hits / max(total, 1)


def score_layer(cap, ckpt, layer, num_experts, limit, device):
    ids = cap["topk_ids"].long()
    gates = cap["topk_weights"].float()
    hidden = cap["hidden_states"]
    sal_sum = [0.0] * num_experts
    count = [0] * num_experts
    for j in range(num_experts):
        tok, slot = (ids == j).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        prefix = f"mtp.{layer}.ffn.experts.{j}"
        w1, w2, w3 = (
            dequant_mxfp4(ckpt.get(f"{prefix}.{w}.weight"), ckpt.get(f"{prefix}.{w}.scale")).to(device)
            for w in ("w1", "w2", "w3")
        )
        x = hidden[tok].to(device, torch.float32)
        gate = torch.clamp(x @ w1.T, max=limit)
        up = torch.clamp(x @ w3.T, min=-limit, max=limit)
        y = (F.silu(gate) * up) @ w2.T
        sal = gates[tok, slot].to(device) * torch.linalg.vector_norm(y, dim=1)
        if not torch.isfinite(sal).all():
            raise SystemExit(f"non-finite saliency for layer {layer} expert {j}")
        sal_sum[j] = float(sal.sum())
        count[j] = int(tok.numel())
    mean = [s / c if c else 0.0 for s, c in zip(sal_sum, count)]
    return mean, sal_sum, count


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--keep", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-agreement", type=float, default=0.98)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from vllm.models.deepseek_v4.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ckpt = Checkpoint(args.model)
    num_experts = ckpt.cfg["n_routed_experts"]
    limit = ckpt.cfg.get("swiglu_limit") or math.inf
    num_layers = 1 + max(int(m.group(1)) for k in ckpt.map if (m := re.match(r"mtp\.(\d+)\.ffn\.gate\.weight$", k)))
    os.makedirs(args.out, exist_ok=True)

    report = {"model": args.model, "num_experts": num_experts, "layers": {}}
    for layer in range(num_layers):
        cap, files = load_capture(args.capture, layer)
        tokens = cap["topk_ids"].shape[0]
        agree = router_agreement(cap, ckpt, layer, args.device, IMAGE_SENTINEL_BASE_ID)
        log(f"layer {layer}: {tokens} tokens from {files} files, router agreement {agree:.4f}")
        if agree < args.min_agreement:
            raise SystemExit(
                f"layer {layer}: captured routing disagrees with the checkpoint router "
                f"({agree:.4f} < {args.min_agreement}); capture does not match this checkpoint"
            )
        mean, sal_sum, count = score_layer(cap, ckpt, layer, num_experts, limit, args.device)
        used = sum(c > 0 for c in count)
        log(f"layer {layer}: {used}/{num_experts} experts routed at least once")
        report["layers"][str(layer)] = {
            "tokens": tokens, "router_agreement": agree,
            "mean": mean, "sum_saliency": sal_sum, "count": count,
        }

    with open(os.path.join(args.out, "saliency.json"), "w") as f:
        json.dump(report, f)
    for n in args.keep:
        kept = {}
        for layer, r in report["layers"].items():
            order = sorted(range(num_experts), key=lambda j: (-r["mean"][j], -r["count"][j], j))
            kept[layer] = sorted(order[:n])
        doc = {"layers": kept, "selection": f"REAP mean saliency, top {n} of {num_experts}",
               "capture_tokens": {k: v["tokens"] for k, v in report["layers"].items()}}
        path = os.path.join(args.out, f"kept-{n}.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        print(f"=== {path}")
        print(json.dumps(doc))


if __name__ == "__main__":
    main()
