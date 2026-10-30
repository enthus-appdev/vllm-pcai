"""Can the ExLlamaV3 fused MoE kernels read expert trellises from pinned host memory (UVA)?

Runs ext.exl3_moe (and exl3_moe_mixedk) at the EAQ TP2 expert shape twice per case:
experts in device memory, then the same bytes in pinned host memory addressed through
UVA device pointers. Each run is its own process so an illegal access cannot hide the
next result. Prints one JSON line per run and a verdict.
"""
import json
import os
import subprocess
import sys

H, I, E, TOPK, K = 5120, 1152, 384, 6, 3


def run(case, placement, tokens, routing):
    import torch
    sys.path.insert(0, "/opt/pcai-sources/exllamav3/tests")
    from exllamav3.ext import exllamav3_ext as ext
    from testlib.exl3 import rand_experts
    from testlib.moe import moe_buffers, routing_layout

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    gen = torch.Generator().manual_seed(0)
    ex = rand_experts(E, H, I, K, gen)
    keep = []

    def place(t):
        if placement == "device":
            return t.to(dev)
        host = t.contiguous().pin_memory()
        keep.append(host)
        return host  # pinned + UVA: the host pointer is a valid device address

    ex = {p: [tuple(place(t) for t in e) for e in ex[p]] for p in ex}
    tabs = {p: tuple(torch.tensor([e[i].data_ptr() for e in ex[p]], dtype=torch.long, device=dev)
                     for i in range(3)) for p in ex}
    y = (torch.randn((tokens, H), generator=gen) * 0.05).half().to(dev)
    if routing == "uniform":
        sel = torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(tokens)]).to(dev)
    else:  # every token picks the same experts, as vLLM's dummy warmup can
        sel = torch.arange(TOPK).repeat(tokens, 1).to(dev)
    w = torch.full((tokens, TOPK), 1.0 / TOPK, dtype=torch.half, device=dev)
    lay = routing_layout(sel, w, E)
    cap = int(lay.expert_count[:E].max().item())
    bufs = moe_buffers(H, I, cap, dev)
    out = torch.zeros((tokens, H), dtype=torch.float, device=dev)
    flags = (False, True) * 3  # mul1 codebook for gate, up, down
    if case == "exl3_moe":
        ext.exl3_moe(y, out, lay.expert_count, lay.token_sorted, lay.weight_sorted, *bufs, 0, K, K, K,
                     *tabs["g"], *tabs["u"], *tabs["d"], *flags, 0.0, -1, None, None, 1, cap, 16)
    else:
        karr = torch.full((E,), K, dtype=torch.int32, device=dev)
        ext.exl3_moe_mixedk(y, out, lay.expert_count, lay.token_sorted, lay.weight_sorted, *bufs, 0,
                            karr, karr, karr, *tabs["g"], *tabs["u"], *tabs["d"], *flags, 0.0, -1,
                            None, None, 1, cap, 16)
    torch.cuda.synchronize()
    return {"finite": bool(torch.isfinite(out).all().item()), "absmax": float(out.abs().max().item()),
            "checksum": float(out.double().sum().item())}


def close(a, b, rel=1e-6):
    """Host and device runs agree: same finiteness, sums within rel (kernels may reduce in a different order)."""
    a, b = json.loads(a), json.loads(b)
    if a["finite"] != b["finite"]:
        return False
    return all(abs(a[k] - b[k]) <= rel * max(abs(a[k]), abs(b[k]), 1e-12) for k in ("absmax", "checksum"))


def main():
    if len(sys.argv) > 1:
        case, placement, tokens, routing = sys.argv[1:5]
        print(json.dumps(run(case, placement, int(tokens), routing)), flush=True)
        return
    results = {}
    for case in ("exl3_moe", "exl3_moe_mixedk"):
        for tokens, routing in ((16, "uniform"), (512, "uniform"), (512, "same")):
            for placement in ("device", "host"):
                key = f"{case} tokens={tokens} routing={routing} {placement}"
                env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
                p = subprocess.run([sys.executable, __file__, case, placement, str(tokens), routing],
                                   capture_output=True, text=True, timeout=600, env=env)
                ok = p.returncode == 0
                detail = p.stdout.strip().splitlines()[-1] if ok else (p.stderr.strip().splitlines() or ["?"])[-1]
                results[key] = (ok, detail)
                print(json.dumps({"run": key, "ok": ok, "result": detail[:400]}), flush=True)
    same = all(close(results[k][1], results[k.replace(" host", " device")][1])
               for k in results if k.endswith(" host") and results[k][0] and results[k.replace(" host", " device")][0])
    print(json.dumps({"verdict": {"all_ok": all(v[0] for v in results.values()),
                                  "host_failures": [k for k, v in results.items() if k.endswith("host") and not v[0]],
                                  "device_failures": [k for k, v in results.items() if k.endswith("device") and not v[0]],
                                  "host_equals_device": same}}), flush=True)


if __name__ == "__main__":
    main()
