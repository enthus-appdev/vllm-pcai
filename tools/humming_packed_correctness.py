#!/usr/bin/env python3
"""One-GPU packed-scale kernel gate using a small real checkpoint expert subset.

Not a full MoE/server qualification. No full-model loading. Exit nonzero on any
compile/launch/nonfinite/bitwise mismatch. Timings include harness output/lock
initialization, not isolated GEMM timing. Run inside the packed-scale image.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 48, 256])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.experts < 6 or args.experts > 256 or args.repeats < 1:
        parser.error("experts must be 6..256 and repeats positive")
    if any(m < 1 or m > 4096 for m in args.rows):
        parser.error("rows must be 1..4096")
    # One compiler at a time; isolate the tuning/cache from serving.
    os.environ["HUMMING_TEST_TUNING_SOURCE"] = "heuristic"
    os.environ["MAX_JOBS"] = "1"
    os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
    os.environ["HUMMING_MAX_PARALLEL_BUILD_WORKERS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    import torch
    from safetensors import safe_open
    from humming import dtypes
    from humming.config import LayerConfig, ComputeConfig, GemmType
    from humming.testing.runner import KernelTestRunner, KernelTestCase
    from humming.transform import transform_humming_tensors

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; CPU checks do not qualify this patch")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This gate is specifically for SM90")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    root = Path(args.model)
    cfg = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    if cfg["hidden_size"] != 4096 or cfg["n_routed_experts"] != 256:
        raise RuntimeError("Unexpected checkpoint architecture")

    def get(key):
        with safe_open(str(root / index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def source(projection):
        weights, scales = [], []
        for expert in range(args.experts):
            prefix = f"mtp.{args.layer}.ffn.experts.{expert}"
            if projection == "w13":
                parts, s_parts = [], []
                for w in ("w1", "w3"):
                    q, s = get(f"{prefix}.{w}.weight"), get(f"{prefix}.{w}.scale")
                    half = q.shape[0] // 2
                    lo, hi = args.tp_rank * half, (args.tp_rank + 1) * half
                    parts.append(q[lo:hi].contiguous())
                    s_parts.append(s[lo:hi].contiguous())
                q, s = torch.cat(parts, 0), torch.cat(s_parts, 0)
            else:
                q, s = get(f"{prefix}.w2.weight"), get(f"{prefix}.w2.scale")
                half, shalf = q.shape[-1] // 2, s.shape[-1] // 2
                q = q[:, args.tp_rank * half:(args.tp_rank + 1) * half].contiguous()
                s = s[:, args.tp_rank * shalf:(args.tp_rank + 1) * shalf].contiguous()
            weights.append(q.view(torch.int32))
            scales.append(s.view(torch.float8_e8m0fnu))
        return {"weight": torch.stack(weights).cuda(),
                "weight_scale": torch.stack(scales).cuda()}

    report = []
    for projection, n, k in (("w13", 2048, 4096), ("w2", 4096, 1024)):
        raw = source(projection)
        assert raw["weight"].shape == (args.experts, n, k // 8)
        assert raw["weight_scale"].shape == (args.experts, n, k // 32)
        runners, kernels = [], []
        for packed in (False, True):
            lc = LayerConfig(sm_version=90, shape_n=n, shape_k=k,
                             num_experts=args.experts, b_dtype=dtypes.float4e2m1,
                             a_dtype=dtypes.float8e4m3, c_dtype=dtypes.bfloat16,
                             bs_dtype=dtypes.float8e8m0, weight_scale_group_size=32,
                             input_scale_group_size=128, use_packed_bs=packed)
            assert lc.use_fused_e8m0_scale and lc.use_packed_k_layout
            cc = ComputeConfig(gemm_type=GemmType.INDEXED)
            runner = object.__new__(KernelTestRunner)
            runner.test_case = KernelTestCase(name=projection, layer_config=lc,
                                             compute_config=cc, top_k=6)
            runner.layer_config, runner.compute_config = lc, cc
            runner.device, runner.tuning_source = 0, "heuristic"
            # The fused transform modifies packed weights: never reuse its input.
            runner.kernel_tensors = transform_humming_tensors(
                lc, {name: value.clone() for name, value in raw.items()})
            groups = runner.prepare_kernels(tuple(args.rows))
            assert len(groups) == 1
            runners.append(runner)
            kernels.append(groups[0][2])
        sizes = [r.kernel_tensors["weight_scale"].numel() for r in runners]
        assert sizes[0] == 2 * sizes[1], sizes
        for m in args.rows:
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            ids = torch.stack([torch.randperm(args.experts, device="cuda")[:6]
                               for _ in range(m)]).to(torch.int32)
            # Same quantized input/routing on both paths.
            _, quant, scale, scale2 = runners[0].prepare_inputs(x)
            outputs, times = [], []
            for runner, choices in zip(runners, kernels):
                kernel_config, tuning, _ = choices[m][0]
                inputs, tensors, _ = runner._prepare_problem(m, quant, ids, tuning["block_shape"][0])
                tensors |= {"inputs": inputs, "input_scale": scale, "input_scale_2": scale2}
                out = runner._launch_kernel(m, tensors, kernel_config)
                again = runner._launch_kernel(m, tensors, kernel_config)
                torch.cuda.synchronize()
                assert torch.isfinite(out).all(), "nonfinite output"
                assert torch.equal(out, again), "unmodified repeat is not bitwise stable; gate inconclusive"
                outputs.append(out)
                # Capture each storage format independently after eager warmup.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = runner._launch_kernel(m, tensors, kernel_config)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, captured), "captured result differs from eager"
                saved_inputs = inputs.clone()
                inputs.zero_()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.count_nonzero(captured).item() == 0, "graph reused stale inputs"
                inputs.copy_(saved_inputs)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, captured), "graph storage did not survive input update"
                # Keep the graph and its tensors alive through its final replay.
                del graph, captured, saved_inputs
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeats):
                    runner._launch_kernel(m, tensors, kernel_config)
                end.record(); end.synchronize()
                times.append(start.elapsed_time(end) / args.repeats)
            if not torch.equal(*outputs):
                delta = (outputs[0].float() - outputs[1].float()).abs().max().item()
                raise AssertionError(f"{projection} M={m}: packed output differs; max_abs={delta}")
            row = {"projection": projection, "rows": m, "bitwise_equal": True,
                   "unpacked_harness_ms": times[0], "packed_harness_ms": times[1]}
            report.append(row)
            print(json.dumps(row), flush=True)
        del runners, kernels, raw
        torch.cuda.empty_cache()
    print("PACKED_SCALE_KERNEL_GATE_PASS " + json.dumps({"cases": len(report),
          "experts": args.experts, "layer": args.layer, "tp_rank": args.tp_rank}), flush=True)


if __name__ == "__main__":
    main()
