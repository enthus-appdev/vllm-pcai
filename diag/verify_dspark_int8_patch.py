"""CPU shape/config tripwire; --installed also checks native CUDA registration.

Does not create CUDA tensors or execute the cooperative kernel.
"""
import ast
import pathlib
import sys
import types


def execute(nodes, namespace):
    code = "from __future__ import annotations\n" + "\n".join(ast.unparse(n) for n in nodes)
    exec(code, namespace)


def verify(root, installed=False):
    root = pathlib.Path(root)
    config = ast.parse((root / "config/speculative.py").read_text())
    field = next(n for n in ast.walk(config) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "dspark_markov_quantization")
    assert ast.literal_eval(field.value) is None, "must remain opt-in"
    guard = next(n for n in ast.walk(config) if isinstance(n, ast.If) and ast.unparse(n.test) == "self.dspark_markov_quantization is not None")
    for method, topk, enabled, invalid in (("dspark", None, "int8", False), ("mtp", None, "int8", True), ("dspark", 64, "int8", True), ("mtp", None, None, False)):
        try:
            execute([guard], {"self": types.SimpleNamespace(method=method, dspark_draft_topk=topk, dspark_markov_quantization=enabled)})
        except ValueError:
            assert invalid
        else:
            assert not invalid
    config_source = ast.unparse(config)
    assert "self.target_model_config.model_weights" in config_source, "#48023 removed"

    path = root / "v1/worker/gpu/spec_decode/dspark/markov_walk.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if (
        isinstance(n, ast.Assign) and not any(isinstance(t, ast.Name) and t.id == "logger" for t in n.targets)
    ) or isinstance(n, ast.FunctionDef) and n.name in ("pick_tiles", "_num_sms", "unsupported_reason")]
    platform = types.SimpleNamespace(is_cuda=lambda: True, is_device_capability=lambda cap: cap == 90, num_compute_units=lambda _: 132)
    torch = types.SimpleNamespace(ops=types.SimpleNamespace(_C=types.SimpleNamespace(dspark_markov_walk=object())))
    ns = {"current_platform": platform, "torch": torch}
    execute(nodes, ns)
    assert ns["pick_tiles"](129280, 132) == 16
    assert ns["pick_tiles"](151936, 132) == 18
    assert ns["pick_tiles"](270337, 132) is None
    fake = types.SimpleNamespace(shape=(129280, 256), dtype=types.SimpleNamespace(is_floating_point=True), device=types.SimpleNamespace(index=0), dim=lambda: 2)
    assert ns["unsupported_reason"](fake, fake, 5) is None
    assert ns["unsupported_reason"](fake, fake, 17) is not None
    platform.is_device_capability = lambda cap: False
    assert "sm_90" in ns["unsupported_reason"](fake, fake, 5)
    platform.is_device_capability = lambda cap: cap == 90
    platform.num_compute_units = lambda _: 1
    assert "too large" in ns["unsupported_reason"](fake, fake, 5)
    speculator = (root / "v1/worker/gpu/spec_decode/dspark/speculator.py").read_text()
    assert 'self.speculative_config.dspark_markov_quantization == "int8"' in speculator
    assert "walker.warmup()" in speculator and "self._markov_walker.supports(" in speculator
    assert "draft watermarking is not supported" in speculator
    assert "logit soft cap is not supported" in speculator
    assert "online " in speculator and "acceptance estimator is not supported" in speculator
    assert "torch.ops.load_library" in path.read_text()
    if installed:
        import torch as actual_torch
        actual_torch.ops.load_library(str(path.with_name("pcai_dspark_markov_walk.so")))
        assert actual_torch._C._dispatch_has_kernel_for_dispatch_key("_C::dspark_markov_walk", "CUDA")
        schema = actual_torch._C._dispatch_find_schema_or_throw("_C::dspark_markov_walk", "").schema()
        assert len(schema.arguments) == 17, schema
        assert schema.arguments[11].name == "draft_tokens"
    print("DSpark int8 CPU tripwire: opt-in/config rejection, Hopper geometry, warmup and native registration OK" if installed else "DSpark int8 CPU source tripwire OK (native build not checked)")


if __name__ == "__main__":
    verify(sys.argv[1], "--installed" in sys.argv[2:])
