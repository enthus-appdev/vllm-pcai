"""CPU-only activation-resolution tripwire; no Humming kernels are executed."""
import ast
import importlib.metadata
import pathlib
import sys
import types


def verify(root, installed=False):
    root = pathlib.Path(root)
    source = (root / "model_executor/layers/fused_moe/oracle/mxfp4.py").read_text()
    tree = ast.parse(source)
    state = {"override": None, "capability": 90}
    envs = types.SimpleNamespace(VLLM_HUMMING_INPUT_QUANT_CONFIG=None)
    namespace = {
        "envs": envs, "kMxfp8Dynamic": "mxfp8", "kFp8Dynamic128Sym": "fp8-group128",
        "_user_moe_activation_override": lambda: state["override"],
        "current_platform": types.SimpleNamespace(is_device_capability_family=lambda cap: cap == state["capability"]),
        "logger": types.SimpleNamespace(info_once=lambda *args: None),
    }
    for name in ("_resolve_activation_key", "_humming_input_schema"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
    module_name = "vllm.model_executor.layers.quantization.utils.humming"
    stub = types.ModuleType(module_name)
    stub.quant_key_to_input_schema = lambda key: key
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = stub
    try:
        schema = namespace["_humming_input_schema"]
        assert schema("mxfp8") == "fp8-group128", "Hopper default must activate block FP8"
        assert schema(None) is None
        state["capability"] = 100
        assert schema("mxfp8") == "mxfp8"
        state["capability"] = 90
        state["override"] = "mxfp8"
        assert schema("mxfp8") == "mxfp8", "explicit user override must win"
        state["override"] = "incompatible"
        try:
            schema("mxfp8")
        except ValueError:
            pass
        else:
            raise AssertionError("conflicting activation override accepted")
        state["override"] = None
        envs.VLLM_HUMMING_INPUT_QUANT_CONFIG = {"allow_fallback": True}
        assert schema("mxfp8") is None, "BF16 control env must bypass default"
    finally:
        if previous is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = previous
    for model in ("deepseek_v4", "deepseek_v41"):
        code = ast.parse((root / f"models/{model}/quant_config.py").read_text())
        calls = [n for n in ast.walk(code) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Mxfp4MoEMethod"]
        assert calls and all(any(k.arg == "activation_key" and ast.unparse(k.value) == "kMxfp8Dynamic" for k in call.keywords) for call in calls)
    assert "_humming_input_schema(None)" in source and "_humming_input_schema(activation_key)" in source
    assert "_patch_humming" not in (root / "utils/humming.py").read_text(), "obsolete DeviceInfo monkey patch remains"
    if installed:
        assert importlib.metadata.version("humming-kernels") == "0.1.18"
        # No resolver-driven upgrade of the base Torch/Triton stack is permitted.
        assert importlib.metadata.version("torch").split("+")[0].startswith("2.13."), "base Torch changed"
    print("FP8 MoE CPU tripwire: SM90 group128 default, override controls and schema plumbing OK")


if __name__ == "__main__":
    verify(sys.argv[1], "--installed" in sys.argv[2:])
