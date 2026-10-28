"""CPU-only source/isolated behavior tripwire; not graph or GPU qualification."""
import ast
import pathlib
import sys
import types
import weakref


def load_function(source, name, namespace):
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec("from __future__ import annotations\n" + ast.unparse(fn), namespace)
    return namespace[name]


def verify(root):
    root = pathlib.Path(root)
    config = (root / "config/speculative.py").read_text()
    methods = {}
    for name in ("use_eagle", "use_dflash", "use_dspark", "needs_eagle_block_drop", "use_eagle_block_drop"):
        load_function(config, name, methods)
    Config = type("Config", (), {k: v for k, v in methods.items() if callable(v)})
    for method in ("eagle", "eagle3", "mtp", "dflash", "dspark", "ngram"):
        for disabled in (False, True):
            c = Config()
            c.method, c.disable_eagle_block_drop = method, disabled
            assert c.use_eagle() == (method != "ngram"), method
            assert c.use_eagle_block_drop() == (method in ("eagle", "eagle3", "mtp") and not disabled), method
    assert "self.draft_model_config.model_weights = (" in config, "#48023 removed"
    assert "self.target_model_config.model_weights" in config

    class Tensor:
        pass

    class LayerName:
        pass

    class IntermediateTensors:
        def __init__(self, tensors):
            self.tensors = tensors

    ns = {"torch": types.SimpleNamespace(Tensor=Tensor), "LayerName": LayerName,
          "weak_ref_tensor": weakref.ref}
    fn = load_function((root / "utils/torch_utils.py").read_text(), "weak_ref_tensors", ns)
    a, b, name = Tensor(), Tensor(), LayerName()
    out = fn((a, [b, (7, name)]))
    assert out[0]() is a and out[1][0]() is b and out[1][1] == (7, name)
    del a, b
    assert out[0]() is None and out[1][0]() is None
    # Only the unsupported-type/IntermediateTensors branches import this module.
    stub = types.ModuleType("vllm.sequence")
    stub.IntermediateTensors = IntermediateTensors
    previous = sys.modules.get("vllm.sequence")
    sys.modules["vllm.sequence"] = stub
    try:
        t = Tensor()
        assert fn(IntermediateTensors({"hidden": t})).tensors["hidden"]() is t
        try:
            fn((1, [None]))
        except ValueError:
            pass
        else:
            raise AssertionError("unknown nested type accepted")
    finally:
        if previous is None:
            del sys.modules["vllm.sequence"]
        else:
            sys.modules["vllm.sequence"] = previous

    scheduler = (root / "v1/core/sched/scheduler.py").read_text()
    ns = {}
    free = load_function(scheduler, "_free_encoder_inputs", ns)
    for computed, placeholders, should_free in ((149, 0, False), (150, 0, True), (150, 1, False), (151, 1, True)):
        freed = []
        self = types.SimpleNamespace(
            encoder_cache_manager=types.SimpleNamespace(get_cached_input_ids=lambda _: {0}),
            is_encoder_decoder=False, num_prefill_lookahead=3,
            _free_encoder_input=lambda request, index: freed.append(index))
        request = types.SimpleNamespace(
            mm_features=[types.SimpleNamespace(mm_position=types.SimpleNamespace(offset=50, length=100))],
            num_computed_tokens=computed, num_output_placeholders=placeholders)
        free(self, request)
        assert bool(freed) == should_free, (computed, placeholders, freed)
    tree = ast.parse(scheduler)
    assert any(isinstance(n, ast.If) and ast.unparse(n.test) == "num_new_tokens == 0"
               and any(isinstance(x, ast.Continue) for x in n.body)
               and "skip_request(request_queue)" in ast.unparse(n) for n in ast.walk(tree)), "encoder stall must requeue"
    assert "min(num_new_tokens, start_pos - num_computed_tokens)" in ast.unparse(tree), "lookahead cap regressed"
    assert "speculative_config.needs_eagle_block_drop()" in ast.unparse(tree)
    print("stability backports: CPU predicates, nested ownership and encoder boundaries OK")


if __name__ == "__main__":
    verify(sys.argv[1])
