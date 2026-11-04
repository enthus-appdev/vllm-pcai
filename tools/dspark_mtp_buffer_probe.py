"""Installed-source CPU branch probe, NOT a GPU/model parity test.

Synthetic auxiliaries and a stub projection exercise ordinary, partial and full
staging. A poison residual fails on any read. No CUDA graphs execute here.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def extract(path, name):
    tree = ast.parse(path.read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    for arg in function.args.args:
        arg.annotation = None
    function.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


store = extract(ROOT / "v1/worker/gpu/spec_decode/dflash/speculator.py",
                "_store_hidden_states")
get_hidden = extract(ROOT / "v1/worker/gpu/spec_decode/utils.py",
                     "get_drafter_hidden_states")


class Poison:
    def __getitem__(self, key):
        raise AssertionError("MTP residual was read")


for staged in (0, 2, 4):
    aux = [torch.full((4, 8), float(i)) for i in range(3)]
    output = torch.full((4, 8), float("nan"))
    consumer = NS(
        _aux_staging=torch.zeros((4, 24)), hidden_states=output,
        model=NS(combine_hidden_states=lambda x: x.reshape(-1, 3, 8).sum(1)),
    )
    store(consumer, 4, Poison(), aux, staged)
    if staged:
        assert torch.equal(consumer._aux_staging[:staged],
                           torch.cat(aux, dim=-1)[:staged])
    assert torch.equal(output[staged:], torch.full((4 - staged, 8), 3.0))
    print("PASS auxiliary consumer ignores MTP residual; staging rows", staged)

# Missing auxiliaries must still consume the existing residual fallback.
residual = torch.arange(32, dtype=torch.float32).reshape(4, 8)
consumer = NS(_aux_staging=None, hidden_states=torch.zeros_like(residual))
store(consumer, 4, residual, None, 0)
assert torch.equal(consumer.hidden_states, residual)
print("PASS no-aux consumer preserves residual fallback")

regular = torch.zeros((4, 8))
model = NS(get_mtp_target_hidden_states=lambda: None)
assert get_hidden(model, regular) is regular
model = NS(get_mtp_target_hidden_states=lambda: residual)
assert torch.equal(get_hidden(model, regular), residual)
assert get_hidden(NS(), regular) is regular
print("PASS runner preserves existing, absent-buffer and absent-getter fallbacks")
