"""Build-time tripwire for the pcai_embed_offload plugin (CPU only, no model weights)."""
import ast
import importlib.metadata as md
import pathlib
import sys
from types import SimpleNamespace

vllm_dir = pathlib.Path(sys.argv[1])
eps = [e.value for e in md.entry_points(group="vllm.general_plugins")]
assert "pcai_embed_offload:register" in eps, eps

# The hook depends on DeepseekV4Model assigning self.embed_tokens in __init__.
src = (vllm_dir / "models/deepseek_v4/nvidia/model.py").read_text()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4Model")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
assert any(isinstance(t, ast.Attribute) and t.attr == "embed_tokens"
           for n in ast.walk(init) if isinstance(n, ast.Assign) for t in n.targets), "embed_tokens assignment moved"

# UVA offloader contract the hook relies on.
uva = (vllm_dir / "model_executor/offloader/uva.py").read_text()
for needle in ("supports_tower_offload = True", "self.cpu_offload_params", "def wrap_modules", "_vllm_is_uva_offloaded"):
    assert needle in uva, needle

import pcai_embed_offload as p

class Off:
    supports_tower_offload = True
    def __init__(self, params): self.cpu_offload_params = params; self.calls = []
    def wrap_modules(self, mods, prefix=""):
        for m in mods:
            m.weight._vllm_is_uva_offloaded = True; self.calls.append(prefix)

class W:
    shape = (8, 4); dtype = "bf16"

model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=W()))
on = Off({"vision", "embed_tokens"}); assert p.maybe_offload_embedding(model, on) and on.calls == ["embed_tokens"]
off = Off({"vision"}); assert not p.maybe_offload_embedding(SimpleNamespace(embed_tokens=SimpleNamespace(weight=W())), off) and not off.calls
assert not p.maybe_offload_embedding(model, SimpleNamespace(cpu_offload_params={"embed_tokens"}))  # not UVA
print("pcai_embed_offload tripwire OK")
