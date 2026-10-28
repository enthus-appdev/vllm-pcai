"""CPU source checks for vllm#60078: SimpleCPUOffload loads wait for queued compute before copying."""
import ast
import pathlib
import sys

src = (pathlib.Path(sys.argv[1]) / "v1/simple_kv_offload/worker.py").read_text()
tree = ast.parse(src)
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start_load_kv")
body = ast.unparse(fn)
assert "self._load_compute_done.record(" in body, "load compute-done event not recorded"
call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "launch_copy")
kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
assert kw.get("is_store") == "False" and kw.get("wait_event") == "self._load_compute_done", kw
assert body.index("_load_compute_done.record(") < body.index("launch_copy("), "event must be recorded before the copy"
init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
assert "self._load_compute_done" in ast.unparse(init)
backend = (pathlib.Path(sys.argv[1]) / "v1/simple_kv_offload/copy_backend.py").read_text()
assert "wait_event" in backend, "copy backend no longer accepts wait_event"
print("vllm#60078 load-order tripwire OK")
