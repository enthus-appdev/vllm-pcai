"""Rebuild only Humming's host launcher using its stable Torch ABI recipe.

Runs without a GPU. Does not rebuild vLLM or CUDA kernels. The native manifest
must match the modified launcher sources, or Humming falls back to runtime JIT.
"""
import ast
import glob
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths


def main():
    spec = importlib.util.find_spec("humming")
    if spec is None or spec.origin is None:
        raise RuntimeError("Humming package not found")
    root = Path(spec.origin).parent
    arch = {"x86_64": "x86_64", "aarch64": "aarch64"}[platform.machine()]
    native = root / "_native" / arch
    source = root / "csrc" / "launcher"
    output = native / "libhumming_launcher.so"
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    torch_lib = Path(torch.__file__).parent / "lib"
    stubs = cuda / "lib64" / "stubs"
    if not (stubs / "libcuda.so").exists():
        raise RuntimeError("CUDA driver link stub not found")
    with tempfile.TemporaryDirectory(prefix="humming-launcher-") as temporary:
        built = Path(temporary) / output.name
        command = [os.environ.get("CXX", "c++"), "-std=c++17", "-O3", "-fPIC", "-shared",
                   "-DUSE_TORCH_STABLE_API=1", "-DTORCH_TARGET_VERSION=0x020B000000000000",
                   str(source / "launcher.cpp"), "-o", str(built)]
        command += [f"-I{path}" for path in include_paths()]
        command += [f"-I{cuda / 'include'}", f"-L{stubs}", "-Wl,--no-as-needed", "-lcuda",
                    str(torch_lib / "libtorch_cpu.so"), str(torch_lib / "libtorch_cuda.so"),
                    "-Wl,-rpath,$ORIGIN/../../../torch/lib", "-Wl,--as-needed"]
        print("Compiling one host translation unit:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        subprocess.run(["readelf", "-h", str(built)], check=True, stdout=subprocess.DEVNULL)
        output.write_bytes(built.read_bytes())
    # Reuse the package's exact hash algorithm without importing GPU-facing modules.
    tree = ast.parse((root / "utils" / "jit.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("hash_to_hex", "hash_path_content")]
    if len(functions) != 2:
        raise RuntimeError("Unexpected Humming source hash API")
    namespace = {"hashlib": hashlib, "glob": glob, "os": os, "Path": Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 "humming-source-hash", "exec"), namespace)
    manifest_path = native / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[output.name] = namespace["hash_path_content"](
        str(source), releative=True, text_only=False)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    print(f"Rebuilt {output}; launcher source hash {manifest[output.name]}", flush=True)


if __name__ == "__main__":
    main()
