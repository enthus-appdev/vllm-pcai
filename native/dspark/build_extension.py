"""Build only the added SM90a operator using the base image's CUDA/Torch toolchain.

No model imports, kernel launches or weight downloads.
The upstream kernel and MIT notice are unchanged; headers are from v0.32.0.
"""
import os
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MAX_JOBS", "2")
# Explicit architecture avoids cpp_extension's GPU-based architecture discovery.
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

setup(
    name="pcai-dspark-markov-walk",
    version="0.1.0",
    packages=[],
    ext_modules=[CUDAExtension(
        "pcai_dspark_markov_walk",
        sources=[str(ROOT / "bindings.cpp"), str(ROOT / "csrc/dspark_markov_walk.cu")],
        include_dirs=[str(ROOT / "include")],
        # Match the v0.32.0 stable extension's ABI target, not a newer Torch ABI.
        define_macros=[("TORCH_TARGET_VERSION", "0x020B000000000000ULL")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                     "-gencode=arch=compute_90a,code=sm_90a"],
        },
        extra_link_args=["-Wl,-rpath," + str(Path(torch.__file__).parent / "lib")],
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
