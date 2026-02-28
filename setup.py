import os
from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension


DEFAULT_CUTLASS_DIR = os.path.join(os.path.dirname(__file__), "external", "cutlass")
CUTLASS_DIR = os.environ.get("CUTLASS_DIR", DEFAULT_CUTLASS_DIR)


if not os.path.isdir(CUTLASS_DIR):
    raise ValueError(f"cutlass directory {CUTLASS_DIR} does not exist.")


ext_modules = [
    CUDAExtension(
        name="rynn_scale.ops._C",
        sources=[
            "rynn_scale/ops/csrc/bindings.cpp",
            "rynn_scale/ops/csrc/grouped_linear_sm90.cu",
            "rynn_scale/ops/csrc/rope_sm90.cu",
        ],
        include_dirs=[
            os.path.join(CUTLASS_DIR, "include"),
            os.path.join(CUTLASS_DIR, "tools", "util", "include"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-DNDEBUG",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_90a,code=sm_90a",
            ],
        },
    )
]


setup(
    name="rynn_scale",
    version="0.1.0",
    description="",
    packages=["rynn_scale"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.9",
)
