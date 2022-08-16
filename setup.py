# import setuptools

import glob
import os
from os import path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "fanjiang", "layers", "op")

    main_source = path.join(extensions_dir, "op.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
    if is_rocm_pytorch:
        assert torch_ver >= [1, 8], "ROCM support requires PyTorch >= 1.8!"

    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "fanjiang._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules



PROJECTS = {
    "fanjiang.projects.wwpred"
}

setup(
    name="fanjiang",
    version="1.0.0",
    author="fanjiang",
    packages=find_packages(),
    # package_dir=PROJECTS,
    python_requires=">=3.6",
    install_requires=[
        # These dependencies are not pure-python.
        # In general, avoid adding more dependencies like them because they are not
        # guaranteed to be installable by `pip install` on all platforms.
        # To tell if a package is pure-python, go to https://pypi.org/project/{name}/#files
        "Pillow>=7.1",  # or use pillow-simd for better performance
        "termcolor>=1.1",
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "tqdm>4.29.0",
        "tensorboard",
        "iopath>=0.1.7,<0.1.10",
        "dataclasses; python_version<'3.7'",
        "omegaconf>=2.2",
        "hydra-core>=1.2.0",
        "matplotlib>=3.5.2",
        "fairscale>=0.4.8",
        "deepspeed>=0.7.0",
        "einops>=0.4.1",
        "scipy>=1.9.0",
        "pandas>=1.4.3",
        "xarray",
        "zarr>=2.12.0",
        "kornia>=0.6.6",
        "netcdf4>=1.6.0",
        # If a new dependency is required at import time (in addition to runtime), it
        # probably needs to exist in docs/requirements.txt, or as a mock in docs/conf.py
    ],
    extras_require={
        # optional dependencies, required by some features
        "all": [
            "shapely",
            "pygments>=2.2",
            "psutil",
        ],
        "dev": [
            "flake8==3.8.1",
            "isort==4.3.21",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
