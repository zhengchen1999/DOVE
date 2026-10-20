"""Build the supplied resample2d kernel for the installed PyTorch/CUDA."""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
setup(
    name="resample2d_cuda",
    ext_modules=[CUDAExtension(
        "resample2d_cuda",
        [str(ROOT / "resample2d_cuda.cc"), str(ROOT / "resample2d_kernel.cu")],
        extra_compile_args={"cxx": ["-std=c++17"], "nvcc": ["-std=c++17"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
