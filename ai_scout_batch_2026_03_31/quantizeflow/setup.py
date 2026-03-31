```python
import os
import re
import sys
import platform
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

# Check if PyTorch is installed and get its C++ extension utility
TORCH_AVAILABLE = False
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    TORCH_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch not found. QuantizeFlow will be built without C++/CUDA extensions. Some features may be unavailable.", file=sys.stderr)
    # Define dummy classes if PyTorch is not available to allow setup.py to run
    # This helps in cases where setup.py might be run in an environment without torch
    # for tasks like linting, although full build will fail.
    class BuildExtension:
        def build_extensions(self):
            print("Skipping C++/CUDA extension build because PyTorch is not available or its C++ extension utility could not be imported.")
    # Dummy Extension classes for setuptools if torch.utils.cpp_extension fails
    class CppExtension(Extension):
        def __init__(self, name, sources, *args, **kwargs):
            super().__init__(name, sources, *args, **kwargs)
            self.library_dirs = [] # Dummy attribute to prevent errors
    class CUDAExtension(Extension):
        def __init__(self, name, sources, *args, **kwargs):
            super().__init__(name, sources, *args, **kwargs)
            self.library_dirs = [] # Dummy attribute to prevent errors


def get_version():
    """Reads the version from __init__.py"""
    try:
        with open("quantizeflow/__init__.py", "r") as f:
            version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", f.read(), re.M)
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass # Handle case where __init__.py might not exist yet for some setup scenarios
    print("WARNING: Unable to find version string in quantizeflow/__init__.py. Using default '0.0.1'.", file=sys.stderr)
    return "0.0.1"


def read_requirements(path):
    """Reads requirements from requirements.txt"""
    if not os.path.exists(path):
        print(f"WARNING: '{path}' not found. No runtime dependencies will be listed.", file=sys.stderr)
        return []
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# --- Extension Configuration ---
ext_modules = []
cmdclass = {}

if TORCH_AVAILABLE:
    cmdclass['build_ext'] = BuildExtension

    # Determine C++ ABI flag based on PyTorch build
    TORCH_CXX_ABI = None
    if hasattr(torch._C, '_GLIBCXX_USE_CXX11_ABI'):
        TORCH_CXX_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

    extra_compile_args_cpp = [
        "-O3",
        "-Wall",
        "-Wextra",
        "-fPIC" # Position-independent code for shared libraries
    ]
    if TORCH_CXX_ABI is not None:
        extra_compile_args_cpp.append(f"-D_GLIBCXX_USE_CXX11_ABI={TORCH_CXX_ABI}")

    # Common source files for _C extension
    common_ext_sources = [
        "quantizeflow/_C/quantizeflow_ext.cpp",
        "quantizeflow/_C/cpu/if4_quant_cpu.cpp",
        "quantizeflow/_C/cpu/if4_gemm_cpu.cpp",
    ]

    # Common include directories for _C extension
    common_ext_include_dirs = [
        os.path.abspath("quantizeflow/_C/common"),
        os.path.abspath("quantizeflow/_C/cpu"),
    ]

    # Check for CUDA availability
    if torch.cuda.is_available() and 'CUDA_HOME' in os.environ:
        print(f"INFO: CUDA detected. CUDA_HOME={os.environ['CUDA_HOME']}")
        
        cuda_ext_sources = common_ext_sources + [
            "quantizeflow/_C/cuda/block_analyzer.cu",
            "quantizeflow/_C/cuda/if4_quant.cu",
            "quantizeflow/_C/cuda/if4_gemm.cu",
        ]

        cuda_ext_include_dirs = common_ext_include_dirs + [
            os.path.abspath("quantizeflow/_C/cuda"),
        ]

        # CUDA specific compiler flags
        # Targeting common architectures. User might need to customize this based on target hardware.
        nvcc_flags = [
            "-O3",
            "--expt-relaxed-constexpr",
            # Example architectures; adjust based on target hardware support
            "-gencode=arch=compute_75,code=sm_75", # Turing
            "-gencode=arch=compute_80,code=sm_80", # Ampere
            "-gencode=arch=compute_86,code=sm_86", # Ampere
            "-gencode=arch=compute_89,code=sm_89", # Ada Lovelace/Hopper
            "-gencode=arch=compute_90,code=sm_90", # Hopper
            "--use_fast_math"
        ]
        if TORCH_CXX_ABI is not None:
            # Propagate C++ ABI flag to NVCC for consistency with host compiler
            nvcc_flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={TORCH_CXX_ABI}")

        ext_modules.append(
            CUDAExtension(
                name="quantizeflow._C", # Name of the compiled module
                sources=cuda_ext_sources,
                include_dirs=cuda_ext_include_dirs,
                extra_compile_args={
                    "cxx": extra_compile_args_cpp,
                    "nvcc": nvcc_flags,
                },
                # PyTorch's CUDAExtension typically links common CUDA libraries automatically
                # libraries=['cublas', 'cudart'] # Explicitly list if needed
            )
        )
    else:
        print("INFO: CUDA not detected or CUDA_HOME not set. Building QuantizeFlow with CPU extensions only.")
        ext_modules.append(
            CppExtension(
                name="quantizeflow._C",
                sources=common_ext_sources,
                include_dirs=common_ext_include_dirs,
                extra_compile_args=extra_compile_args_cpp,
            )
        )
else:
    print("WARNING: PyTorch not found or its C++ extension utility could not be imported. No C++/CUDA extensions will be built.")


# --- Setup function call ---
setup(
    name="quantizeflow",
    version=get_version(),
    author="QuantizeFlow Team",
    author_email="contact@quantizeflow.org", # Placeholder contact email
    description="A PyTorch/JAX extension for dynamic, block-adaptive quantization (e.g., IF4).",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "QuantizeFlow library for advanced ultra-low precision quantization.",
    long_description_content_type="text/markdown",
    url="https://github.com/quantizeflow/quantizeflow", # Placeholder GitHub repository URL
    license="Apache-2.0", # Common for deep learning projects
    packages=find_packages(exclude=("tests", "examples", "benchmarks")),
    package_dir={"quantizeflow": "quantizeflow"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    # Pass ext_modules and cmdclass only if extensions are configured
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False, # Important for packages with C++ extensions
)
```