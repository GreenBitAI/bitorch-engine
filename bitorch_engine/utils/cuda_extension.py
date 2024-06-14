import os
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Any

from torch.utils.cpp_extension import CUDAExtension

from bitorch_engine.extensions import EXTENSION_PREFIX


SUPPORTED_CUDA_ARCHS = ["sm_80", "sm_75"]


def get_cuda_arch():
    cuda_arch = os.environ.get("BIE_CUDA_ARCH", "sm_80")
    if cuda_arch not in SUPPORTED_CUDA_ARCHS:
        warnings.warn(f"Warning: BIE_CUDA_ARCH={cuda_arch} may not be supported yet.")
    return cuda_arch


def get_cuda_extension(root_path: Path, relative_name: str, relative_sources) -> Any:
    if os.environ.get("BIE_BUILD_SEPARATE_CUDA_ARCH", "false") == "true":
        relative_name = f"{relative_name}-{get_cuda_arch()}"
    return CUDAExtension(
        name=EXTENSION_PREFIX + relative_name,
        sources=[str(root_path / rel_path) for rel_path in relative_sources],
        **get_kwargs()
    )


def gcc_version():
    """
    Determines the version of GCC (GNU Compiler Collection) installed on the system.

    This function executes the 'gcc --version' command using subprocess.run and parses the output
    to extract the GCC version number. If GCC is not found or an error occurs during parsing,
    it returns a default version number of 0.0.0.

    The function checks if the output contains the string "clang" to identify if clang is masquerading
    as gcc, in which case it also returns 0.0.0, assuming GCC is not installed.

    Returns:
        tuple: A tuple containing three integers (major, minor, patch) representing the version of GCC.
               Returns (0, 0, 0) if GCC is not found or an error occurs.
    """
    output = subprocess.run(['gcc', '--version'], check=True, capture_output=True, text=True)
    if output.returncode > 0 or "clang" in output.stdout:
        return 0, 0, 0
    first_line = output.stdout.split("\n")[0]
    try:
        version = first_line.split(" ")[-1]
        major, minor, patch = list(map(int, version.split(".")))
        return major, minor, patch
    except:
        return 0, 0, 0


def get_kwargs() -> Dict[str, Any]:
    """
    Generates keyword arguments for compilation based on the GCC version and CUDA architecture.

    This function dynamically constructs a dictionary of extra compilation arguments
    for both C++ and CUDA (nvcc) compilers. It includes flags to suppress deprecated
    declarations warnings, specify the OpenMP library path, and set the CUDA architecture.
    Additionally, it conditionally adjusts the nvcc host compiler to GCC 11 if the detected
    GCC version is greater than 11.

    Returns:
        Dict[str, Any]: A dictionary containing the 'extra_compile_args' key with nested
                        dictionaries for 'cxx' and 'nvcc' compilers specifying their
                        respective extra compilation arguments.
    """
    # Retrieve the current GCC version to determine compatibility and required flags
    major, minor, patch = gcc_version()

    # Initialize the base kwargs dict with common compile arguments for cxx and nvcc
    kwargs = {
        "extra_compile_args": {
            "cxx": [
                "-Wno-deprecated-declarations",  # Suppress warnings for deprecated declarations
                "-L/usr/lib/gcc/x86_64-pc-linux-gnu/10.3.0/libgomp.so",  # Specify libgomp library path for OpenMP
                "-fopenmp", # Enable OpenMP support
            ],
            "nvcc": [
                "-Xcompiler",
                "-fopenmp",  # Pass -fopenmp to the host compiler via nvcc
                f"-arch={get_cuda_arch()}",  # Set CUDA architecture, default to 'sm_80'
                "-DARCH_SM_75" if get_cuda_arch() == 'sm_75' else "-DARCH_SM_80",  # set architecture macro
            ],
        },
    }
    # If the GCC version is greater than 11, adjust the nvcc host compiler settings
    if major > 11:
        print("Using GCC 11 host compiler for nvcc.")
        # Specify GCC 11 as the host compiler for nvcc:
        kwargs["extra_compile_args"]["nvcc"].append("-ccbin=/usr/bin/gcc-11")
    return kwargs
