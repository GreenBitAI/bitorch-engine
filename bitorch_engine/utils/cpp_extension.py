from pathlib import Path
from typing import Dict, Any
from torch.utils.cpp_extension import CppExtension, IS_MACOS
from bitorch_engine.extensions import EXTENSION_PREFIX
from bitorch_engine.utils.arch_helper import linux_arch_ident, ARCH_CPU, check_cpu_instruction_support


def get_cpp_extension(root_path: Path, relative_name: str, relative_sources) -> Any:
    return CppExtension(
        name=EXTENSION_PREFIX + relative_name,
        sources=[str(root_path / rel_path) for rel_path in relative_sources],
        **get_kwargs()
    )


def get_kwargs() -> Dict[str, Any]:
    """
    Generates keyword arguments for compilation settings, tailored for specific system architectures and operating systems.

    This function configures compiler flags and arguments based on the operating system and CPU architecture. It ensures
    that the appropriate flags are used for MacOS, ARM architectures, and potentially checks for specific CPU instruction
    set support (commented out by default).

    Returns:
        A dictionary containing:
        - `include_dirs`: A list of directories for the compiler to look for header files.
        - `libraries`: A list of libraries to link against. This varies between MacOS (`omp`) and other systems (`gomp`).
        - `extra_compile_args`: A list of additional arguments to pass to the compiler. This includes flags for warnings,
          OpenMP support, and architecture-specific optimizations.

    Note:
        - The function checks if the operating system is MacOS and adjusts the compilation flags accordingly.
        - For ARM architectures on Linux (not MacOS), it adds flags for ARMv8.2-A features and sets the CPU model
          for further optimizations if the model is detected as ARM A55 or A76.
        - The commented code snippet shows how to conditionally add compilation flags based on CPU instruction support,
          such as AVX2.
    """
    extra_compile_args = [
        '-Wall',
        '-Wno-deprecated-register',
    ]
    if IS_MACOS:
        extra_compile_args.append('-Xpreprocessor')

    extra_compile_args.append('-fopenmp')

    if linux_arch_ident.is_arm() and not IS_MACOS:
        extra_compile_args.append('-march=armv8.2-a+fp16+dotprod')
        if linux_arch_ident.get_arm_model() is ARCH_CPU.ARM_A55:
            extra_compile_args.append('-mcpu=cortex-a55')
        if linux_arch_ident.get_arm_model() is ARCH_CPU.ARM_A76:
            extra_compile_args.append('-mcpu=cortex-a76')

    ## can use this code to check the cpu support for instructions
    # if check_cpu_instruction_support('avx2'):
    #     extra_compile_args.append('-mavx2')

    return {
        "include_dirs": [
            "/usr/local/opt/llvm/include",
        ],
        "libraries": [
            "omp" if IS_MACOS else "gomp",
        ],
        "extra_compile_args": extra_compile_args
    }
