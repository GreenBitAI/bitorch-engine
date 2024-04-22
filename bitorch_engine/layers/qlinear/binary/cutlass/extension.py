from pathlib import Path

from bitorch_engine.utils.cuda_extension import get_cuda_extension

CUDA_REQUIRED = True
CUTLASS_REQUIRED = True


def get_ext(path: Path):
    """
    Get the CUDA extension for binary linear cutlass.

    Args:
        path (Path): The path to the CUDA extension.

    Returns:
        Extension: The CUDA extension for binary linear cutlass.
    """
    ext = get_cuda_extension(
        path,
        relative_name='binary_linear_cutlass',
        relative_sources=[
            'binary_linear_cutlass.cpp',
            'binary_linear_cutlass_kernel.cu',
        ]
    )
    ext.include_dirs.extend(['.'])
    return ext
