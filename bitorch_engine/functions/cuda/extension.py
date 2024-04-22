from pathlib import Path

from bitorch_engine.utils.cuda_extension import get_cuda_extension

CUDA_REQUIRED = True


def get_ext(path: Path):
    """
    Generate CUDA extension for specified path.

    Args:
        path (Path): Path to the directory containing CUDA extension files.

    Returns:
        Extension: CUDA extension for specified path.
    """
    return get_cuda_extension(
        path,
        relative_name='functions_cuda',
        relative_sources=[
            'functions_cuda.cpp',
            'functions_cuda_kernel.cu',
        ]
    )
