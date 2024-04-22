from pathlib import Path

from bitorch_engine.utils.cuda_extension import get_cuda_extension

CUDA_REQUIRED = True
CUTLASS_REQUIRED = True

def get_ext(path: Path):
    """
    Return CUDA extension for Q Linear Cutlass.

    Args:
        path (Path): Path to the directory containing the extension files.

    Returns:
        Extension: CUDA extension for Q Linear Cutlass.
    """
    return get_cuda_extension(
        path,
        relative_name='q_linear_cutlass',
        relative_sources=[
            'q_linear_cutlass.cpp',
            'q4_linear_cutlass_kernel.cu',
            'q8_linear_cutlass_kernel.cu',
        ]
    )
