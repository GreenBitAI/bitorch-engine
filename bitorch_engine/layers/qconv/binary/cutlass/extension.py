from pathlib import Path

from bitorch_engine.utils.cuda_extension import get_cuda_extension

CUDA_REQUIRED = True
CUTLASS_REQUIRED = True


def get_ext(path: Path):
    """
    Obtains the CUDA extension for Cutlass-based binary convolution.

    Args:
        path (Path): The path to the directory containing the necessary source files
                     for the Cutlass-based binary convolution operation.

    Returns:
        Extension: The CUDA extension for the Cutlass-based binary convolution.
    """
    return get_cuda_extension(
        path,
        relative_name='binary_conv2d_cutlass',
        relative_sources=[
            'binary_conv2d_cutlass.cpp',
            'binary_conv2d_cutlass_kernel.cu',
        ]
    )
