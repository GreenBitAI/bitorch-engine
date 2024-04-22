from pathlib import Path

from bitorch_engine.utils.cuda_extension import get_cuda_extension

CUDA_REQUIRED = True

def get_ext(path: Path):
    """
    Get the CUDA extension for quantized linear operations.

    Args:
        path (Path): The path to the CUDA extension directory.

    Returns:
        Any: The CUDA extension module.
    """
    ext = get_cuda_extension(
        path,
        relative_name='q_linear_cuda',
        relative_sources=[
            'q_linear_cuda.cpp',
            'mpq_linear_cuda_kernel.cu',
            'mbwq_linear_cuda_kernel.cu',
        ]
    )
    ext.include_dirs.extend(['exl2'])
    return ext