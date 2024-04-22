from pathlib import Path

from bitorch_engine.utils.cpp_extension import get_cpp_extension


def get_ext(path: Path):
    """
    Retrieves the C++ extension for binary convolution.

    Args:
        path (Path): The path to the directory containing the binary convolution C++ code.

    Returns:
        torch.utils.cpp_extension.CppExtension: The C++ extension for binary convolution.
    """
    return get_cpp_extension(
        path,
        relative_name='binary_conv_cpp',
        relative_sources=['binary_conv.cpp']
    )
