from pathlib import Path

from bitorch_engine.utils.cpp_extension import get_cpp_extension


def get_ext(path: Path):
    """Retrieve C++ extension details for binary linear module.

    Args:
        path (Path): Path to the directory containing the extension module.

    Returns:
        Any: Extension module details.
    """
    return get_cpp_extension(
        path,
        relative_name='binary_linear_cpp',
        relative_sources=['binary_linear.cpp']
    )
