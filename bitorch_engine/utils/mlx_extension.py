from pathlib import Path

from torch.utils.cpp_extension import CppExtension

from bitorch_engine.extensions import EXTENSION_PREFIX

from .mlx_path import get_mlx_include_path, get_mlx_lib_path


def get_mlx_extension(root_path: Path, relative_name: str, relative_sources) -> CppExtension:
    """
     Creates and returns a CppExtension for compiling a C++ extension with MLX library support.

     This function is designed to simplify the configuration of a C++ extension that depends on the MLX library by
     automatically setting up include directories, library directories, and other necessary compilation and runtime settings.

     Parameters:
        root_path (Path): The root directory path where the C++ source files are located. This path is used to resolve the full paths to the source files specified in `relative_sources`.
        relative_name (str): A relative name for the extension. This name is prefixed with a predefined prefix and used as the extension's name.
        relative_sources (Iterable[str]): A list or iterable of relative paths to the C++ source files, relative to `root_path`. These source files constitute the extension being compiled.

     Returns:
        CppExtension: An instance of CppExtension configured with paths to include directories, library directories, and other settings needed to compile and link the extension with the MLX library.
     """
    include_path = get_mlx_include_path()
    lib_path = get_mlx_lib_path()
    return CppExtension(
        name=EXTENSION_PREFIX + relative_name,
        sources=[str(root_path / rel_path) for rel_path in relative_sources],
        include_dirs=[include_path],
        library_dirs=[lib_path],            # To find mlx during compilation
        runtime_library_dirs=[lib_path],    # To find mlx during runtime
        libraries=['mlx'],
        extra_compile_args=['-std=c++17'],
    )