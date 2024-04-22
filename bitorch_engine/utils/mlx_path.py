import os
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Union


def get_mlx_include_path() -> Union[str, None]:
    """
    Attempts to find the include path for the mlx library.

    This function searches for the 'mlx.h' header file associated with the mlx library
    in various possible locations where the library might be installed. The search follows this order:
    1. Looks within the package's submodule search locations if the mlx package is installed.
    2. Checks the system's default include path under the Python environment's prefix.
    3. Scans paths specified in the 'CPATH' environment variable.

    Returns:
        str: The absolute path to the directory containing 'mlx.h' if found.
        None: If the 'mlx.h' file cannot be found in any of the searched locations.
    """
    filename = "mlx.h"

    mlx_spec = find_spec("mlx")
    if mlx_spec is not None:
        for search_path in mlx_spec.submodule_search_locations:
            path = Path(search_path) / "include"
            if path.exists() and ((path / filename).exists() or (path / "mlx" / filename).exists()):
                return str(path.resolve())

    prefix_path = Path(sys.prefix).resolve()
    if (prefix_path / "include" / "mlx" / filename).exists():
        return str(prefix_path / "include")

    for path in os.environ.get("CPATH", "").split(":"):
        path = Path(path).resolve()
        if (path / "include" / filename).exists():
            return str(path / "include")
        elif (path / filename).exists():
            return str(path)

    return None

def get_mlx_lib_path() -> Union[str, None]:
    """
    Attempts to find the library path for 'libmlx.dylib'.

    This function searches for the 'libmlx.dylib' file in various locations to determine
    the library path for mlx (a hypothetical library). The search follows this order:

    1. Within the 'mlx' package's installation directory, if the package is installed.
       It looks for a 'lib' directory under the package's submodule search locations.
    2. In the 'lib' directory under the Python environment's prefix directory. This is
       typically where libraries are installed for the current Python environment.
    3. In directories specified in the 'LIBRARY_PATH' environment variable. This is a
       colon-separated list of directories where libraries are searched for on Unix-like
       systems.

    The function returns the path as a string if 'libmlx.dylib' is found, or None if the
    library cannot be found in any of the searched locations.

    Returns:
        str or None: The path to 'libmlx.dylib' if found, otherwise None.
    """
    filename = "libmlx.dylib"

    mlx_spec = find_spec("mlx")
    if mlx_spec is not None:
        for search_path in mlx_spec.submodule_search_locations:
            path = Path(search_path) / "lib"
            if path.exists() and ((path / filename).exists() or (path / "mlx" / filename).exists()):
                return str(path.resolve())

    prefix_path = Path(sys.prefix).resolve()
    if (prefix_path / "lib" / filename).exists():
        return str(prefix_path / "lib")

    for path in os.environ.get("LIBRARY_PATH", "").split(":"):
        path = Path(path)
        if (path / "lib" / filename).exists():
            return str(path / "lib")
        elif (path / filename).exists():
            return str(path)

    return None

def is_mlx_available() -> bool:
    """
    Checks if the MLX library is available for use.

    This function determines the availability of the MLX library by verifying both the include path and the library path of MLX.
    It does this by calling two functions: `get_mlx_include_path` and `get_mlx_lib_path`.
    For the MLX library to be considered available, both of these functions must return a value that is not `None`.

    Returns:
        bool: True if both the MLX include path and the MLX library path are available (i.e., not None),
        indicating that the MLX library is available for use. Otherwise, returns False.
    """
    return get_mlx_include_path() is not None and get_mlx_lib_path() is not None