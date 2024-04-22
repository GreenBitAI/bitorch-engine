import os
from pathlib import Path
from typing import Union, Tuple


def check_path(p: str) -> Tuple[bool, str]:
    """
    Checks the provided path for the existence of the 'cutlass.h' file in expected directories.

    This function attempts to locate the 'cutlass.h' header file in three potential locations
    relative to the provided path:
    1. Directly within the provided path.
    2. Within a 'cutlass' directory inside the provided path.
    3. Within an 'include/cutlass' directory structure inside the provided path.

    Args:
        p (str): The base path as a string where the search for 'cutlass.h' will begin.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - A boolean indicating whether 'cutlass.h' was found.
            - A string representing the path where 'cutlass.h' was found, or an empty string if not found.
    """
    p = Path(p)
    if (p / "cutlass.h").is_file():
        return True, str(p)
    if (p / "cutlass" / "cutlass.h").is_file():
        return True, str(p)
    if (p / "include" / "cutlass" / "cutlass.h").is_file():
        return True, str(p / "include")
    return False, ""


def find_cutlass(check_only: bool = True) -> Union[bool, str]:
    """
    Searches for the Cutlass library in predefined and environment-specified directories.

    This function iterates through a list of potential directories where Cutlass might be located.
    It checks each directory to see if Cutlass exists there. The search paths include '/usr/local/include'
    and any paths specified in the 'CPATH' environment variable.

    Args:
        check_only (bool): Determines the behavior of the function upon finding Cutlass.
                           If True, the function returns a boolean indicating the presence of Cutlass.
                           If False, it returns the path where Cutlass is found.

    Returns:
        Union[bool, str]: Depending on the value of `check_only`, this function either returns:
                          - A boolean value indicating whether Cutlass was found (True) or not (False).
                          - The string path to the directory where Cutlass is located. If not found, returns an empty string.

    Note:
        The function utilizes `check_path(p)`, a separate function not shown here, to determine
        if Cutlass is present in each directory `p`. It is assumed that `check_path(p)` returns
        a tuple (bool, str), where the boolean indicates success, and the string represents the path.
    """
    success, path = False, ""
    search_paths = ["/usr/local/include"] + os.environ.get("CPATH", "").split(":")
    for p in search_paths:
        if check_only:
            print("Searching Cutlass in:", p)
        success, path = check_path(p)
        if success:
            if check_only:
                print("Found Cutlass in:", p)
            break
    return success if check_only else path


def is_cutlass_available() -> bool:
    return find_cutlass()


def get_cutlass_include_path() -> str:
    return find_cutlass(check_only=False)
