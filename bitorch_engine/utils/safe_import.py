import importlib
from typing import Any

from bitorch_engine.extensions import EXTENSION_PREFIX

MESSAGE = """The extension '{}' could not be imported. It is either not yet implemented or was not build correctly.
This message is expected during the build process. If it appears later on, try installing the package again."""


class ExtensionModulePlaceholder:
    """
    A placeholder class for extension modules.

    This class serves as a placeholder for dynamically loaded extension modules. It is designed to
    intercept attribute access and modifications, raising a runtime error if any operation other than
    setting the initial name is attempted. This behavior ensures that any misuse of the placeholder,
    such as accessing or modifying attributes that are not supposed to exist in the placeholder state,
    is promptly identified during development.

    Attributes:
        _name (str): The name of the extension module this placeholder represents.

    Methods:
        __init__: Initializes the placeholder with the name of the extension module.
        __getattr__: Intercepts attribute access attempts, raising a RuntimeError.
        __setattr__: Intercepts attribute modification attempts, allowing only the _name attribute to be set.
    """
    def __init__(self, name: str) -> None:
        """
        Initializes the ExtensionModulePlaceholder with a specified name.

        Args:
            name (str): The name of the extension module this placeholder represents.
        """
        self._name = name

    def __getattr__(self, item: str) -> Any:
        """
        Handles attribute access attempts for the placeholder.

        This method raises a RuntimeError to indicate that the attempted access is invalid, as the
        placeholder should not be used for accessing attributes.

        Args:
            item (str): The name of the attribute being accessed.

        Returns:
            Any: This method does not return but raises RuntimeError instead.

        Raises:
            RuntimeError: Indicates an invalid attribute access attempt.
        """
        raise RuntimeError(MESSAGE.format(self._name))

    def __setattr__(self, key: Any, value: Any) -> None:
        """
        Handles attribute modification attempts for the placeholder.

        This method allows setting the _name attribute only. Any other attempt to modify attributes
        raises a RuntimeError to indicate that the operation is invalid.

        Args:
            key (Any): The name of the attribute to be modified.
            value (Any): The new value for the attribute.

        Raises:
            RuntimeError: Indicates an invalid attribute modification attempt, except for the _name attribute.
        """
        if key == "_name":
            self.__dict__["_name"] = value
            return
        raise RuntimeError(MESSAGE.format(self._name))


def import_extension(module_name: str, not_yet_implemented: bool = False) -> Any:
    """
    Dynamically imports a Python extension module by name, providing a safe mechanism to handle cases
    where the module is not available or not yet implemented. This function is particularly useful for
    conditionally importing modules that provide optional functionality or are platform-specific.

    If the module is marked as not yet implemented (not_yet_implemented=True), or if the module cannot be
    found during import, the function returns a placeholder object instead of raising an ImportError. This
    allows the application to continue running and gracefully handle the absence of the module.

    Args:
        module_name (str): The name of the module to be imported. The actual module name will be prefixed
                           with a predefined prefix defined in `EXTENSION_PREFIX` to form the full module name.
        not_yet_implemented (bool, optional): A flag indicating whether the module is known to be not yet
                                              implemented. If True, the function immediately returns a placeholder
                                              without attempting to import the module. Defaults to False.

    Returns:
        Any: An imported module if successful, or an instance of `ExtensionModulePlaceholder` if the module
             is not implemented or cannot be found.

    Example:
        binary_linear_cuda = import_extension("binary_linear_cuda")
            This example attempts to import a module named "binary_linear_cuda" (prefixed appropriately),
            returning the module if found, or a placeholder if not found or not implemented.

    Note:
        This function prints a warning message to the console if the module cannot be found, informing the
        user of the issue without interrupting the execution of the program.
    """
    if not_yet_implemented:
        return ExtensionModulePlaceholder(module_name)

    try:
        return importlib.import_module(EXTENSION_PREFIX + module_name)
    except ModuleNotFoundError:
        print("Warning:", MESSAGE.format(module_name))
        return ExtensionModulePlaceholder(module_name)
