from abc import ABC
from typing import Tuple

from bitorch.layers import CustomImplementationMixin, QLinearBase
from bitorch.layers.extensions import LayerRecipe


class QLinearImplementationMixin(CustomImplementationMixin, ABC):
    """
    A mixin class for QLinear layer implementations that provides common utility functions
    and checks specific to quantized linear layers. This mixin extends CustomImplementationMixin
    and implements the Abstract Base Class (ABC) to ensure that subclasses provide implementations
    for abstract methods defined in parent classes.

    The class provides a method to check if a given layer configuration can be cloned
    based on specific constraints related to quantized linear layers.
    """
    @classmethod
    def can_clone(cls, recipe: LayerRecipe) -> Tuple[bool, str]:
        """
        Determines if a QLinear layer described by the given recipe can be cloned.

        The method checks if the layer configuration meets certain criteria necessary
        for cloning a quantized linear layer. Specifically, it checks if the layer
        includes bias, and if the number of input features is divisible by 32, as
        these are current limitations for cloning such layers.

        Args:
            recipe (LayerRecipe): An object containing the configuration parameters
                                  of the layer to be cloned.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean and a string. The boolean
                              indicates whether the layer can be cloned (True if it can,
                              False otherwise). The string provides a message explaining
                              why the layer cannot be cloned if the boolean is False.
        """
        # Extract layer arguments from the recipe
        args = QLinearBase.get_args_as_kwargs(recipe)
        if args["bias"]:
            return False, f"bias is not yet supported."
        # Check if the number of input features is divisible by 32
        if args["in_features"] % 32 != 0:
            return False, f"in_features ({args['in_features']}) is not divisible by 32."
        # Layer can be cloned if all checks pass
        return True, ""
