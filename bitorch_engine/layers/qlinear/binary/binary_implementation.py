from abc import ABC
from typing import Tuple

from bitorch.layers import QLinearBase
from bitorch.layers.extensions import LayerRecipe
from bitorch.quantizations import Sign, SwishSign

from bitorch_engine.layers.qlinear.qlinear_implementation import QLinearImplementationMixin


class BinaryLinearImplementationMixin(QLinearImplementationMixin, ABC):
    """
    A mixin class for binary linear layer implementations that extends the quantized linear layer implementation mixin (QLinearImplementationMixin).
    This class provides specialized methods to determine if a layer can be cloned based on the quantization functions used for inputs and weights.

    The class supports binary quantization functions such as Sign and SwishSign for both inputs and weights. It leverages the `can_clone` class method
    to check if the specified quantization functions are supported for cloning a layer according to a given recipe.

    Attributes:
        None specified explicitly, but inherits from QLinearImplementationMixin and ABC.

    Methods:
        can_clone: Class method to determine if a layer can be cloned based on its quantization functions for inputs and weights.
    """
    @classmethod
    def can_clone(cls, recipe: LayerRecipe) -> Tuple[bool, str]:
        """
        Determines if a layer can be cloned based on its quantization functions for inputs and weights.

        This method checks if the layer's input and weight quantization functions are among the supported binary quantization functions.
        If either quantization function is not supported, the method returns False along with a message indicating which quantization
        function is not supported.

        Args:
            recipe (LayerRecipe): An object containing the configuration and parameters for the layer to be cloned.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the layer can be cloned and a string message.
            If the layer can be cloned, the boolean is True, and the string is empty. If the layer cannot be cloned due to unsupported
            quantization functions, the boolean is False, and the string contains a message specifying the unsupported quantization function.
        """
        supported_quantization_functions = (Sign, SwishSign) # Define supported quantization functions
        args = QLinearBase.get_args_as_kwargs(recipe)  # Retrieve layer arguments as keyword arguments

        # Check if input quantization function is supported
        if args["input_quantization"].__class__ not in supported_quantization_functions:
            return False, f"the input quantization {args['input_quantization'].name} is not yet supported."

        # Check if weight quantization function is supported
        if args["weight_quantization"].__class__ not in supported_quantization_functions:
            return False, f"the weight quantization {args['weight_quantization'].name} is not yet supported."

        # Call superclass method to perform any additional checks
        return super().can_clone(recipe)
