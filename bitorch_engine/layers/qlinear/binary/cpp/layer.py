from typing import Any

import torch
from bitorch import RuntimeMode
from bitorch.layers import QLinearBase
from bitorch.layers.extensions import LayerRecipe
from bitorch.layers.register import QLinearImplementation
from torch.autograd import Function

from bitorch_engine.utils.safe_import import import_extension
from ..binary_implementation import BinaryLinearImplementationMixin
from ..layer import BinaryLinearBase
from bitorch_engine.utils.model_helper import flatten_x, unflatten_x

binary_linear_cpp = import_extension("binary_linear_cpp")


class BinaryLinearForward(Function):
    """
    A custom autograd function for performing forward pass of binary linear layer.
    This function uses a custom C++ backend for efficient computation.

    Args:
        ctx (torch.autograd.function.FunctionCtx): The context for storing information for backward computation.
        input (torch.Tensor): The input tensor.
        weights (torch.Tensor): The binary weights tensor.
        m (int): The batch size.
        n (int): The number of output features.
        k (int): The number of input features.

    Returns:
        torch.Tensor: The output tensor after applying the binary linear transformation.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weights: torch.Tensor, m: int, n: int, k: int) -> torch.Tensor:
        input, shape = flatten_x(input)
        output = binary_linear_cpp.forward(input, weights, m, n, k)
        output = unflatten_x(output, shape)
        return output


@QLinearImplementation(RuntimeMode.CPU)
class BinaryLinearCPP(BinaryLinearImplementationMixin, BinaryLinearBase):
    """
    A class representing the binary linear layer implemented in C++ for CPU runtime mode.
    Inherits from BinaryLinearBase and mixes in BinaryLinearImplementationMixin for common functionality.

    This class supports creating a clone of itself from a given LayerRecipe, allowing for easy replication
    and modification of layer parameters.
    """
    @classmethod
    def create_clone_from(cls, recipe: LayerRecipe) -> Any:
        """
        Creates a clone of this layer based on the provided LayerRecipe.

        Args:
            recipe (LayerRecipe): The recipe containing the parameters for the clone.

        Returns:
            Any: A new instance of this class with parameters derived from the recipe.
        """
        args = QLinearBase.get_args_as_kwargs(recipe)
        input_features, output_features = args["in_features"], args["out_features"]
        input_features //= 8
        new_layer = cls(input_features, output_features)
        new_layer.set_weight_data(recipe.layer.weight.data)
        new_layer.generate_quantized_weight(qweight_only=True)
        return new_layer

    def __init__(
        self,
        input_features: int,
        out_features: int,
        device: torch.device = None,
    ) -> None:
        """
        Initializes the BinaryLinearCPP layer.

        Args:
            input_features (int): The number of input features (divided by 8 for binary).
            out_features (int): The number of output features.
            device (torch.device, optional): The device on which to perform computations.
        """
        super().__init__(input_features, out_features, device)

    def prepare_params(self) -> None:
        """
        Prepares and initializes the model parameters for training.
        One can use "prepare_bie_layers" method from project_root.utils.model_helper to call this function.
        """
        pass

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Generates the quantized weight matrix for this layer and optionally clears the original weight.

        Args:
            qweight_only (bool, optional): If True, the original weight matrix is cleared to save memory.
        """
        # Generate packed weight using custom C++ function
        self.qweight = binary_linear_cpp.w_pack(
                                    self.weight,    # Original weight
                                    self.output_features,  # n
                                    self.input_features,  # k
                                    )
        if qweight_only:
            self.weight = None # Clear the original weight matrix if specified

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the binary linear layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the binary linear transformation.
        """
        # Check input validity
        self._check_forward(x)
        # pass m, n, k
        m = x.size(dim=0)  # batch size
        k = x.size(dim=1)  # input features
        n = self.output_features  # output features
        return BinaryLinearForward.apply(x, self.opt_weight, m, n, k)
