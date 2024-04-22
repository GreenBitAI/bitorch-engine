from typing import Any

import torch
from bitorch import RuntimeMode
from bitorch.layers.extensions import LayerRecipe
from bitorch.layers.qlinear import QLinearImplementation, QLinearBase

from .binary import BinaryLinear
from .binary.layer import BinaryLinearBase
from .nbit import nBitLinearBase
from .qlinear_implementation import QLinearImplementationMixin


@QLinearImplementation(RuntimeMode.INFERENCE_AUTO)
class QLinearInf(QLinearImplementationMixin, BinaryLinearBase):
    """
    QLinearInf is a class for quantized linear layers optimized for inference.
    It inherits from QLinearImplementationMixin and BinaryLinearBase to utilize
    quantization functionalities and binary linear operations.

    This class specifically handles inference operations with quantized weights,
    potentially using different bit widths for activations and weights.
    """
    @classmethod
    def create_clone_from(cls, recipe: LayerRecipe, device: torch.device = None) -> Any:
        """
        Creates a clone of the layer from a given recipe, adjusting input feature dimensions
        and setting up quantization parameters based on the recipe's specifications.

        Args:
            recipe (LayerRecipe): A configuration object containing layer specifications.
            device (torch.device, optional): The device on which to create the layer. Defaults to None.

        Returns:
            Any: An instance of the cloned layer with quantization applied.
        """
        args = QLinearBase.get_args_as_kwargs(recipe)
        input_features, output_features = args["in_features"], args["out_features"]
        input_features //= 32
        new_layer = cls(
            input_features,
            output_features,
            device=device,
            a_bit=args["input_quantization"].bit_width,
            w_bit=args["input_quantization"].bit_width,
        )
        new_layer.set_weight_data(recipe.layer.weight.data.to(device=device))
        new_layer.generate_quantized_weight(qweight_only=True)
        return new_layer

    def __init__(
            self,
            input_features: int,
            out_features: int,
            device=None,
            a_bit: int = 1,
            w_bit: int = 1,
            bias=False,
    ) -> None:
        """
        Initializes the QLinearInf layer with specified input and output feature dimensions,
        quantization bit widths, and device. Currently, bias is not supported and must be False.

        Args:
            input_features (int): The dimension of input features after bit-packing.
            out_features (int): The dimension of output features (hidden states).
            device (optional): The device on which to initialize the layer. Defaults to None.
            a_bit (int, optional): Bit width for activation quantization. Defaults to 1.
            w_bit (int, optional): Bit width for weight quantization. Defaults to 1.
            bias (bool, optional): Indicates if bias is used. Currently must be False.

        Raises:
            AssertionError: If bias is set to True.
        """
        super().__init__(input_features, out_features, device)
        assert not bias, "currently QLinearInf only supports bias = False"
        self.layer = None
        if a_bit == 1 and w_bit == 1:
            self.layer = BinaryLinear(input_features, out_features, device=device)
        else:
            self.layer = nBitLinearBase(
                input_features, out_features, a_bit, w_bit, device
            )

    def prepare_params(self) -> None:
        """
        Prepares the parameters of the layer for quantization and inference,
        calling the corresponding method of the underlying binary or n-bit linear layer.
        """
        self.layer.prepare_params()

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Generates and sets the quantized weights for the layer, optionally focusing
        only on the quantized weights without affecting the original weights.

        Args:
            qweight_only (bool, optional): If True, only quantized weights are generated. Defaults to False.
        """
        self.layer.generate_quantized_weight(qweight_only=qweight_only)

    def set_weight_data(self, x: torch.Tensor):
        """
        Sets the weight data for the layer.

        Args:
            x (torch.Tensor): The tensor containing the weight data.
        """
        self.layer.set_weight_data(x)

    def set_quantized_weight_data(self, x: torch.Tensor):
        """
        Sets the quantized weight data for the layer.

        Args:
            x (torch.Tensor): The tensor containing the quantized weight data.
        """
        self.layer.set_quantized_weight_data(x)

    @property
    def weight(self):
        """
        Property to access the weight tensor of the layer.

        Returns:
            torch.Tensor: The weight tensor.
        """
        return self.layer.weight

    @property
    def opt_weight(self):
        """
        Property to access the optimized weight tensor of the layer, which may
        include quantized or otherwise transformed weights for efficient inference.

        Returns:
            torch.Tensor: The optimized weight tensor.
        """
        return self.layer.opt_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwards the input tensor x through the quantized linear layer, performing
        the linear operation with quantized weights.

        Args:
            x (torch.Tensor): The input tensor to forward through the layer.

        Returns:
            torch.Tensor: The output tensor after passing through the layer.
        """
        return self.layer(x)
