import torch
from torch import nn
import math
from bitorch_engine.utils.model_helper import qweight_update_fn


class BinaryLinearParameter(torch.nn.Parameter):
    """
    A custom parameter class for binary linear layer, extending torch.nn.Parameter.

    This class is designed to support binary linear layer, particularly useful
    in models requiring efficient memory usage and specialized optimization techniques.

    Args:
        data (torch.Tensor, optional): The initial data for the parameter. Defaults to None.
        requires_grad (bool, optional): Flag indicating whether gradients should be computed
                                        for this parameter in the backward pass. Defaults to True.
    """

    def __new__(cls,
                data: torch.Tensor = None,
                requires_grad: bool = True
                ):
        return super().__new__(cls, data=data, requires_grad=requires_grad)

    @staticmethod
    def update(qweight: torch.nn.Parameter, exp_avg_s: torch.Tensor = None, exp_avg_l: torch.Tensor = None,
               step: torch.Tensor = None, lr: float = 1e-4, weight_decay: float = 0.0, beta1: float = 0.99,
               beta2: float = 0.9999, eps: float = 1e-6, dtype=torch.half, correct_bias=None, projector=None,
               grad:torch.Tensor=None) -> None:
        """
        This method defines how to update quantized weights with quantized gradients.
        It may involve operations such as applying momentum or adjusting weights based on some optimization algorithm.

        Args:
            qweight (torch.nn.Parameter): The current quantized weight parameter to be updated.
            exp_avg_s (torch.Tensor, optional): Exponential moving average of squared gradients. Used in optimization algorithms like Adam.
            exp_avg_l (torch.Tensor, optional): Exponential moving average of the gradients. Also used in optimizers like Adam.
            step (torch.Tensor, optional): The current step or iteration in the optimization process. Can be used to adjust learning rate or for other conditional operations in the update process.
            lr (float, optional): Learning rate. A hyperparameter that determines the step size at each iteration while moving toward a minimum of a loss function.
            weight_decay (float, optional): Weight decay (L2 penalty). A regularization term that helps to prevent overfitting by penalizing large weights.
            beta1 (float, optional): The exponential decay rate for the first moment estimates. A hyperparameter for optimizers like Adam.
            beta2 (float, optional): The exponential decay rate for the second-moment estimates. Another hyperparameter for Adam-like optimizers.
            eps (float, optional): A small constant for numerical stability.
            dtype (torch.dtype, optional): The data type to be used for computations.
            correct_bias (optional): Whether to apply bias correction (specific to certain models like BERT).
            projector (optinal): Whether use a gradient projector.
            grad (optional): gradient tensor will be used if projector used.

        Returns:
            None: The function is expected to update the `qweight` in-place and does not return anything.

        Raises:
            NotImplementedError: Indicates that the function has not yet been implemented.
        """
        assert isinstance(qweight, BinaryLinearParameter), 'Error: the type of qweight must be ' \
                                                           'BinaryLinearParameter. '
        qweight_update_fn(qweight=qweight, exp_avg_s=exp_avg_s, exp_avg_l=exp_avg_l,
                          step=step, lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                          correct_bias=correct_bias, eps=eps, dtype=dtype, projector=projector, grad=grad)


class BinaryLinearBase(nn.Module):
    """
    Base class for binary linear layers, supporting both floating-point and quantized weights.

    This class is designed to facilitate the creation of binary linear layers,
    where weights can be represented in a quantized format for efficient computation,
    especially on hardware that supports binary operations. It provides a foundation
    for implementing various types of binary linear operations, including fully connected
    layers and convolutional layers with binary weights.

    Attributes:
        bits_binary_word (int): Number of bits in a binary word, default is 8.
        input_features (int): Dimension of input features after bit-packing.
        output_features (int): Dimension of output features or hidden states.
        weight (nn.Parameter): Floating-point weights, used for training and initialization.
        qweight (nn.Parameter): Quantized weights, used for inference when training is False.
        device (torch.device): Device on which the layer's tensors will be allocated.
        dtype (torch.dtype): Data type of the layer's floating-point weights.
        symmetric (bool): Indicates if the quantization should be symmetric around 0.

    Methods:
        reset_parameters: Initializes or resets the layer's parameters.
        set_weight_data: Sets the layer's floating-point weights from an external tensor.
        set_quantized_weight_data: Sets the layer's quantized weights from an external tensor.
        generate_quantized_weight: Converts the floating-point weights to quantized format.
        _check_forward: Validates the compatibility of input tensor and weights before forward pass.
        opt_weight (property): Returns the optimal weights for the current mode (training or inference).
        set_bits_binary_word: Sets the number of bits used in a binary word for quantization.
    """
    def __init__(
        self, input_features: int, out_features: int, device: torch.device = None,
            dtype: torch.dtype = torch.float, symmetric: bool = True
    ) -> None:
        """
        Initializes the BinaryLinearBase class with specified configurations.

        Args:
            input_features (int): Dimension of input features after bit-packing.
            out_features (int): Dimension of output features or hidden states.
            device (torch.device, optional): Device on which to allocate tensors. Defaults to None.
            dtype (torch.dtype, optional): Data type for floating-point weights. Defaults to torch.float.
            symmetric (bool, optional): If True, quantization is symmetric around 0. Defaults to True.
        """
        super().__init__()
        self.bits_binary_word = 8
        self.input_features = input_features
        self.output_features = out_features
        self.qweight = None
        self.device = device
        self.dtype = dtype
        self.symmetric = symmetric
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes or resets the floating-point weight parameters using Kaiming uniform initialization.
        """
        self.weight = nn.Parameter(
            torch.Tensor(
                self.output_features, self.input_features
            ).type(self.dtype)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def set_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the floating-point weight parameter from an external tensor.

        Args:
            x (torch.Tensor): A tensor containing the new weight data.
        """
        assert (self.dtype == x.dtype), \
            "dtype mismatch. Expected: '{}', but '{}' found".format(
            torch.float,
            x.dtype
        )
        self.weight = nn.Parameter(x)

    def prepare_params(self) -> None:
        """
        Prepares and initializes the model parameters for training.

        Note:
            This method MUST be called after model initialization and before training starts to ensure the weights are
            properly prepared for efficient computation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def set_quantized_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the quantized weight parameter from an external tensor, disabling gradient computation.

        Args:
            x (torch.Tensor): A tensor containing the new quantized weight data.
        """
        self.qweight = nn.Parameter(x, requires_grad=False)

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Converts the floating-point weights to a quantized format through bit-packing.

        Args:
            qweight_only (bool, optional): If True, only updates the qweight without modifying the floating-point weight. Defaults to False.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _check_forward(self, x: torch.Tensor) -> None:
        """
        Validates the compatibility of the input tensor and weights before the forward pass.

        Args:
            x (torch.Tensor): The input tensor for the forward pass.
        """
        if x.dtype is not torch.uint8:
            assert (
                x.size(dim=-1) % self.bits_binary_word == 0
            ), "Input tensor dimension ({}) must be divisible by {}."\
                .format(x.size(dim=-1), self.bits_binary_word)

        if self.qweight is not None:
            bits_condt = 1 if x.dtype is torch.uint8 else self.bits_binary_word
            assert (
                self.qweight.nelement()
                == x.size(dim=-1) * self.output_features / bits_condt
            ), "Weight and input tensor mismatch. {}:{}".format(
                self.qweight.nelement(),
                x.size(dim=-1) * self.output_features / bits_condt
            )
        else:
            if x.dtype is torch.uint8:
                assert self.weight.size(dim=1) / self.bits_binary_word == x.size(dim=-1), \
                    "Weight and input tensor mismatch."
            else:
                assert self.weight.size(dim=1) == x.size(dim=-1), \
                    "Weight and input tensor mismatch."

    @property
    def opt_weight(self) -> nn.Parameter:
        """
        Returns the optimal weight parameter for the current mode (training or inference).

        Returns:
            nn.Parameter: The floating-point weights during training or the quantized weights during inference.
        """
        if not self.training and self.qweight is None:
            self.generate_quantized_weight()
        return self.weight if self.training else self.qweight

    def set_bits_binary_word(self, num_bit: int) -> None:
        """
        Sets the number of bits used in a binary word for the purpose of quantization.

        Args:
            num_bit (int): The number of bits to use in a binary word.
        """
        self.bits_binary_word = num_bit