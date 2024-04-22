import torch
from torch import nn
from bitorch_engine.utils.model_helper import qweight_update_fn


class BinaryConv2dBase(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, device=None,
                 dtype: torch.dtype = torch.float, symmetric: bool = True) -> None:
        """
        Initializes a base class for binary convolution layers.

        This class is designed to serve as a base for binary convolution operations,
        where the convolution weights are quantized to binary values. It initializes
        the layer's parameters and sets up the necessary configurations for binary
        convolution.

        Args:
            in_channels (int): The number of channels in the input image.
            out_channels (int): The number of channels produced by the convolution.
            kernel_size (int): The size of the convolving kernel.
            stride (int): The stride of the convolution. Defaults to 1.
            padding (int): Zero-padding added to both sides of the input. Defaults to 0.
            dilation (int): The spacing between kernel elements. Defaults to 1.
            device: The device on which to allocate tensors. Defaults to None.
            dtype (torch.dtype): The desired data type of the parameters. Defaults to torch.float.
            symmetric (bool): Indicates whether to use symmetric quantization. Defaults to True.
        """
        super(BinaryConv2dBase, self).__init__()
        # Initialize class variables and parameters here
        self.bits_binary_word = 8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = None
        self.qweight = None
        self.device = device
        self.dtype = dtype
        self.symmetric = symmetric
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the layer's parameters.

        This method reinitializes the weight parameter of the convolution layer,
        typically with a new random value.
        """
        self.weight = torch.nn.Parameter(torch.empty(
            (self.out_channels, self.in_channels,
             self.kernel_size, self.kernel_size)))

    def set_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the weight data for the convolution layer.

        Args:
            x (torch.Tensor): A tensor containing the weights for the convolution layer.
        """
        self.weight = nn.Parameter(x, requires_grad=False)

    def set_quantized_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the quantized weight data for the convolution layer.

        Args:
            x (torch.Tensor): A tensor containing the quantized weights for the convolution layer.
        """
        self.qweight = nn.Parameter(x, requires_grad=False)

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Generates quantized weights from the current weights.

        This method should implement the logic to convert the layer's weights
        to their quantized form. This is an abstract method and must be implemented
        by subclasses.

        Args:
            qweight_only (bool): If True, the original weight tensor is discarded to save memory.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def prepare_params(self) -> None:
        """
        Prepares and initializes the model parameters for training.

        Note:
            This method MUST be called after model initialization and before training starts to ensure the weights are
            properly prepared for efficient computation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _check_forward(self, x: torch.Tensor) -> None:
        """
        Checks if the input tensor is compatible with the layer's configuration.

        This method validates the dimensions and size of the input tensor against
        the layer's expected input specifications.

        Args:
            x (torch.Tensor): The input tensor to check.
        """
        assert x.size(dim=1) % self.bits_binary_word == 0, \
            "Input tensor dimension must be divisible by {}.".format(self.bits_binary_word)
        assert x.size(dim=1) == self.in_channels, \
            "Dimension mismatch of the input Tensor {}:{}".format(x.size(dim=1), \
                                                                  self.in_channels)

    @property
    def opt_weight(self):
        """
        Returns the weight for the layer.

        This property checks if the layer is in training mode and returns
        the appropriate weight tensor (original or quantized) for use in
        computations.

        Returns:
            torch.nn.Parameter: The optimized weight tensor.
        """
        if not self.training and self.qweight is None:
            self.generate_quantized_weight()
        return self.weight if self.training else self.qweight

    def set_bits_binary_word(self, num_bit: int) -> None:
        """
        Sets the number of bits for binary quantization.

        Args:
            num_bit (int): The number of bits to use for binary quantization.
        """
        self.bits_binary_word = num_bit


class BinaryConvParameter(torch.nn.Parameter):
    """
    A custom parameter class for binary conv layer, extending torch.nn.Parameter.

    This class is designed to support binary conv layers, particularly useful
    in models requiring efficient memory usage and specialized optimization techniques.

    Args:
        data (torch.Tensor, optional): The initial data for the parameter. Defaults to None.
        requires_grad (bool, optional): Flag indicating whether gradients should be computed
                                        for this parameter in the backward pass. Defaults to True.
    """
    def __new__(cls,
                data: torch.Tensor=None,
                requires_grad: bool=True
                ):
        return super().__new__(cls, data=data, requires_grad=requires_grad)

    @staticmethod
    def update(qweight: torch.nn.Parameter, exp_avg_s: torch.Tensor=None, exp_avg_l: torch.Tensor=None,
               step: torch.Tensor=None, lr:float=1e-4, weight_decay:float=0.0, beta1:float=0.99,
               beta2:float=0.9999, eps: float = 1e-6, dtype=torch.half, correct_bias=None, projector=None,
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
        assert isinstance(qweight, BinaryConvParameter), 'Error: the type of qweight must be ' \
                                                              'BinaryConvParameter. '
        qweight_update_fn(qweight=qweight, exp_avg_s=exp_avg_s, exp_avg_l=exp_avg_l,
                          step=step, lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                          correct_bias=correct_bias, eps=eps, dtype=dtype, projector=projector, grad=grad)