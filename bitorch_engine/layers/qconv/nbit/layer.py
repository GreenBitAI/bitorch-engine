import math
from torch import nn
import torch
from torch.nn import init
from bitorch_engine.utils.model_helper import qweight_update_fn

class nBitConvParameter(torch.nn.Parameter):
    """
    A custom parameter class for n-bit conv layer, extending torch.nn.Parameter.

    This class is designed to support n-bit conv layers, particularly useful
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
        assert isinstance(qweight, nBitConvParameter), 'Error: the type of qweight must be ' \
                                                              'nBitConvParameter. '
        qweight_update_fn(qweight=qweight, exp_avg_s=exp_avg_s, exp_avg_l=exp_avg_l,
                          step=step, lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                          correct_bias=correct_bias, eps=eps, dtype=dtype, projector=projector, grad=grad)


class nBitConv2dBase(nn.Module):
    def __init__(self, in_channels: int,
                        out_channels: int,
                        kernel_size: int,
                        stride: int = 1,
                        padding: int = 0,
                        dilation: int = 1,
                        a_bit: int = 4,
                        w_bit: int = 4,
                        device=None,
                        dtype=torch.float):
        """
        Initializes the nBitConv2dBase module, a base class for creating convolutional layers with n-bit quantized weights.

        Args:
            in_channels (int): The number of input channels in the convolutional layer.
            out_channels (int): The number of output channels in the convolutional layer.
            kernel_size (int): The size of the convolutional kernel.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            padding (int, optional): The padding added to all sides of the input tensor. Defaults to 0.
            dilation (int, optional): The spacing between kernel elements. Defaults to 1.
            a_bit (int, optional): The bit-width for activation quantization. Defaults to 4.
            w_bit (int, optional): The bit-width for weight quantization. Defaults to 4.
            device (optional): The device on which the module will be allocated. Defaults to None.
            dtype (optional): The desired data type of the parameters. Defaults to torch.float.
        """
        super(nBitConv2dBase, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.dtype = dtype
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.weight = None
        self.qweight = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes or resets the weight parameter using Kaiming uniform initialization.
        """
        self.weight = torch.nn.Parameter(torch.empty(
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def prepare_params(self) -> None:
        """
        Prepares and initializes the model parameters for training.

        Note:
            This method MUST be called after model initialization and before training starts to ensure the weights are
            properly prepared for efficient computation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def set_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the weight parameter with the provided tensor.

        Args:
            x (torch.Tensor): The tensor to be used as the new weight parameter.
        """
        self.weight = nn.Parameter(x, requires_grad=False)

    def set_quantized_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the quantized weight parameter with the provided tensor.

        Args:
            x (torch.Tensor): The tensor to be used as the new quantized weight parameter.
        """
        self.qweight = nn.Parameter(x, requires_grad=False)

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Generates and sets the quantized weight based on the current weight parameter.
        This method should be overridden by subclasses to implement specific quantization logic.

        Args:
            qweight_only (bool, optional): If True, the original weight tensor is discarded to save memory.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _check_forward(self, x: torch.Tensor) -> None:
        """
        Checks the input tensor before forward pass. This method should be implemented by subclasses.

        Args:
            x (torch.Tensor): The input tensor to the layer.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def opt_weight(self):
        """
        Returns the proper weight parameter for the forward pass. If the model is in evaluation mode and
        quantized weights are available, it returns the quantized weights; otherwise, it returns the original weights.

        Returns:
            torch.nn.Parameter: The optimal weight parameter for the forward pass.
        """
        if not self.training and self.qweight is None:
            self.generate_quantized_weight()
        return self.weight if self.training else self.qweight