from torch.autograd import Function
import torch
import math
import typing

from bitorch_engine.utils.safe_import import import_extension
from ..layer import BinaryConv2dBase, BinaryConvParameter
from bitorch_engine.utils.model_helper import init_weight
from bitorch_engine.utils.quant_operators import nv_tensor_quant

binary_conv2d_cutlass = import_extension("binary_conv2d_cutlass")


class BinaryConv2dForward(Function):
    """
    Implements the forward pass for a binary convolutional layer.

    This class defines a custom forward and backward pass for binary convolution using PyTorch. The forward pass
    includes integrated scaling and value range conversion for the inputs and weights. If the operation is performed
    during training, necessary variables for the backward pass are saved.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_w: torch.Tensor,
                is_train: bool, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
        """
        Performs the forward pass of the binary convolution.

        Args:
            ctx: Context object to stash information for backward computation.
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Filter weights.
            scale_a (torch.Tensor): Scaling factor for the input.
            scale_w (torch.Tensor): Scaling factor for the weights.
            is_train (bool): Flag indicating if the forward pass is for training.
            kernel_size (int) Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding added to all four sides of the input.
            dilation (int): Spacing between kernel elements.

        Returns:
            torch.Tensor: The output tensor of the binary convolution operation.
        """
        if is_train:
            # variables for backward
            ctx.save_for_backward(x, weight, scale_w, scale_a)
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
        # forward with integrated scaling and value range conversion
        out = binary_conv2d_cutlass.forward(x, weight, scale_a.item()*scale_w.item(), is_train,
                                            kernel_size, stride, padding, dilation)
        return out.to(x.dtype)

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
         Implements the backward pass for binary convolution.

         Args:
             ctx: The context object with saved variables from the forward pass.
             output_gradient (torch.Tensor): Gradient of the loss with respect to the output.

         Returns:
             Tuple[torch.Tensor, ...]: Gradients of the loss with respect to the inputs and weights, and None for
             non-tensor arguments.
         """
        input, weight, scale_w, scale_a = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        input_sign = input.sign()

        # int8 weight
        wt = weight.type(output_gradient.dtype).sign()*scale_w
        # grad calculation
        grad_input = torch.nn.grad.conv2d_input(input.shape, wt, output_gradient, stride=stride,
                                                    padding=padding, dilation=dilation)
        ## ====== calcualtes grad_weight ====== ##
        grad_weight = torch.nn.grad.conv2d_weight(input_sign*scale_a, weight.shape, output_gradient,
                                                  stride=stride, padding=padding, dilation=dilation)
        # grad clip range input
        q_w = input / scale_a
        indicate_small = q_w < -1
        indicate_large = q_w > 1
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_input = grad_input * indicate_middle

        # grad of the scaling factor for activation
        grad_scale_a = torch.sum(grad_input * input_sign * (1.0 / math.sqrt(input.numel())))
        # convert to int8 grad_w
        grad_weight = nv_tensor_quant(grad_weight)[0]

        return grad_input, grad_weight, grad_scale_a, None, None, None, None, None, None


class BinaryConv2dCutlass(BinaryConv2dBase):
    """
    A specialized convolutional layer class that implements binary convolution using the CUTLASS library.
    This class inherits from BinaryConv2dBase and adds specific initializations and methods
    for operating with binary weights and possibly quantized activations.

    Attributes:
        bits_binary_word: CUTLASS utilizes uint8_t as the container type for binary operations.
        bias_a (torch.nn.Parameter): Bias parameter for activation quantization.
        scale_a (torch.nn.Parameter): Scale parameter for activation quantization.
        scale_w (torch.nn.Parameter): Scale parameter for weight quantization.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the BinaryConv2dCutlass layer with additional parameters specific to CUTLASS implementation.
        Args:
            *args: Variable length argument list for base class.
            **kwargs: Arbitrary keyword arguments for base class.
        """
        super(BinaryConv2dCutlass, self).__init__( *args, **kwargs)
        # CUTLASS utilizes uint8_t as the container type for binary operations.
        self.bits_binary_word = 8
        # Initialize bias and scale parameters for activations.
        self.bias_a = torch.nn.Parameter(torch.zeros(self.in_channels, dtype=self.dtype))
        self.scale_a = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.scale_w = torch.nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=False)

    def prepare_params(self) -> None:
        """
        Prepares and initializes the model parameters for training, specifically converting floating-point weights
        to int8 format.

        This method leverages the `init_weight` function to convert the model's floating-point weights to int8,
        achieving a significant reduction in memory usage. It also computes a scale for the weights, which is essential
        for maintaining the numerical fidelity of the model's computations in the lower precision format. The conversion
        to int8 format is particularly beneficial for accelerating training and inference on hardware that supports
        lower precision arithmetic.

        Note:
            This method MUST be called after model initialization and before training starts to ensure the weights are
            properly prepared for efficient computation.

            One can use "prepare_bie_layers" method from project_root.utils.model_helper to call this function.
        """
        self.weight, self.scale_w.data = init_weight(self.weight, cls=BinaryConvParameter)


    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Performs bit-packing on the 32-bit weights to generate quantized weights.
        Args:
            qweight_only: If True, the original weight tensor is discarded to save memory.
        """
        self.qweight = torch.nn.Parameter(
            binary_conv2d_cutlass.w_pack(self.weight)
        )
        if qweight_only:
            self.weight = None

    def set_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adjusts the activation values by initializing scale_a based on the layer's input and adds bias.
        Args:
            x: The input tensor to the convolutional layer.
        Returns:
            The adjusted input tensor.
        """
        if not self.scale_a.is_nonzero():
            scale = (2 * x.abs().mean()) if self.symmetric else (4 * x.abs().mean())
            self.scale_a.data = scale.to(self.dtype)
        # learnable shift input
        return x + self.bias_a.view(1, -1, 1, 1)

    def set_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the weight data from the input tensor and re-initializes from pre-trained weights if available.
        Args:
            x: The input tensor to set as the new weight data.
        """
        super().set_weight_data(x)
        self.prepare_params()

    def _check_forward(self, x: torch.Tensor) -> None:
        """
        Checks if the input tensor's dimensions are compatible with the binary word size.
        Args:
            x: The input tensor to the forward pass.
        Raises:
            AssertionError: If the input tensor's total number of elements is not divisible by bits_binary_word.
        """
        assert x.numel() % self.bits_binary_word == 0, \
            "Input tensor dimension must be divisible by {}.".format(self.bits_binary_word)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the binary convolutional layer.
        Args:
            x: Input tensor with shape (N, C_in, H, W).
        Returns:
            The output tensor of the convolution operation.
        """
        self._check_forward(x)
        x = self.set_activation(x)
        return BinaryConv2dForward.apply(x, self.opt_weight, self.scale_a, self.scale_w,
                    self.training, self.kernel_size, self.stride, self.padding, self.dilation)
