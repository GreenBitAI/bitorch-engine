import math

import torch
from torch.autograd import Function
import typing

from bitorch_engine.utils.safe_import import import_extension
from ..layer import nBitConv2dBase
from bitorch_engine.functions.cuda import q4_unpack_and_scaling_tensor

q4_conv_cutlass = import_extension("q4_conv_cutlass")


class Q4Conv2dCutlassForward(Function):
    """
    A custom forward function for 4-bit quantized convolution using CUTLASS kernels.

    This class implements the forward and backward passes for a 4-bit quantized convolution
    operation, intended for use with PyTorch's autograd mechanism. It utilizes CUTLASS low-precision
    computation primitives for efficient GPU acceleration.

    Attributes:
        forward (Function): Performs the forward pass of the 4-bit quantized convolution.
        backward (Function): Implements the backward pass for gradient computation.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_w: torch.Tensor,
                is_train: bool, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
        """
        Forward pass for the 4-bit quantized convolution.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object to save information for backward computation.
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            scale_a (torch.Tensor): Scaling factor for input quantization.
            scale_w (torch.Tensor): Scaling factor for weight quantization.
            is_train (bool): Flag indicating if the model is in training mode.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding added to both sides of the input.
            dilation (int): Spacing between kernel elements.

        Returns:
            torch.Tensor: The result of the 4-bit quantized convolution operation.
        """
        outputs = q4_conv_cutlass.forward(x, weight, scale_a, scale_w, is_train,
                                         kernel_size, stride, padding, dilation)
        output = outputs[0]
        q_a = outputs[1]
        q_w = outputs[2]

        if is_train:
            # variables for backward
            ctx.save_for_backward(x, q_a, q_w, scale_w, scale_a)
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.weight_shape = weight.shape
        return output.permute(0, 3, 1, 2)

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Backward pass for the 4-bit quantized convolution.

        This method computes the gradients of the input tensor and weight tensor based on the gradient of the layer's output.
        It also calculates the gradient of the scale factor used in quantization. The method uses saved tensors and
        attributes from the forward pass for this computation.

        Args:
            ctx (torch.autograd.function.BackwardCFunction): Context object with saved tensors and options.
            output_gradient (torch.Tensor): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            Tuple[torch.Tensor, ...]: Gradients of the input tensor, weight tensor, and scale factors, with None placeholders for non-differentiable parameters.
        """
        input, q_a, q_w, scale_w, scale_a = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        weight_shape = ctx.weight_shape

        # input grad
        qw_unpacked = q4_unpack_and_scaling_tensor(q_w, scale_w).permute(0, 3, 1, 2)
        grad_input = torch.nn.grad.conv2d_input(input.shape, qw_unpacked, output_gradient,
                                                stride=stride, padding=padding, dilation=dilation)
        # weight grad
        qa_unpacked = q4_unpack_and_scaling_tensor(q_a, scale_a).permute(0, 3, 1, 2)
        grad_weight = torch.nn.grad.conv2d_weight(qa_unpacked, weight_shape, output_gradient,
                                                  stride=stride, padding=padding, dilation=dilation)

        # grad clip range input
        sw = input / scale_a
        indicate_small = sw < -8
        indicate_large = sw > 7
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_input.mul_(indicate_middle)

        # grad of the scaling factor for activation
        grad_scale_a = \
            ((indicate_small * -8.0 + indicate_large * 7.0 + indicate_middle * (
                -sw + sw.round())) * grad_input * (1.0 / math.sqrt(input.numel() * 7.0))).sum().unsqueeze(dim=0)

        return grad_input, grad_weight, grad_scale_a, None, None, None, None, None, None


class Q4Conv2dCutlass(nBitConv2dBase):
    """
    A specialized 4-bit quantized convolutional layer using CUTLASS kernels.

    This class extends nBitConv2dBase to implement a 4-bit quantized convolution
    layer optimized for CUTLASS. It supports quantization for both weights and activations,
    aiming to reduce model size and improve computational efficiency while maintaining
    accuracy.

    Attributes:
        bias_a (torch.nn.Parameter): Bias parameter for activation quantization.
        scale_a (torch.nn.Parameter): Scale parameter for activation quantization.
        scale_w (torch.nn.Parameter): Scale parameter for weight quantization.
        eps (torch.Tensor): A small epsilon value to prevent division by zero in calculations.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the Q4Conv2dCutlass layer with provided arguments.

        Args:
            *args: Variable length argument list for base class.
            **kwargs: Arbitrary keyword arguments for base class.
        """
        super(Q4Conv2dCutlass, self).__init__(*args, **kwargs)
        self.bias_a = torch.nn.Parameter(torch.zeros(self.in_channels, dtype=self.dtype))
        self.scale_a = torch.nn.Parameter(torch.tensor(0, dtype=self.dtype))
        self.register_buffer('scale_w', torch.tensor(1, dtype=torch.float))
        self.register_buffer('eps', torch.tensor(0.00001).type(self.dtype))

    def prepare_params(self) -> None:
        """
        Prepares and initializes the model parameters for training.

        Note:
            This method MUST be called after model initialization and before training starts to ensure the weights are
            properly prepared for efficient computation.

            One can use "prepare_bie_layers" method from project_root.utils.model_helper to call this function.
        """
        self.scale_w.data = 2 * self.weight.abs().mean() / 5.6345
        self.scale_w.data = torch.where(self.scale_w > self.eps, self.scale_w, self.eps)

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Performs weight quantization and optionally releases the floating-point weights.

        This method should be called before saving the model weights, especially for inference.

        Args:
            qweight_only (bool): If True, releases the floating-point weight after quantization.
            It will save runtime memory for inference.
        """
        self.qweight = torch.nn.Parameter(
            q4_conv_cutlass.w_pack(self.weight, self.scale_w)
        )
        if qweight_only:
            self.weight = None

    def set_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates scale of input and shift the input using a learnable bias_a.

        Args:
            x (torch.Tensor): The input activation tensor.

        Returns:
            torch.Tensor: The quantized activation tensor.
        """
        # Calculation based on a fixed quantization formula
        # alpha = 2 * tensor.abs().mean() / math.sqrt(Qp) where Qp=127 for 8-bit
        if not self.scale_a.is_nonzero():
            scale = 2 * x.abs().mean() / 11.269
            self.scale_a.data = scale.to(self.dtype)
        # learnable shift input
        return x + self.bias_a.view(1, -1, 1, 1)

    def _check_forward(self, x: torch.Tensor) -> None:
        """
        Checks the input tensor before forward pass. Ensure compatibility with CUTLASS requirements.

        Args:
            x (torch.Tensor): The input tensor to the layer.

        Raises:
            AssertionError: If input or output channels are not divisible by 32, due to CUTLASS constraints.
        """
        # this is the cutlass specific constraints
        assert self.in_channels % 32 == 0, "Input channel dimension must be divisible by 32."
        assert self.out_channels % 32 == 0, "Output channel dimension must be divisible by 32."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the 4-bit quantized convolution using CUTLASS.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            The output tensor of the convolution operation.

        """
        self._check_forward(x)
        x = self.set_activation(x)
        return Q4Conv2dCutlassForward.apply(x, self.opt_weight, self.scale_a, self.scale_w,
                    self.training, self.kernel_size, self.stride, self.padding, self.dilation)
