import torch
from ..layer import nBitLinearBase, nBitLinearParameter
from bitorch_engine.utils.model_helper import init_weight

import math
from torch.autograd import Function
import typing
from bitorch_engine.utils.quant_operators import q8_quantization, nv_tensor_quant

from bitorch_engine.utils.safe_import import import_extension
from bitorch_engine.utils.model_helper import flatten_x, unflatten_x

q_linear_cutlass = import_extension("q_linear_cutlass")


class Q8LinearFunction(Function):
    """
    Implements a quantized linear function using 8-bit quantization for both activations and weights.

    This class is designed to perform forward and backward passes of a linear layer with quantization,
    leveraging CUTLASS kernels for efficient computation. The quantization scheme is based on fixed-point
    representation, where 'scale_a' and 'scale_w' are scaling factors for activations and weights, respectively.

    """

    # Note taht both forward and  backward are @staticmethods
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_w: torch.Tensor,
                eps: torch.Tensor, is_train: bool) -> torch.Tensor:
        """
         Forward pass of the quantized linear function.

         Args:
             ctx: Context object to save variables for backward computation.
             x (torch.Tensor): Input tensor.
             weight (torch.Tensor): Weight tensor.
             scale_a (torch.Tensor): Scaling factor for the activations.
             scale_w (torch.Tensor): Scaling factor for the weights.
             eps (torch.Tensor): Epsilon tensor for quantization to avoid division by zero.
             is_train (bool): Flag indicating whether the forward pass is for training or inference.

         Returns:
             torch.Tensor: The output tensor of the quantized linear operation.
         """
        q_a = q8_quantization(x, scale_a, eps).to(torch.int8) if x.dtype != torch.int8 else x
        q_a, shape = flatten_x(q_a)

        if weight.dtype != torch.int8:
            q_w, scale_w = q8_quantization(weight, None, eps)
            q_w = q_w.to(torch.int8)
        else:
            q_w = weight

        output = q_linear_cutlass.q8_forward(q_a, q_w, False, scale_a, scale_w)

        if is_train:
            ctx.save_for_backward(x, q_a, q_w, scale_a, scale_w)

        output = unflatten_x(output, shape)
        return output

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Backward pass of the quantized linear function.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            output_gradient (torch.Tensor): Gradient of the loss with respect to the output of this layer.

        Returns:
            Tuple containing gradients with respect to the input, weight, scale_a, and None placeholders
            for scale_w, eps, and is_train which do not require gradients.
        """
        output_gradient, shape = flatten_x(output_gradient)

        input, q8_activation, q8_weight, scale_a, scale_w = ctx.saved_tensors

        grad_a = output_gradient.mm(q8_weight.to(output_gradient.dtype) * scale_w)
        grad_weight = output_gradient.t().mm(q8_activation.to(output_gradient.dtype) * scale_a)  # (n, k)

        # grad clip range input
        q_w = input / scale_a
        indicate_small = q_w < -128
        indicate_large = q_w > 127
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_input = grad_a * indicate_middle
        grad_input.mul_(scale_a)

        ## grad of the scaling factor for activation
        grad_scale_a = \
            ((indicate_small * -128.0 + indicate_large * 127.0 + indicate_middle * (
                -q_w + q_w.round())) * grad_input * (1.0 / math.sqrt(input.numel()*127.0))).sum().unsqueeze(dim=0)

        grad_input = unflatten_x(grad_input, shape)

        return grad_input, grad_weight, grad_scale_a, None, None, None


class Q8LinearCutlass(nBitLinearBase):
    """
    Implements an 8-bit quantized linear layer using CUTLASS for efficient computation.

    This class inherits from `nBitLinearBase` and adds specific functionality for handling 8-bit quantized weights
    and activations, aiming at reducing memory footprint and accelerating computation on compatible hardware. It
    introduces parameters for scaling and bias adjustment of activations to maintain accuracy with quantized values.

    Attributes:
        bias_a (torch.nn.Parameter): Bias for the activation function, ensuring the quantized model's accuracy.
        scale_a (torch.nn.Parameter): Scale factor for activations, used in quantization to maintain numerical stability.
        scale_w (torch.nn.Parameter): Scale factor for weights, adjusting the quantized weights' magnitude.
        eps (torch.Tensor): A small epsilon value to prevent division by zero in computations, set to 0.00001 by default.

    Methods:
        prepare_params: Prepares and initializes the model parameters for training, converting weights to int8 format.
        generate_quantized_weight: Quantizes the weights, preparing them for efficient storage or computation.
        _check_forward: Verifies the input dimensions match the weight dimensions.
        set_activation: Quantizes activation to 8-bit using the scale factor `scale_a` and adjusts with `bias_a`.
        forward: Defines the forward pass for the quantized linear layer.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the Q8LinearCutlass layer, setting up parameters for activation scaling, weight scaling,
        and a small epsilon value for numerical stability.
        """
        super(Q8LinearCutlass, self).__init__(*args, **kwargs)
        self.bias_a = torch.nn.Parameter(torch.zeros(self.in_channels, dtype=self.dtype))
        self.scale_a = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
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
        pass

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Performs weight quantization, preparing them for efficient computation or storage.

        Args:
            qweight_only (bool): If True, only quantizes the weights without altering other parameters.
        """
        qw, self.scale_w.data = q8_quantization(self.weight, None, self.eps)
        self.qweight = torch.nn.Parameter(
            qw.to(torch.int8)
        )
        if qweight_only:
            self.weight = None

    def _check_forward(self, x: torch.Tensor) -> None:
        """
        Verifies that the input tensor's dimension matches that of the weights.

        Args:
            x (torch.Tensor): The input tensor for the forward pass.
        """
        assert x.size(dim=1) == self.weight.size(dim=1), "Error: input and weights' dim mismatch."

    def set_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes activation to 8-bit and applies a learnable bias adjustment.

        Args:
            x (torch.Tensor): The activation tensor before quantization and bias adjustment.

        Returns:
            torch.Tensor: The adjusted activation tensor.
        """
        # alpha = 2 * tensor.abs().mean() / math.sqrt(Qp) where Qp=127 for 8-bit
        if not self.scale_a.is_nonzero():
            scale = 2 * x.abs().mean() / 11.269
            self.scale_a.data = scale.to(self.dtype)
        # learnable shift input
        return x + self.bias_a.expand_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the quantized linear layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch size, number of features).

        Returns:
            torch.Tensor: The output of the quantized linear layer.
        """
        self._check_forward(x)
        x = self.set_activation(x)
        return Q8LinearFunction.apply(x, self.opt_weight, self.scale_a, self.scale_w, self.eps, self.training)