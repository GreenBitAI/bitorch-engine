import torch
from ..layer import nBitLinearBase, nBitLinearParameter

import math
from torch.autograd import Function
import typing

from bitorch_engine.utils.safe_import import import_extension
from bitorch_engine.utils.model_helper import flatten_x, unflatten_x
from bitorch_engine.functions.cuda import q4_unpack_and_scaling_tensor


q_linear_cutlass = import_extension("q_linear_cutlass")


class Q4LinearFunction(Function):
    """
    Implements a custom linear function with quantization for forward and backward passes.
    This function specifically supports 4-bit quantization for activations and weights during the forward pass,
    and provides gradients for input, weights, and scale factors during the backward pass.

    Note: Both forward and backward methods are static methods.

    The forward pass quantizes inputs and weights to 4-bits using specified scale factors and performs linear (fully connected) operation.
    The backward pass computes gradients for the quantized inputs, weights, and scale factors, considering the quantization effects.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_w: torch.Tensor,
                eps: torch.Tensor, is_train: bool) -> torch.Tensor:
        """
        Forward pass of the Q4Linear function.

        Args:
            ctx: Context object to save variables for backward computation.
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            scale_a (torch.Tensor): Scale factor for input quantization.
            scale_w (torch.Tensor): Scale factor for weight quantization.
            eps (torch.Tensor): Epsilon tensor for quantization to avoid division by zero.
            is_train (bool): Flag indicating if the model is in training mode.

        Returns:
            torch.Tensor: The output tensor after applying the linear operation on quantized inputs and weights.
        """
        x, shape = flatten_x(x)

        outputs = q_linear_cutlass.q4_forward(x, weight, scale_a, scale_w, False, is_train)
        output = outputs[0].to(x.dtype)
        q_a = outputs[1]
        q_w = outputs[2]

        if is_train:
            ctx.save_for_backward(x, q_a, q_w, scale_a, scale_w)

        output = unflatten_x(output, shape)
        return output

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Backward pass of the Q4Linear function.

        Computes gradients for the input tensor, weight tensor, and scale factors based on the output gradient.
        Adjusts gradients based on quantization ranges and effects to ensure proper gradient flow for quantized operations.

        Args:
            ctx: Context object with saved tensors from the forward pass.
            output_gradient (torch.Tensor): Gradient of the loss with respect to the output of this function.

        Returns:
            Tuple[torch.Tensor, ...]: Tuple containing gradients for input tensor, weight tensor, scale factor for activation,
                                      and None placeholders for scale_w and eps which do not receive gradients directly.
        """
        output_gradient, shape = flatten_x(output_gradient)

        input, q_a, q_w, scale_a, scale_w = ctx.saved_tensors

        grad_input = output_gradient.mm(q4_unpack_and_scaling_tensor(q_w, scale_w))  # (m, n)*(n, k) = (m, k)
        grad_weight = output_gradient.t().mm(q4_unpack_and_scaling_tensor(q_a, scale_a))

        # grad clip range input
        q_w = input / scale_a
        indicate_small = q_w < -8
        indicate_large = q_w > 7
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_input.mul_(indicate_middle)

        ## grad of the scaling factor for activation
        grad_scale_a = \
            ((indicate_small * -8.0 + indicate_large * 7.0 + indicate_middle * (
                -q_w + q_w.round())) * grad_input * (1.0 / math.sqrt(input.numel() * 7.0))).sum().unsqueeze(dim=0)

        grad_input = unflatten_x(grad_input, shape)

        return grad_input, grad_weight, grad_scale_a, None, None, None


class Q4LinearCutlass(nBitLinearBase):
    """
    This class implements a quantized linear layer using the CUTLASS library, specifically designed for 4-bit quantization.

    Attributes:
        bias_a (torch.nn.Parameter): Bias parameter for the activations, initialized to zeros.
        scale_a (torch.nn.Parameter): Scale parameter for the activation quantization, initialized to zero.
        scale_w (torch.nn.Parameter): Scale parameter for the weight quantization, initialized to one.
        eps (torch.Tensor): A small epsilon value used to avoid division by zero, registered as a buffer.

    The class inherits from `nBitLinearBase`, extending it with 4-bit quantization capabilities. It introduces parameters
    and methods for managing and applying quantization to both weights and activations within a linear (fully connected) layer context.

    The quantization process involves scaling floating-point weights to a 4-bit representation, which significantly reduces
    memory usage and computational cost, especially on hardware that supports low-precision arithmetic. This class is designed
    to work with neural network models where efficiency and speed are critical, such as on edge devices or in high-performance
    computing environments.

    Methods:
        prepare_params: function will be called between initialization, checkpoint loading and the actual forward pass.
        generate_quantized_weight: Quantizes the weights and optionally removes the floating-point weights to save memory.
        _check_forward: Checks that the input dimensions are compatible with the weight dimensions and certain constraints are met.
        set_activation: Quantizes the activation to 8-bit representation.
        forward: Defines the computation performed at every call.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the Q4LinearCutlass layer, setting up parameters for activation and weight quantization.
        """
        super(Q4LinearCutlass, self).__init__(*args, **kwargs)
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
        self.scale_w.data = 2 * self.weight.abs().mean() / 5.6345
        self.scale_w.data = torch.where(self.scale_w > self.eps, self.scale_w, self.eps)

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Performs weight quantization. This method should be called before saving the model's weights
        to ensure that the weights are quantized. If `qweight_only` is set to True, the original weights
        are discarded to save memory, keeping only the quantized weights.

        Args:
            qweight_only (bool): If True, retains only the quantized weights and discards the original weights.
        """
        self.qweight = torch.nn.Parameter(
            q_linear_cutlass.q4_w_pack(self.weight, self.scale_w)
        )
        if qweight_only:
            self.weight = None

    def _check_forward(self, x: torch.Tensor) -> None:
        """
        Validates the input tensor dimensions against the weight dimensions and checks if they are compatible.
        Additionally, it enforces specific size constraints depending on the mode (training or inference).

        Args:
            x (torch.Tensor): The input tensor to the layer.

        Raises:
            AssertionError: If any of the dimension checks fail.
        """
        assert x.size(dim=1) == self.weight.size(dim=1), "Error: input and weights' dim mismatch."
        if self.training:
            assert x.size(dim=0) % 32 == 0, "Batch size must be divisible by 32 for the training mode."
        assert self.weight.size(dim=1) % 32 == 0, "Output channel dimension must be divisible by 32."
        assert x.size(dim=1) % 32 == 0, "Input channel dimension must be divisible by 32."

    def set_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the activation to 8-bit by applying a scaling factor and optionally adds a learnable bias.
        The scaling factor is computed based on the input tensor's mean absolute value, considering the
        quantization precision.

        Args:
            x (torch.Tensor): The input activation tensor.

        Returns:
            torch.Tensor: The quantized activation tensor.
        """
        # alpha = 2 * tensor.abs().mean() / math.sqrt(Qp) where Qp=127 for 8-bit
        if not self.scale_a.is_nonzero():
            scale = 2 * x.abs().mean() / 11.269
            self.scale_a.data = scale.to(self.dtype)
        # learnable shift input
        return x + self.bias_a.expand_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer. It performs necessary pre-checks on the input tensor, quantizes the
        activations, and then applies the quantized linear function using the quantized weights.

        Args:
            x (torch.Tensor): The input tensor with shape (batch size, number of features).

        Returns:
            torch.Tensor: The output tensor after applying the quantized linear transformation.
        """
        self._check_forward(x)
        x = self.set_activation(x)
        return Q4LinearFunction.apply(x, self.opt_weight, self.scale_a, self.scale_w, self.eps, self.training)


class Q4MatMulFunction(Function):
    """
    This class implements a custom autograd function for quantized matrix multiplication (MatMul)
    using 4-bit quantization. It quantizes the inputs to 4 bits, performs the MatMul operation,
    and then dequantizes the result. This operation is designed to work with low-precision arithmetic
    to improve computational efficiency while maintaining reasonable accuracy.

    Both the forward and backward methods are implemented as static methods, allowing this class
    to be used directly without instantiation.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, x_clip: torch.Tensor, y_clip: torch.Tensor,
                eps: torch.Tensor, is_train: bool) -> torch.Tensor:
        """
        Forward pass of the Q4MatMulFunction. Quantizes inputs, performs MatMul, and dequantizes the output.

        Args:
            ctx: Context object to save tensors for backward computation.
            x (torch.Tensor): Input tensor 1.
            y (torch.Tensor): Input tensor 2.
            x_clip (torch.Tensor): Clipping value for input tensor x, used for quantization range.
            y_clip (torch.Tensor): Clipping value for input tensor y, used for quantization range.
            eps (torch.Tensor): Epsilon value to avoid division by zero during quantization.
            is_train (bool): Flag indicating if the model is in training mode.

        Returns:
            torch.Tensor: The result of the quantized MatMul operation.
        """
        shape_pre = list(x.size()[:-2])

        # outputs[0-2]: result, q4_x, q4_y
        outputs = q_linear_cutlass.q4_matmul(x, y, x_clip, y_clip)
        output = outputs[0].to(x.dtype)

        if is_train:
            ctx.save_for_backward(x, y, outputs[1], outputs[2], x_clip, y_clip, eps)

        # reshape to (bs, num_head, seq_lengh, hid_size_per_head)
        output = output.view(shape_pre + [output.size(-2), output.size(-1)])

        return output*x_clip*y_clip

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Backward pass of the Q4MatMulFunction. Computes gradients for the input tensors based on the output gradient.

        This method calculates the gradients for both input tensors and their clipping values,
        taking into account the quantization performed during the forward pass. It uses a custom backward
        operation defined in `q_linear_cutlass` to compute these gradients efficiently.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            output_gradient (torch.Tensor): Gradient of the loss with respect to the output of this function.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing gradients for x, y, x_clip, y_clip, and None placeholders
                                       for eps and is_train, as these do not require gradients.
        """
        shape_pre = list(output_gradient.size()[:-2])

        x, y, q4_x, q4_y, x_clip, y_clip, eps = ctx.saved_tensors

        scale_grad = 2 * output_gradient.abs().mean() / 11.269 #squre(127)

        grad_x, grad_y = \
            q_linear_cutlass.q4_matmul_backward(
                output_gradient, q4_x, q4_y, x_clip, y_clip, scale_grad)

        # grad clip range x
        q_w = x / x_clip
        indicate_small = q_w < -128
        indicate_large = q_w > 127
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        # reshape to (bs, num_head, seq_lengh, hid_size_per_head)
        grad_x = grad_x.view(shape_pre + [grad_x.size(-2), grad_x.size(-1)])
        grad_x.mul_(indicate_middle)

        ## grad of the scaling factor for activation
        grad_scale_x = \
            ((indicate_small * -128.0 + indicate_large * 127.0 + indicate_middle * (
                -q_w + q_w.round())) * grad_x * (1.0 / math.sqrt(x.numel()*127.0))).sum().unsqueeze(dim=0)

        # grad clip range y
        q_w = y / y_clip
        indicate_small = q_w < -128
        indicate_large = q_w > 127
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_y = grad_y.view(shape_pre + [grad_y.size(-2), grad_y.size(-1)])
        grad_y.mul_(indicate_middle)

        ## grad of the scaling factor for activation
        grad_scale_y = \
            ((indicate_small * -128.0 + indicate_large * 127.0 + indicate_middle * (
                -q_w + q_w.round())) * grad_y * (1.0 / math.sqrt(y.numel()*127.0))).sum().unsqueeze(dim=0)

        return grad_x, grad_y, grad_scale_x, grad_scale_y, None, None


class Q4MatMul(torch.nn.Module):
    """
    A custom PyTorch module for performing quantized matrix multiplication, specifically designed
    for 4-bit quantization. This module quantizes inputs before multiplication based on a dynamic
    scaling factor and aims to maintain high precision in low-bitwidth computations.

    Attributes:
        device (torch.device): The device on which tensors will be allocated.
        dtype (torch.dtype): The data type for the parameters and outputs.
        x_clip (torch.nn.Parameter): The dynamic scaling factor for the first input tensor.
        y_clip (torch.nn.Parameter): The dynamic scaling factor for the second input tensor.
        eps (torch.Tensor): A small epsilon value to prevent division by zero in computations.

    Args:
        dtype (torch.dtype): The desired data type for computations (default: torch.float).
        device (torch.device, optional): The device on which to perform computations.
        *args: Variable length argument list for the parent class.
        **kwargs: Arbitrary keyword arguments for the parent class.
    """
    def __init__(self, dtype = torch.float, device = None, *args, **kwargs):
        super(Q4MatMul, self).__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        self.x_clip = torch.nn.Parameter(torch.tensor(0, dtype=self.dtype))
        self.y_clip = torch.nn.Parameter(torch.tensor(0, dtype=self.dtype))
        self.register_buffer('eps', torch.tensor(0.00001).type(self.dtype))

    def set_activation_scale(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Dynamically sets the scaling factors for input tensors x and y, based on their
        respective values. This scaling helps in quantizing the activations for multiplication.

        The scale is calculated as: alpha = 2 * tensor.abs().mean() / math.sqrt(Qp),
        where Qp = 127 for 8-bit quantization, adjusted here for 4-bit quantization.

        Args:
            x (torch.Tensor): The first input tensor for matrix multiplication.
            y (torch.Tensor): The second input tensor for matrix multiplication.
        """
        if not self.x_clip.is_nonzero():
            scale = 2 * x.abs().mean() / 11.269
            self.x_clip.data = scale.to(self.dtype)
        if not self.y_clip.is_nonzero():
            scale = 2 * x.abs().mean() / 11.269
            self.y_clip.data = scale.to(self.dtype)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q4MatMul module. Validates input dimensions, sets activation scales,
        and performs quantized matrix multiplication.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The result of the quantized matrix multiplication.

        Raises:
            AssertionError: If the input tensors do not have more than two dimensions.
        """
        assert (x.dim() > 2 and y.dim()> 2), \
            "Expected tensor dim > 2, but got input_dim: '{}', other_dim: {}".format(
                x.dim(),
                y.dim()
            )
        self.set_activation_scale(x, y)
        return Q4MatMulFunction.apply(x, y, self.x_clip, self.y_clip, self.eps, self.training)
