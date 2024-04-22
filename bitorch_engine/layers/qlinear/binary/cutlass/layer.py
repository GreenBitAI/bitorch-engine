import torch
import math
from bitorch_engine.layers.qlinear.binary import BinaryLinearBase, BinaryLinearParameter
from bitorch_engine.utils.safe_import import import_extension
from torch.autograd import Function
import typing
from bitorch_engine.utils.model_helper import init_weight
from bitorch_engine.utils.model_helper import flatten_x, unflatten_x, \
    pad_last_2_dims_to_multiple_of_128, \
    binary_matmul_forward_post_processing

from bitorch_engine.utils.quant_operators import nv_tensor_quant
if torch.cuda.is_available():
    from torch._utils import _get_device_index as _torch_get_device_index


binary_linear_cutlass = import_extension("binary_linear_cutlass")


class BinaryLinearForward(Function):
    """
    Implements the forward and backward passes of a binary linear layer.

    This class supports binary linear operations where both the inputs and weights are binarized.
    It integrates scaling and value range conversion directly within the forward pass,
    and provides gradient calculations for backpropagation with respect to the input, weights,
    and scaling factors in the backward pass.

    The forward method supports any input shape by flattening the input, and it can optionally
    save tensors for the backward pass if training is enabled.

    Args:
        input (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor, expected to be binary.
        scale_a (torch.Tensor): The scaling factor for the activation.
        scale_w (torch.Tensor): The scaling factor for the weights.
        gemm_kernel_id (int): An identifier for the GEMM kernel to be used in the CUTLASS library.
        is_train (bool): Flag indicating whether the operation is in training mode.

    Returns:
        torch.Tensor: The output tensor from the binary linear operation,
                      scaled and converted back to the input's dtype.

    Note:
        This implementation relies on the CUTLASS library for efficient binary linear operations.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_w: torch.Tensor,
                gemm_kernel_id: int, is_train: bool) -> torch.Tensor:
        # support any shape flatten
        input, shape = flatten_x(input)
        if is_train:
            ctx.save_for_backward(input, weight, scale_w, scale_a)
        # forward with integrated scaling and value range conversion
        out = binary_linear_cutlass.forward(input, weight, scale_a.item()*scale_w.item(),
                                           False, gemm_kernel_id)
        out = unflatten_x(out, shape)
        return out.to(input.dtype)

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Implements the backward pass of a binary linear layer.

        This method calculates the gradients with respect to the input tensor, weight tensor,
        and the scaling factor for the activation. It also handles gradient clipping for the
        input tensor based on its value range after scaling.

        Args:
            ctx (Any): The autograd context, storing saved tensors from the forward pass.
            output_gradient (torch.Tensor): The gradient of the loss with respect to the output of the layer.

        Returns:
            tuple: A tuple containing gradients with respect to the input tensor, weight tensor,
                   scaling factor for the activation, followed by three `None` placeholders for
                   gradients that are not calculated (scale_w, gemm_kernel_id, is_train).

        Note:
            This method assumes that the weight tensor is of type int8 during the gradient calculation,
            and it performs the sign operation on weights to maintain the binarized nature.
            The method supports backpropagation through layers that use binary weights and activations.
        """
        output_gradient, shape = flatten_x(output_gradient)

        input, weight, scale_w, scale_a = ctx.saved_tensors
        # int8 weight
        wt = weight.type(output_gradient.dtype)
        # grad calculation
        grad_input = output_gradient.mm(wt.sign()*scale_w) # (m, k)

        ## ====== calculates grad_weight bfloat16 ====== ##
        input_sign = input.sign()
        grad_weight = output_gradient.t().mm(input_sign*scale_a) # (n, k)

        # grad clip range input
        q_w = input / scale_a
        indicate_small = q_w < -1
        indicate_large = q_w > 1
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_input.mul_(indicate_middle)

        # grad of the scaling factor for activation
        grad_scale_a = torch.sum(grad_input * input_sign * (1.0 / math.sqrt(input.numel())))

        # reshape to original
        grad_input = unflatten_x(grad_input, shape)

        # convert to int8 grad_w
        grad_weight = nv_tensor_quant(grad_weight)[0]

        return grad_input, grad_weight, grad_scale_a, None, None, None


class BinaryLinearCutlass(BinaryLinearBase):
    """
    A specialized binary linear layer that leverages CUTLASS for efficient low-precision computations.

    This class extends BinaryLinearBase, incorporating specific optimizations for binary neural networks.
    It uses CUTLASS kernels for binary GEMM operations, optimizing the execution on GPUs. The layer supports
    both training and inference in binary precision, significantly reducing memory footprint and computational
    costs.

    Attributes:
        bits_binary_word (int): The bit-width of the binary words used in CUTLASS operations, typically set to 8.
        gemm_kernel_id (int): Identifier for the CUTLASS kernel to be used for the GEMM operation.
        bias_a (torch.nn.Parameter): Layer-wise bias parameter for input activations.
        scale_a (torch.nn.Parameter): Scale factor for input activations, aiding in quantization.
        scale_w (torch.nn.Parameter): Scale factor for weights, essential for maintaining numerical accuracy.

    Methods:
        prepare_params(): Prepares and initializes model parameters for training, converting weights to int8 format.
        generate_quantized_weight(qweight_only=False): Performs bit-packing on 32-bit weights to reduce memory usage.
        set_activation(x): Normalizes input activations using layer-wise scale and bias parameters.
        set_weight_data(x): Prepares the weight data for computation, calling `prepare_params` internally.
        select_gemm_kernel(x): Evaluates and selects the appropriate CUTLASS kernel based on input dimensions.
        forward(x): Defines the forward pass for the binary linear layer, leveraging CUTLASS for efficient computation.
    """
    def __init__(self, *args, **kwargs):
        super(BinaryLinearCutlass, self).__init__(*args, **kwargs)
        # cutlass uses uint8_t as container, default kernel id
        self.bits_binary_word = 8
        self.gemm_kernel_id = 3
        # input scale (layer-wise) and bias (dim=input_features)
        self.bias_a = torch.nn.Parameter(torch.zeros(self.input_features, dtype=self.dtype))
        self.scale_a = torch.nn.Parameter(torch.tensor(0, dtype=self.dtype))
        self.scale_w = torch.nn.Parameter(torch.tensor(1, dtype=self.dtype), requires_grad=False)

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
        self.weight, self.scale_w.data = init_weight(self.weight, cls=BinaryLinearParameter)

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Performs bit-packing on the 32-bit floating-point weights to reduce the model's memory footprint.

        This method converts the full-precision weights to quantized format, specifically designed for
        binary linear operations. It facilitates efficient computation on hardware that supports binary
        operations by reducing the weight representation to 8 bits.

        Args:
            qweight_only (bool): If True, the original floating-point weights are discarded to save memory,
                                 leaving only the quantized weights.

        Note:
            The quantized weights are stored as a new parameter `qweight` within the class.
        """
        self.qweight = torch.nn.Parameter(
            binary_linear_cutlass.w_pack(self.weight, False)
        )
        if qweight_only:
            self.weight = None

    def set_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input activations using a layer-wise scale factor and adds bias. This method is called
        during the forward pass to apply preprocessing to the input activations.

        Args:
            x (torch.Tensor): The input activations to the binary linear layer.

        Returns:
            torch.Tensor: The normalized and biased activations ready for the binary linear operation.

        Note:
            The scale factor `scale_a` is dynamically initialized based on the input's statistical properties
            if it has not been set previously.
        """
        # use layer norm of the activation value to initialize scale_a
        if not self.scale_a.is_nonzero():
            scale = (2 * x.abs().mean()) if self.symmetric else (4 * x.abs().mean())
            self.scale_a.data = scale.to(self.dtype)
        # learnable shift input
        return x + self.bias_a.expand_as(x)

    def set_weight_data(self, x: torch.Tensor):
        """
        Prepares the weight data for the binary linear operation. This method is an extension of the
        base class's method and additionally calls `prepare_params` to ensure that the weights are
        properly formatted for efficient computation.

        Args:
            x (torch.Tensor): The activation tensor that may influence how weights are prepared.
        """
        super().set_weight_data(x)
        self.prepare_params()

    def select_gemm_kernel(self, x: torch.Tensor) -> None:
        """
        Selects the most appropriate GEMM kernel from the available CUTLASS kernels for the binary operation.

        This selection is based on the dimensions of the input activation tensor and the layer's output features.
        The method evaluates available kernels and chooses the optimal one for the given dimensions, enhancing
        computational efficiency.

        Args:
            x (torch.Tensor): The input activation tensor used to determine the optimal GEMM kernel.

        Returns:
            The ID of the selected GEMM kernel which will be used for subsequent operations.

        Note:
            This function is intended to be called during the warmup phase of the model, before actual training
            or inference begins.
        """
        m = x.size(dim=0)
        k = x.size(dim=1)
        n = self.output_features
        kernel_id = binary_linear_cutlass.kernel_eval(_torch_get_device_index(self.weight.device), m, n, k)
        if 1 <= kernel_id <= 9:
            self.gemm_kernel_id = kernel_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the binary linear layer.

        This method applies normalization and bias to the input activations, selects the appropriate
        GEMM kernel, and performs the binary linear operation using the optimized CUTLASS kernel.

        Args:
            x (torch.Tensor): Input tensor with shape (batch size, number of input features).

        Returns:
            The output of the binary linear operation, ready for further processing in the network.
        """
        self._check_forward(x)
        x = self.set_activation(x)
        return BinaryLinearForward.apply(x, self.opt_weight, self.scale_a, self.scale_w,
                                         self.gemm_kernel_id, self.training)


class BinaryMatMulFunction(Function):
    """
    This class implements a custom autograd function for binary matrix multiplication.
    It utilizes specialized hardware acceleration (e.g., CUTLASS) for efficient binary operations
    and is optimized for handling binary inputs with padding to multiples of 128 for better performance.

    The forward pass performs binary matrix multiplication with additional steps to handle padding,
    while the backward pass computes gradients with respect to the inputs, considering clipping thresholds.
    """
    @staticmethod
    def forward(ctx, x, y, x_clip, y_clip) -> torch.Tensor:
        """
        Performs the forward pass of the binary matrix multiplication.

        Args:
            ctx: The context object for storing information for backward computation.
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.
            x_clip (torch.Tensor): The clipping value for the first input tensor.
            y_clip (torch.Tensor): The clipping value for the second input tensor.

        Returns:
            torch.Tensor: The output tensor resulting from the binary matrix multiplication.
        """
        shape_pre = list(x.size()[:-2])
        ctx.save_for_backward(x, y, x_clip, y_clip)

        # Pad along a specified dimension to the nearest multiple of 128
        x, x_pad_sec_last = pad_last_2_dims_to_multiple_of_128(x)
        y, y_pad_sec_last = pad_last_2_dims_to_multiple_of_128(y)

        out = binary_linear_cutlass.matmul(x, y, x_clip.item()*y_clip.item())

        # dealing with truncate, reshape and domain conversion
        out = binary_matmul_forward_post_processing(out, shape_pre, x_pad_sec_last,
                                                    y_pad_sec_last, x.size(-1))
        return out.to(x.dtype) * x_clip * y_clip


    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the gradients for the backward pass of the binary matrix multiplication.

        Args:
            ctx: The context object where saved tensors are retrieved.
            output_gradient (torch.Tensor): The gradient of the loss with respect to the output of the forward pass.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Gradients with respect to the inputs x, y,
            and their respective clipping values x_clip and y_clip.
        """
        x, y, x_clip, y_clip = ctx.saved_tensors

        # grad calculation
        grad_x = output_gradient.matmul(y.sign()*y_clip) # (..., m, k)
        grad_y = output_gradient.transpose(-1, -2).matmul(x.sign()*x_clip) # (..., n, k)

        # grad clip range x
        q_w = x / x_clip
        indicate_small = q_w < -1
        indicate_large = q_w > 1
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_x = grad_x * indicate_middle
        # grad clip range y
        q_w = y / y_clip
        indicate_small = q_w < -1
        indicate_large = q_w > 1
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_y = grad_y * indicate_middle

        grad_x_clip = torch.sum(grad_x * x.sign() * (1.0 / math.sqrt(x.numel())))
        grad_y_clip = torch.sum(grad_y * y.sign() * (1.0 / math.sqrt(y.numel())))
        return grad_x, grad_y, grad_x_clip, grad_y_clip


class BinaryMatMul(torch.nn.Module):
    """
    A PyTorch module for binary matrix multiplication. This module wraps the BinaryMatMulFunction
    and includes parameters for the clipping values used in the binary operations. It is designed
    for integration into neural networks where binary operations can offer computational benefits.

    Attributes:
        dtype (torch.dtype): Data type of the clipping parameters. Defaults to torch.float.
        x_clip (torch.nn.Parameter): Clipping parameter for the first input tensor.
        y_clip (torch.nn.Parameter): Clipping parameter for the second input tensor.
    """
    def __init__(self, dtype = torch.float, *args, **kwargs):
        """
        Initializes the BinaryMatMul module with optional dtype argument for the clipping parameters.

        Args:
            dtype (torch.dtype, optional): The data type for the clipping parameters.
        """
        super(BinaryMatMul, self).__init__(*args, **kwargs)
        self.dtype = dtype
        self.x_clip = torch.nn.Parameter(torch.tensor(0, dtype=self.dtype))
        self.y_clip = torch.nn.Parameter(torch.tensor(0, dtype=self.dtype))

    def set_activation_scale(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Sets the clipping parameters based on the input tensors. If the clipping values are not already set,
        they are initialized based on the mean absolute values of the inputs.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.
        """
        if not self.x_clip.is_nonzero():
            scale = 2 * x.abs().mean()
            self.x_clip.data = scale.to(self.dtype)
        if not self.y_clip.is_nonzero():
            scale = 2 * y.abs().mean()
            self.y_clip.data = scale.to(self.dtype)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BinaryMatMul module. Ensures input tensors are of appropriate dimensions
        and applies the binary matrix multiplication function.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The result of the binary matrix multiplication.
        """
        assert (x.dim() > 2 and y.dim()> 2), \
            "Expected tensor dim > 2, but got input_dim: '{}', other_dim: {}".format(
                x.dim(),
                y.dim()
            )
        self.set_activation_scale(x, y)
        return BinaryMatMulFunction.apply(x, y, self.x_clip, self.y_clip)