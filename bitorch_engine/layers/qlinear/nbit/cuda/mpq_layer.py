import torch
from torch.autograd import Function
import typing
import math

from bitorch_engine.layers.qlinear.nbit import MPQLinearBase
from bitorch_engine.utils.safe_import import import_extension
from bitorch_engine.utils.model_helper import flatten_x, unflatten_x


q_linear_cuda = import_extension("q_linear_cuda")


class MPQLinearCudaFunction(Function):
    """
    A custom autograd function for mixed-precision quantized (MPQ) linear operations on CUDA.

    This function supports forward and backward passes of a linear layer with quantized weights,
    allowing for efficient computation on CUDA devices. It is specifically designed for scenarios
    where both activations and weights are quantized to different bit-widths for reduced memory
    footprint and computational efficiency, particularly useful in low-power and memory-constrained
    environments such as edge devices.

    The forward pass performs quantized matrix multiplication using custom CUDA kernels, while the
    backward pass computes gradients with respect to the input and updates the privileged gradient
    of the quantized weights.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, qweight: torch.Tensor, a_bit: int, w_bit: int, scales: torch.Tensor,
                zeros: torch.Tensor, g_idx: torch.Tensor, asym: bool, is_training: bool,
                privileged_grad: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass for the MPQ linear operation.

        Args:
           ctx: Context object to save information for backward computation.
           x (torch.Tensor): Input tensor.
           qweight (torch.Tensor): Quantized weights.
           a_bit (int): Activation quantization bit-width.
           w_bit (int): Weight quantization bit-width.
           scales (torch.Tensor): Quantization scales.
           zeros (torch.Tensor): Quantization zero points for asymmetric quantization.
           g_idx (torch.Tensor): Indices for grouped quantization.
           asym (bool): Flag for asymmetric quantization.
           is_training (bool): Training mode flag.

        Returns:
           torch.Tensor: The result of the quantized linear operation.
       """
        x, shape = flatten_x(x)
        output = q_linear_cuda.mpq_forward(x, qweight, scales, zeros, g_idx, a_bit, w_bit, asym)
        if is_training:
            qweight.scales = scales
            qweight.zeros = zeros
            qweight.g_idx = g_idx
            qweight.w_bit = w_bit
            qweight.privileged_grad = privileged_grad
            qweight.asym = asym
            qweight.layer_type = 1
            ctx.a_bit = a_bit
            ctx.save_for_backward(x, qweight)
        output = unflatten_x(output, shape)
        return output

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
         Backward pass for the MPQ linear operation.

         Computes gradients with respect to the input and updates the privileged gradient of the quantized weights.

         Args:
             ctx: Context object with saved tensors from the forward pass.
             output_gradient (torch.Tensor): Gradient of the loss with respect to the output of this operation.

         Note:
             A specific optimizer bitorch_engine.optim.mpq_adamw is available in conjunction with this layer.

         Returns:
             tuple: A tuple containing gradients with respect to the input and None placeholders for other arguments
                    that do not require gradients.
         """
        output_gradient, shape = flatten_x(output_gradient)
        input, qweight = ctx.saved_tensors
        w_bit = qweight.w_bit
        a_bit = ctx.a_bit
        asym = qweight.asym

        if qweight.requires_grad:  # This additional check is required by peft training.
            assert qweight.privileged_grad != None, f"The previledge gradient of qweight can not be None in backward pass."

        output_gradient = output_gradient.to(input.dtype) # to fp16 or fb16 and (m, n) -> (n, m)

        #==================================================================#
        ## grad_input = output_gradient.mm(weight) # (m, n)*(n, k)=(m, k)
        # (k/32*w_bit, n) * (n, m) = (k, m) -> (m, k)
        grad_input = q_linear_cuda.mpq_grad_input(qweight.data, qweight.scales, qweight.zeros, qweight.g_idx,
                                                  output_gradient, a_bit, w_bit, asym)
        #==================================================================#

        # (n, m) * (m, k) = (n, k)
        if qweight.requires_grad:  # This additional check is required by peft training.
            qweight.privileged_grad = output_gradient.t().mm(input).t()  # (k, n)

        grad_input = unflatten_x(grad_input, shape)

        return grad_input, qweight, None, None, None, None, None, None, None, None


class MPQLinearCuda(MPQLinearBase):
    """
    Represents a CUDA-compatible implementation of the mixed precision quantized (MPQ) linear layer,
    inheriting from MPQLinearBase. This class is specifically optimized for CUDA devices, supporting
    operations with quantized weights and activations in a mixed precision format.

    The layer supports quantization bits for weights (w_bit) of 2, 4, or 8 and fixed activation
    bit (a_bit) of 16, ensuring compatibility with common hardware accelerators and optimizing
    performance for deep learning inference tasks on CUDA-enabled GPUs.

    Attributes:
        qweight (torch.nn.Parameter): Quantized weights of the layer, adhering to specified precision.
        w_bit (int): Bit width for weight quantization.
        a_bit (int): Bit width for activation quantization, fixed at 16.
        scales (torch.Tensor): Scale factors for quantized weights, calculated during parameter preparation.
        zeros (torch.Tensor): Zero points for quantized weights, supporting asymmetric quantization.

    Methods:
        check_parameters: Validates the quantization parameters to ensure they meet the requirements.
        prepare_params: Prepares and decompresses quantized parameters for the forward pass. Must be called
                        before performing inference to correctly setup layer parameters.
        forward: Executes the forward pass of the layer using quantized operations.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the MPQLinearCuda layer with given arguments and keyword arguments, setting up
        the layer for CUDA execution with mixed precision quantized weights and activations.
        """
        super(MPQLinearCuda, self).__init__(*args, **kwargs)
        self.qweight.layer_type = 1
        self.check_parameters()

    def check_parameters(self) -> None:
        """
        Ensures that the quantization bit widths for weights (w_bit) and activations (a_bit) are valid.
        Raises an assertion error if the conditions are not met.
        """
        assert self.w_bit in [1, 2, 4, 8], f"The value of w_bit ({self.w_bit}) must be 1, 2, 4 or 8."
        assert self.a_bit == 16, f"The value of a_bit ({self.a_bit}) must be 16."

    def prepare_params(self) -> None:
        '''
        This method should be executed before the actual forward pass. It mainly decompress quantized parameters
        such as qscale and qzero. This step could be simplified or eliminated in the future by having a kernel
        implementation that can decompress during kernel computation.

        One can use "prepare_bie_layers" method from project_root.utils.model_helper to call this function.
        '''
        try:
            if self.use_gba_quant:
                if self.group_size < 256:  # we don't need double quantization for larger group size
                    buffer_shape = (math.ceil(self.in_channels / self.group_size), self.out_channels)
                    if self.asym:
                        qscales = self.qscales.unsqueeze(-1) if self.w_bit == 2 else self.qscales
                        self.zeros = self.qzeros
                    else:
                        qstatistic = self.qstatistic.to(torch.uint8)
                        qscales = (qstatistic & 0xF0) >> 4
                        qzeros = qstatistic & 0x0F
                        self.zeros = ((qzeros.to(self.dtype) - self.qzeros_zeros) * self.qzeros_scales).view(buffer_shape)
                    self.scales = ((qscales.to(self.dtype) - self.qscales_zeros) * self.qscales_scales).view(buffer_shape)
                # release some buffers which will not be used anymore
                del self.qscales_zeros
                del self.qscales_scales
                if self.asym:
                    del self.qscales
                else:
                    del self.qstatistic
                    del self.qzeros_zeros
                    del self.qzeros_scales
            else: # gptq
                self.zeros = self.qzeros

            # release variables not in use
            if self.disable_bias:
                del self.bias
            # only used for loading checkpoints
            del self.wf
            # Manually trigger PyTorch's garbage collector
            torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(f"Error occurred during parameter preparation in MPQLinearCuda layer: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MPQLinearCuda layer using quantized weights and activations.

        Args:
            x (torch.Tensor): The input tensor with shape (batch size, number of features).

        Returns:
            torch.Tensor: The output tensor resulting from the quantized linear transformation and bias addition.
        """
        data_device = x.device
        if not all(tensor.device == data_device for tensor in [self.qweight, self.scales, self.zeros, self.g_idx]):
            raise RuntimeError("Some tensors are not on the correct device, please make sure to move the layer to "
                               "the correct device and call 'finalize_quantized_layers'.")
        out = MPQLinearCudaFunction.apply(x, self.qweight, self.a_bit, self.w_bit,
                                      self.scales, self.zeros, self.g_idx, self.asym, self.training, self.privileged_grad)
        if not self.disable_bias:
            out = out + self.bias
        return out
