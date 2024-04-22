from typing import TypeVar, Any
import math
import torch
from bitorch import RuntimeMode
from bitorch.layers import QLinearBase
from bitorch.layers.extensions import LayerRecipe
from bitorch.layers.register import QLinearImplementation
from torch._utils import _get_device_index as _torch_get_device_index
from torch.autograd import Function

from bitorch_engine.utils.safe_import import import_extension
from .bmm import BMM
from ..binary_implementation import BinaryLinearImplementationMixin
from ..layer import BinaryLinearBase, BinaryLinearParameter
import typing
from bitorch_engine.utils.quant_operators import nv_tensor_quant
from bitorch_engine.utils.model_helper import init_weight
from bitorch_engine.utils.model_helper import flatten_x, unflatten_x

binary_linear_cuda = import_extension("binary_linear_cuda")

T = TypeVar("T")


class BinaryLinearForward(Function):
    """
    Implements the forward and backward passes for binary linear operation.

    This class performs the binary linear transformation using custom CUDA operations,
    suitable for efficiently processing binary weights and activations in deep learning models.

    Attributes:
        ctx: The context object used to save information for backward computation.
        input: The input tensor for the linear operation.
        weight: The binary weight tensor.
        bmm_type: An enumeration indicating the type of binary matrix multiplication kernel to use.
        scale_a: The scaling factor for the input activation.
        scale_w: The scaling factor for the weight.
        is_train: training or eval
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bmm_type: BMM,
                scale_a: torch.Tensor, scale_w: torch.Tensor, is_train: bool) -> torch.Tensor:
        """
        Forward pass for binary linear operation.

        Args:
            ctx: The context object.
            input: The input tensor.
            weight: The binary weight tensor.
            bmm_type: The binary matrix multiplication kernel type.
            scale_a: The scaling factor for the input activation.
            scale_w: The scaling factor for the weight.

        Returns:
            The result of the binary linear operation scaled by the activation and weight scaling factors.
        """
        input, shape = flatten_x(input)
        if is_train:
            ctx.save_for_backward(input, weight, scale_w, scale_a)
        out = binary_linear_cuda.forward(input, weight, bmm_type, True).to(input.dtype)
        out = unflatten_x(out, shape)
        return out*scale_a*scale_w

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Backward pass for the binary linear operation, computing gradients for input, weight,
        and scale factors based on the output gradient received from subsequent layers.

        Args:
            ctx (torch.autograd.function.BackwardCFunction): The autograd context that stores
                information from the forward pass, including saved tensors for use in gradient
                calculations.
            output_gradient (torch.Tensor): The gradient of the loss with respect to the output
                of the binary linear operation. This tensor is used to compute the gradients for
                the input and weight tensors.

        Returns:
            typing.Tuple[torch.Tensor, ...]: A tuple containing gradients with respect to the
                input tensor (`grad_input`), weight tensor (`grad_weight`), a None placeholder for
                the bias (since this operation doesn't involve a bias term), and the scaling factor
                for the activation (`grad_scale_a`). The gradient for the weight scaling factor is
                not computed and thus returned as None.

        Note:
            - The gradients computation involves sign operations and scaling by the respective
              scaling factors (`scale_w` for weights and `scale_a` for activations) to account for
              the binarization effect in the forward pass.
            - Gradient clipping is applied to `grad_input` to ensure the updates remain within
              the expected range, reflecting the constraints of using binary weights and activations.
            - This method requires a custom optimizer capable of handling int8 gradients and weights,
              as the standard optimizers may not support direct updates with int8 tensors.
        """
        output_gradient, shape = flatten_x(output_gradient)

        input, weight, scale_w, scale_a = ctx.saved_tensors

        wt = weight.type(output_gradient.dtype)
        # grad calculation
        grad_input = output_gradient.mm(wt.sign()*scale_w) # (m, k)

        ## ====== calcualtes grad_weight  ====== ##
        input_sign = input.sign()
        grad_weight = output_gradient.t().mm(input_sign*scale_a) # (n, k)

        # grad clip range input
        q_w = input / scale_a
        indicate_small = q_w < -1
        indicate_large = q_w > 1
        indicate_middle = 1.0 - indicate_small.float() - indicate_large.float()
        grad_input.mul_(indicate_middle)
        ## grad of the scaling factor for activation
        grad_scale_a = torch.sum(grad_input * input_sign * (1.0 / math.sqrt(input.numel())))

        # reshape to original
        grad_input = unflatten_x(grad_input, shape)
        # convert to int8 grad_w
        grad_weight = nv_tensor_quant(grad_weight)[0]

        return grad_input, grad_weight, None, grad_scale_a, None, None


@QLinearImplementation(RuntimeMode.GPU)
class BinaryLinearCuda(BinaryLinearBase, BinaryLinearImplementationMixin):
    """
    A CUDA implementation of binary linear layers for neural networks. This class specializes in handling
    binary weights and activations for efficient computation on GPU devices. It extends BinaryLinearBase
    and mixes in BinaryLinearImplementationMixin to leverage both generic and hardware-specific optimizations.

    Attributes:
        bmm_type (BMM): Specifies the type of binary matrix multiplication kernel to use.
        bits_binary_word (int): Defines the bit width of the binary words used in CUTLASS operations.
        bias_a (torch.nn.Parameter): Layer-wise bias for input activations.
        scale_a (torch.nn.Parameter): Layer-wise scale for input activations to manage quantization effects.
        scale_w (torch.nn.Parameter): Scale for the weights to maintain numerical stability in lower precision.

    Args:
        *args: Variable length argument list for base class initialization.
        bmm_type (BMM): Enum indicating the binary matrix multiplication (BMM) kernel type.
        **kwargs: Arbitrary keyword arguments for base class initialization.
    """
    def __init__(self, *args, bmm_type: BMM = BMM.ADAPTIVE, **kwargs):
        super(BinaryLinearCuda, self).__init__(*args, **kwargs)
        # cutlass uses uint8_t as container
        self.bits_binary_word = 8
        self.bmm_type = bmm_type
        # input scale (layer-wise) and bias (dim=input_features)
        self.bias_a = torch.nn.Parameter(torch.zeros(self.input_features, dtype=self.dtype))
        self.scale_a = torch.nn.Parameter(torch.tensor(0, dtype=self.dtype))
        self.register_buffer('scale_w', torch.tensor(1, dtype=self.dtype))

    @classmethod
    def create_clone_from(cls, recipe: LayerRecipe, device: torch.device = None) -> Any:
        """
        Creates a clone of this layer from a given recipe and device.

        Args:
            recipe (LayerRecipe): A recipe object containing layer configuration and weights.
            device (torch.device, optional): The device on which the layer should be deployed.

        Returns:
            An instance of BinaryLinearCuda with configurations and weights copied from the recipe.
        """
        args = QLinearBase.get_args_as_kwargs(recipe)
        input_features, output_features = args["in_features"], args["out_features"]
        new_layer = cls(input_features, output_features, device)
        new_layer.set_weight_data(recipe.layer.weight.data)
        return new_layer

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

    @property
    def device_id(self) -> int:
        """
        Returns the device index of the current device.

        Returns:
            int: The index of the device.
        """
        return _torch_get_device_index(self.device)

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Generates and sets the quantized weight parameter from the current weights. A bit-packing CUDA kernel
        will be called to do this job.

        Args:
            qweight_only (bool): If True, the original weight tensor is discarded to save memory.
        """
        self.qweight = torch.nn.Parameter(
            binary_linear_cuda.w_pack(
            self.weight,
            self.bmm_type.value,
            True)
        )
        if qweight_only:
            self.weight = None

    @staticmethod
    def w_pack(weights: torch.Tensor, bmm_type: BMM) -> torch.Tensor:
        """
        Packs the given floating-point weights into a binary format suitable for binary matrix multiplication.

        Args:
            weights (torch.Tensor): The floating-point weight tensor to be packed.
            bmm_type (BMM): The binary matrix multiplication kernel type to be used.

        Returns:
            torch.Tensor: The packed binary weights.
        """
        return binary_linear_cuda.w_pack(weights, bmm_type.value, True)

    def set_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sets and scales the activation tensor x using the layer's scaling parameter and bias.

        Args:
            x (torch.Tensor): The input activation tensor.

        Returns:
            torch.Tensor: The scaled and biased activation tensor.
        """
        # use layer norm of the activation value to initialize scale_a
        if not self.scale_a.is_nonzero():
            scale = (2 * x.abs().mean()) if self.symmetric else (4 * x.abs().mean())
            self.scale_a.data = scale.to(self.dtype)
        # learnable shift input
        return x + self.bias_a.expand_as(x)

    def set_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the weight data for this layer and prepares the parameters for training.

        Args:
            x (torch.Tensor): The new weight tensor.
        """
        super().set_weight_data(x)
        self.prepare_params()

    def forward(self, x: torch.Tensor, bmm_type: BMM = BMM.ADAPTIVE) -> torch.Tensor:
        """
        Forward pass for the binary linear layer. Applies quantized matrix multiplication based on the specified
        BMM type, scales and biases the input activations, and returns the output tensor.

        Args:
            x (torch.Tensor): The input activation tensor.
            bmm_type (BMM): The type of binary matrix multiplication kernel to use.

        Returns:
            torch.Tensor: The output tensor of the binary linear operation.
        """
        self._check_forward(x)
        if self.bmm_type is not bmm_type:
            self.bmm_type = bmm_type
        if self.bmm_type is BMM.BTC32: # constraint for bit-tensorcore kernel
            # m, n, k
            m = x.size(dim=0)
            k = x.size(dim=1)
            n = self.output_features
            if m % 8 != 0 or k % 128 != 0 or n % 8 != 0:
                raise Exception("Invalid matrix dimensions for bit-tensorcore (BTC) kernel m:{}, n:{}, k:{}. "
                                "Guidelines: m and n must be multiplies of 8, and k must be multiplies of 128.".format(m, n, k))
        x = self.set_activation(x)
        return BinaryLinearForward.apply(x, self.opt_weight, self.bmm_type.value, self.scale_a, self.scale_w, self.training)
