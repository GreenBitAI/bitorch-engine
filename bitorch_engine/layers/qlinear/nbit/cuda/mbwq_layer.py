import torch
from torch.autograd import Function
import typing
import math

from bitorch_engine.layers.qlinear.nbit import MPQLinearBase, MPQWeightParameter
from bitorch_engine.utils.safe_import import import_extension
from bitorch_engine.utils.model_helper import flatten_x, unflatten_x
from .utils import unpack_qweight, make_group_map

q_linear_cuda = import_extension("q_linear_cuda")


class MBWQLinearCudaFunction(Function):
    """
    Custom CUDA function for performing forward and backward passes in MBWQ Linear layers.

    This function supports both forward and backward passes, implemented as static methods.
    The forward pass calculates the output of the MBWQ Linear layer based on the input tensor and quantized weights.
    The backward pass computes gradients with respect to the input tensor and quantized weights.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, qweight: torch.Tensor, use_mbw: bool, is_train: bool,
                scales: torch.Tensor, zeros: torch.Tensor, group_size: int, q_perm: torch.Tensor=None, bits: int=4,
                privileged_grad: torch.Tensor=None, q_group_map: torch.Tensor=None, rows: list=None
    ) -> torch.Tensor:
        """
        Perform the forward pass of MBWQ Linear layer using CUDA.

        This method computes the output of a linear layer with mixed binary weight quantization (MBWQ), optimizing the
        computation for CUDA-enabled devices. It supports both standard quantization and MBWQ modes, dynamically adjusting
        the computation based on the `use_mbw` flag. Additionally, it can operate in both training and inference modes,
        indicated by the `is_train` flag.

        Args:
            ctx (Any): Autograd context, used for saving variables needed for backward computation.
            x (torch.Tensor): Input tensor, representing the data that will be processed by the layer.
            qweight (torch.Tensor): Quantized weights tensor, which contains the quantized values of the weights used in the layer.
            use_mbw (bool): Flag indicating whether to use Mixed Binary Weight Quantization (MBWQ) mode for processing.
            is_train (bool): Flag indicating whether the operation is being performed in training mode.
            scales (torch.Tensor): Scale factors for quantization.
            zeros (torch.Tensor): Zero points for quantization.
            group_size (int): The size of groups for group-wise quantization.
            q_perm (torch.Tensor, optional): Permutation tensor for reordering the quantized weights.
            q_group_map (torch.Tensor, optional): Mapping tensor for group-wise quantization.
            rows (list, optional): Contains distribution and permutation information for weights in MBWQ mode.
            bits (int): q_weight's bitwidth.

        Returns:
            torch.Tensor: The output tensor of the forward pass, after processing by the MBWQ Linear layer.

        Note:
            This method is specifically optimized for CUDA computation and should be used when performance on CUDA-enabled
            devices is a priority. The implementation details and parameter usage may be subject to change as the method is
            experimental and optimized for advanced use cases.
        """
        x, shape = flatten_x(x)

        if not use_mbw:
            output = q_linear_cuda.mbwq_q4_forward(x, qweight, scales, zeros, group_size, q_perm, bits)
        else:
            output = q_linear_cuda.mbwq_exl2_forward(x, qweight, scales, zeros, q_perm, q_group_map, rows, False) # set whether use_cublas

        if is_train:
            qweight.scales = scales
            qweight.zeros = zeros
            qweight.privileged_grad = privileged_grad
            qweight.q_perm = q_perm
            qweight.group_size = group_size
            qweight.q_group_map = q_group_map
            qweight.rows = rows
            qweight.layer_type = 2
            qweight.w_bit = bits
            qweight.asym = False
            qweight.g_idx = None
            ctx.save_for_backward(x, qweight)

        output = unflatten_x(output, shape)
        return output

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Perform the backward pass of MBWQ Linear layer.

        Args:
            ctx (Any): Autograd context.
            output_gradient (torch.Tensor): Output gradient.

        Returns:
            Tuple[torch.Tensor, ...]

        Note:
            This method is experimental and may not guarantee error-free or consistent behavior.
        """
        output_gradient, shape = flatten_x(output_gradient)
        input, qweight = ctx.saved_tensors

        if qweight.requires_grad: # This additional check is required by peft training.
            assert qweight.privileged_grad != None, f"The previledge gradient of qweight can not be None in backward pass."

        output_gradient = output_gradient.to(input.dtype) # to fp16 or fb16 and (m, n) -> (n, m)

        #============== this python version should be replaced by our cuda kernel in the furture =============#
        # (k/32*w_bit, n) * (n, m) = (k, m) -> (m, k)
        weights = unpack_qweight(qweight).to(input.dtype)
        grad_input = output_gradient.mm(weights.t()) # (m, n)*(n, k) = (m, k)
        #======================================================================================================#

        # (n, m) * (m, k) = (n, k)
        if qweight.requires_grad: # This additional check is required by peft training.
            qweight.privileged_grad = output_gradient.t().mm(input).t()  # (k, n)

        grad_input = unflatten_x(grad_input, shape)

        # manual release fp weights
        del weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return grad_input, qweight, None, None, None, None, None, None, None, None, None, None


class MBWQLinearCuda(MPQLinearBase):
    """
    Implements a Mixed-BitWidth Quantized (MBWQ) linear layer for CUDA devices. This layer extends the functionality of
    MPQLinearBase by supporting mixed bit-width quantization schemes to optimize model size and computational efficiency
    while running on CUDA-enabled hardware. It allows for flexible quantization configurations across different parts
    of the network, enabling fine-grained control over the trade-off between accuracy and performance.

    Attributes:
        use_mbw (bool): Flag to enable mixed-bitwidth quantization. When set to True, the layer uses different
                        quantization bit widths for different parts of the weight matrix. Defaults to True.
        qweight (torch.Tensor): Quantized weights of the layer, stored with specified bit-widths as per `use_mbw` setting.
        rows (list): A list of 7 elements containing information about the distribution of weights across different
                     bit-widths and kernel permutation information for mixed-bitwidth settings.
        scales (torch.Tensor): Scale factors used for quantization, applicable in mixed-bitwidth mode.
        zeros (torch.Tensor): Zero-point values used for quantization, applicable in mixed-bitwidth mode.

    Methods:
        check_parameters: Validates the layer's parameters to ensure compatibility with the chosen quantization settings.
        load_state_dict: Custom logic to load the layer's state dictionary, handling mixed-bitwidth configurations.
        set_scales: Sets the scale factors for quantization, necessary for initializing or updating the quantized model.
        set_zeros: Sets the zero points for quantization, necessary for initializing or updating the quantized model.
        prepare_params: Prepares the layer's parameters for inference, optimizing memory usage and computational efficiency.
        forward: Defines the forward pass of the layer with quantized weights and possibly mixed bit-width configurations.
        q42fp_weight: reconstructs fp weight from q4-quantized qweight.
        exl2fp_weight: reconstructs fp weight from exl2-quantized qweight.
    """
    def __init__(self, *args, use_mbw: bool=True, groups=64, rows_packed=64, **kwargs) -> None:
        """
        Initializes the MBWQLinearCuda layer with optional mixed-bitwidth quantization.

        Args:
            *args: Variable length argument list to be passed to the base class initializer.
            use_mbw (bool, optional): Specifies whether to use mixed-bitwidth quantization. Defaults to True.
            **kwargs: Arbitrary keyword arguments to be passed to the base class initializer.
        """
        super(MBWQLinearCuda, self).__init__(*args, **kwargs)
        self.qweight.layer_type = 2
        self.use_mbw = use_mbw
        self.groups = groups
        self.rows_packed = rows_packed

        # self.rows will store 7 elements:
        # rows_8=rows[0],rows_6=rows[1],rows_5==rows[2],rows_4=rows[3],rows_3=rows[4],rows_2=rows[5] and kernel_p=rows[6]
        # kernel_p saves permutation information across different bit widths
        # Example: 0b00000010: 2bit, 0b00001110: 2,3,4bit, 0b10111010: 2,4,5,6,8bit
        self.rows = [0] * 7

        self.check_parameters()

    def check_parameters(self) -> None:
        """
        Validates the layer's parameters against the expected constraints for data type and quantization settings.
        """
        assert self.dtype == torch.half, f"The value of dtype ({self.dtype}) must be torch.half."

        self.register_buffer('q_perm', torch.zeros((self.in_channels), dtype=torch.short))
        self.register_buffer('channel_scale', torch.ones((1, 1, self.in_channels), dtype=self.dtype))

        if not self.use_mbw: # only support 4-bit and 2-bit qweight
            # if use mixed-bitwidth quantized, otherwise only support a 4-bit and 2-bit kernel sofar
            assert self.w_bit in [2, 4], f"The value of w_bit ({self.w_bit}) must be 4 or 2."
            assert self.group_size >=32, f"The value of group_size ({self.group_size}) must >= 32."
        else: # mixed-bit width configuration
            sz_shape = (
                math.ceil(self.groups),
                math.ceil(self.out_channels)
            )

            self.qweight = MPQWeightParameter(
                torch.empty((self.rows_packed, self.out_channels),
                            dtype=torch.int32),
                requires_grad=self.requires_grad,
                layer_type=2
            )

            self.register_buffer('q_groups', torch.empty((self.groups*2), dtype=torch.short))
            self.register_buffer('zeros', torch.empty(sz_shape, dtype=self.dtype))
            self.register_buffer('scales', torch.empty(sz_shape, dtype=self.dtype))
            self.q_group_map = None

    def load_state_dict(self, state_dict, strict=True) -> None:
        """
        Custom logic to load the state dictionary, handling special cases for mixed-bitwidth quantization.

        Args:
            state_dict (dict): The state dictionary to load.
            strict (bool): Specifies whether to enforce strict loading of state_dict keys.
        """
        # TODO: this method needs to be tested!!!
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if own_state[name].shape != param.shape:
                    # Handle shape mismatch situations
                    # we directly load and use the exl2 related parameters.
                    if name in ['scales', 'zeros', 'q_perm', 'q_groups', 'q_group_map', 'qweight']:
                        # Processing logic when shapes do not match
                        print(
                            f"Warning: Shape mismatch for: {name}, expected: {own_state[name].shape}, got: {param.shape}. "
                            f"Use the value in state_dict directly.")
                        # ======= Use the value in state_dict directly ======== #
                        own_state[name].data = param.data
                else:
                    # Shape matching, copy value directly
                    own_state[name].copy_(param)
            elif strict:
                raise KeyError(f"Missing key {name} in own state")

        # If strict=False, you can choose to ignore or do other processing for keys that do not exist in the model.
        if not strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if missing:
                print(f"Warning: Missing keys in state_dict: {missing}")

    def set_scales(self, scales: torch.Tensor=None) -> None:
        """
        Sets the scale factors for quantization, necessary for initializing or updating the quantized model.

        Args:
            scales (torch.Tensor, optional): The scale factors to be applied. If None, no action is taken.
        """
        self.scales = scales
        self.qweight.scales = scales

    def set_zeros(self, zeros: torch.Tensor=None) -> None:
        """
        Sets the zero points for quantization, necessary for initializing or updating the quantized model.

        Args:
            zeros (torch.Tensor, optional): The zero points to be applied. If None, no action is taken.
        """
        self.zeros = zeros
        self.qweight.zeros = zeros

    def prepare_params(self) -> None:
        '''
        This method should be executed before the actual forward pass. Prepare for inference
        from memory management and parameter adaptation.

        One can use "prepare_bie_layers" method from project_root.utils.model_helper to call this function.
        '''
        try:
            # set qweight attribute
            self.qweight.scales = self.scales
            self.qweight.zeros = self.zeros
            self.qweight.q_perm = self.q_perm

            height = self.q_perm.size(0)
            groups = self.scales.size(0)
            if self.use_mbw:
                self.qweight.data, self.rows = q_linear_cuda.mbwq_trans_qweight(self.qweight, self.q_groups, True,
                                                                            height, groups, self.w_bit)
                if self.q_group_map is None:
                    self.q_group_map = make_group_map(self.q_groups, self.qweight.shape[0])

                # set qweight attribute
                self.qweight.q_group_map = self.q_group_map
                self.qweight.rows = self.rows
            else:
                dummy_q_group = torch.empty((1), dtype=torch.short, device=self.qweight.device)
                self.qweight.data, rows = q_linear_cuda.mbwq_trans_qweight(self.qweight, dummy_q_group, False,
                                                                            height, groups, self.w_bit)
            # release
            del self.qzeros_zeros
            del self.qzeros_scales
            del self.qscales_zeros
            del self.qscales_scales
            del self.qstatistic

            if self.disable_bias:
                del self.bias
            del self.wf
            del self.g_idx
            torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(f"Error occurred during parameter preparation in MBWQLinearCuda layer: {e}")

    @staticmethod
    def q42fp_weight(qweight: torch.Tensor,
                     scales: torch.Tensor,
                     zeros: torch.Tensor,
                     group_size: int,
                     bits: int,
                     q_perm: torch.Tensor) -> torch.Tensor:
        """
        Converts quantized weights (qweight) from 4-bit and 2-bit quantization back to full-precision (floating-point) weights.

        This function is used for de-quantizing weights that were previously quantized to 4-bit and 2-bit representation,
        applying the necessary scales and zero points for accurate reconstruction.

        Args:
            qweight (torch.Tensor): The quantized weights tensor in 4-bit and 2-bit representation.
            scales (torch.Tensor): The scale factors associated with each quantized weight for conversion back to full-precision.
            zeros (torch.Tensor): The zero points associated with each quantized weight, used during the de-quantization process.
            group_size (int): The size of the weight groups that were quantized together. This parameter is crucial for correctly reshaping the tensor during de-quantization.
            bits (int): 4- or 2-bits
            q_perm (torch.Tensor): A permutation tensor that specifies the order of quantized weights.

        Returns:
            torch.Tensor: The de-quantized (full-precision) weights tensor.
        """
        return q_linear_cuda.mbwq_q42fp_weight(qweight, scales, zeros, group_size, bits, q_perm)

    @staticmethod
    def exl2fp_weight(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                      q_perm: torch.Tensor, q_group_map: torch.Tensor,
                      rows: list) -> torch.Tensor:
        """
        Reconstructs full-precision weights from weights that were quantized using an extended level 2 (exl2) quantization scheme.

        This function handles the de-quantization process for weights that were quantized with a more complex,
        perhaps non-linear, quantization method. It requires multiple parameters including scales, zero points,
        and mappings that define how the quantized values should be translated back to full-precision values.

        Args:
            qweight (torch.Tensor): The quantized weights tensor.
            scales (torch.Tensor): Scale factors for the quantized weights.
            zeros (torch.Tensor): The quantized zero points.
            q_perm (torch.Tensor): A permutation tensor that specifies the order of quantized weights.
            q_group_map (torch.Tensor): A mapping tensor that groups quantized weights.
            rows (list): A list specifying the rows (or indices) of weights to be processed.

        Returns:
            torch.Tensor: The reconstructed full-precision weights tensor.
        """
        return q_linear_cuda.mbwq_exl2fp_weight(qweight, scales, zeros, q_perm, q_group_map, rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MBWQLinearCuda layer with quantized weights and mixed bit-width configurations.

        Args:
            x (torch.Tensor): The input tensor to the layer.

        Returns:
            torch.Tensor: The output tensor resulting from applying the quantized linear transformation.
        """
        x = x.mul(self.channel_scale)

        if not self.use_mbw:
            out = MBWQLinearCudaFunction.apply(x, self.qweight, self.use_mbw, self.training, self.scales, self.zeros,
                                               self.group_size, self.q_perm, self.w_bit, self.privileged_grad)
        else:
            out = MBWQLinearCudaFunction.apply(x, self.qweight, self.use_mbw, self.training, self.scales, self.zeros,
                                               self.group_size, self.q_perm, self.w_bit, self.privileged_grad, self.q_group_map, self.rows)
        if not self.disable_bias:
            out = out + self.bias
        return out
