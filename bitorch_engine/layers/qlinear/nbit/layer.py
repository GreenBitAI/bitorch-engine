import math
from torch import nn
import torch
from torch.nn import init
from bitorch_engine.utils.model_helper import qweight_update_fn


class MPQWeightParameter(nn.Parameter):
    """
    A custom parameter class for quantized weights, extending torch.nn.Parameter,
    with additional attributes specific to quantization.

    Attributes:
        privileged_grad: Optional tensor for privileged gradients (not used in standard backpropagation).
        scales, zeros: Quantization scales and zero points for the affine quantization.
        g_idx: Group index for weight quantization.
        w_bit: Bit-width for weight quantization.
        asym: Flag to indicate if asymmetric quantization is used.
        group_size: The size of quantization groups.
        layer_type: Type of layer (e.g., MPQLinear: 1, MBWQLinear: 2).
        q_perm: Permutation indices for quantization groups.
        qscales_zeros, qscales_scales, qzeros_zeros, qzeros_scales: Additional quantization parameters for calculating
            (q)scales and (q)zeros.
        q_group_map: Mapping from weights to quantization groups.
        rows: Storing rows information for each bit-width in the quantized weight matrix.

    Parameters:
        data (Tensor, optional): Parameter tensor.
        requires_grad (bool, optional): If the parameter requires gradient. Default: True.
        The rest of the parameters are specific to the quantization process and are optional.
    """
    def __new__(cls, data=None,
                requires_grad: bool = True,
                privileged_grad: torch.Tensor = None,
                scales: torch.Tensor = None,
                zeros: torch.Tensor = None,
                g_idx: torch.Tensor = None,
                w_bit: int = -1,
                asym: bool = False,
                group_size: int = -1,
                layer_type: int = -1,
                q_perm: torch.Tensor = None,
                qscales_zeros: torch.Tensor = None,
                qscales_scales: torch.Tensor = None,
                qzeros_zeros: torch.Tensor = None,
                qzeros_scales: torch.Tensor = None,
                q_group_map: torch.Tensor = None,
                rows: list = None
                ):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, data: torch.Tensor=None,
                requires_grad: bool=True,
                privileged_grad: torch.Tensor=None,
                scales: torch.Tensor=None,
                zeros: torch.Tensor=None,
                g_idx: torch.Tensor=None,
                w_bit: int=-1,
                asym: bool=False,
                group_size: int=-1,
                layer_type: int=-1,
                q_perm: torch.Tensor=None,
                qscales_zeros: torch.Tensor=None,
                qscales_scales: torch.Tensor=None,
                qzeros_zeros: torch.Tensor=None,
                qzeros_scales: torch.Tensor=None,
                q_group_map: torch.Tensor=None,
                rows: list=None):
        self.privileged_grad = privileged_grad
        self.scales = scales
        self.zeros = zeros
        self.g_idx = g_idx
        self.w_bit = w_bit
        self.asym = asym
        self.group_size = group_size
        self.layer_type = layer_type # layer_type: MPQLinear: 1, MBWQLinear: 2
        self.q_perm = q_perm
        self.qscales_zeros = qscales_zeros
        self.qscales_scales = qscales_scales
        self.qzeros_zeros = qzeros_zeros
        self.qzeros_scales = qzeros_scales
        self.q_group_map = q_group_map
        self.rows = rows

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
        assert isinstance(qweight, MPQWeightParameter), 'Error: the type of qweight must be ' \
                                                              'MPQWeightParameter. '
        qweight_update_fn(qweight=qweight, exp_avg_s=exp_avg_s, exp_avg_l=exp_avg_l,
                          step=step, lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                          correct_bias=correct_bias, eps=eps, dtype=dtype, projector=projector, grad=grad)


class nBitLinearParameter(torch.nn.Parameter):
    """
    A custom parameter class for n-bit linear layer, extending torch.nn.Parameter.

    This class is designed to support n-bit linear layers, particularly useful
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
        assert isinstance(qweight, nBitLinearParameter), 'Error: the type of qweight must be ' \
                                                              'nBitLinearParameter. '
        qweight_update_fn(qweight=qweight, exp_avg_s=exp_avg_s, exp_avg_l=exp_avg_l,
                          step=step, lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                          correct_bias=correct_bias, eps=eps, dtype=dtype, projector=projector, grad=grad)


class nBitLinearBase(nn.Module):
    """
    A base class for n-bit Quantization-Aware Training (QAT) linear layers. This class provides a framework for
    implementing layers that operate with low-bitwidth activations and weights during training, and supports
    quantization for efficient inference. It maintains both floating-point and quantized weights to facilitate
    the QAT process.

    Attributes:
        in_channels (int): The dimension of input features after bit-packing, indicating the number of input features to the layer.
        out_channels (int): The dimension of output features, indicating the number of output features produced by the layer.
        a_bit (int): The bit-width for activations used during training. Defaults to 4 bits.
        w_bit (int): The bit-width for weights used during training and inference. Defaults to 4 bits.
        device: The device on which the layer's parameters are stored. Defaults to `None`, which means the default device is used.
        dtype: The data type for the layer's parameters. Defaults to `torch.float`.

    Note:
        This class is designed to be subclassed by specific implementations of n-bit linear layers, which should
        provide mechanisms for parameter preparation (`prepare_params`), weight quantization (`generate_quantized_weight`),
        and other necessary operations.
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 a_bit: int = 4,
                 w_bit: int = 4,
                 device=None,
                 dtype=torch.float) -> None:
        super(nBitLinearBase, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.dtype = dtype
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.weight = None
        self.qweight = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes or resets the floating-point weights of the layer using Kaiming uniform initialization.
        """
        self.weight = torch.nn.Parameter(
            torch.Tensor(self.out_channels, self.in_channels))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def set_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the floating-point weights of the layer to the provided tensor.

        Args:
            x (torch.Tensor): The tensor to set as the new weights.
        """
        self.weight = nn.Parameter(x, requires_grad=False)

    def prepare_params(self) -> None:
        """
        Prepares and initializes the model parameters for training.

        Note:
            This method MUST be called after model initialization and before training starts to ensure the weights are
            properly prepared for efficient computation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def set_quantized_weight_data(self, x: torch.Tensor) -> None:
        """
        Sets the quantized weights of the layer to the provided tensor.

        Args:
            x (torch.Tensor): The tensor to set as the new quantized weights.
        """
        self.qweight = nn.Parameter(x, requires_grad=False)

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Generates and sets the quantized weights based on the current floating-point weights. This method must be
        implemented by subclasses and is crucial for converting floating-point weights to low-bitwidth quantized weights
        for inference.

        Args:
            qweight_only (bool): If `True`, only quantized weights are generated.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _check_forward(self, x: torch.Tensor) -> None:
        """
        A placeholder method for checking the inputs to the forward pass. This method must be implemented by subclasses
        to ensure the input tensor is suitable for processing by the layer.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def opt_weight(self) -> torch.nn.Parameter:
        """
        Returns the optimal weight for the current mode (training or inference). If the model is in inference mode
        and quantized weights are not yet generated, it triggers quantized weight generation.

        Returns:
            torch.nn.Parameter: The current optimal weights (floating-point for training, quantized for inference).
        """
        if not self.training and self.qweight is None:
            self.generate_quantized_weight()
        return self.weight if self.training else self.qweight


class MPQLinearBase(nn.Module):
    """
    Base class for mixed precision quantized (MPQ) linear layers, designed to support
    the computational needs of large language models (LLMs) with mixed precision quantization,
    such as 16-bit activations and 4-bit weights for efficient inference. It introduces optimized
    computation for bitwise unpacking of quantized weights and 16-bit floating-point matrix multiplication,
    tailored for various hardware platforms.

    Different to nBitLinearBase, MPQLinearBase serves as the base class for mixed precision quantized linear layers.
    This special class is mainly to support the mixed precision linear layer in the current LLMs model,
    such as using 16-bit activation and 4-bit quantization weight for inference.
    During the reasoning process, two main calculation processes are introduced, namely bitwise unpacking of qweight
    from lower bits to 16-bit float, and 16-bit matrix multiplication. Correspondingly, the performance of these
    two processes has been optimized on different hardware.

    Attributes:
        in_channels (int): The number of input features after bit-packing, representing the dimensionality
                           of the input space.
        out_channels (int): The number of output features, representing the dimensionality of the output space.
        a_bit (int): The bit-width used for activation quantization, defaulting to 16 bits for high precision.
        w_bit (int): The bit-width used for weight quantization, aiming to reduce memory footprint and computational cost.
        dtype (torch.dtype): The data type for computations within this layer, typically torch.half for efficiency.
        group_size (int): The grouping size for quantization, affecting scale and zero-point calculation.
                          A value of -1 indicates that the entire input width is treated as one group.
        use_gba_quant (bool): Flag to indicate the use of GBA-specific quantization techniques over GPTQ-compliant methods.
        dq_group_size (int): Double quantization group size, specific to GBA quantization, for further granularity in quantization.
        dq_mode (int): Double quantization mode, catering to different versions and requirements of LLaMA models.
        disable_bias (bool): Whether to include a bias term in the linear calculation. Disabling can reduce parameters and computation.
        asym (bool): Indicates whether asymmetric quantization is used, offering an alternative to symmetric quantization strategies.

    Methods:
        initialize(): Initializes parameters and quantization buffers based on the selected quantization method.
        init_gptq(): Sets up parameters specific to GPTQ quantization.
        init_gba(): Configures buffers and scales for GBA quantization, accommodating for asymmetry and double quantization modes.
        set_qweight_data(data): Updates the quantized weight tensor with new data.
        generate_quantized_weight(): Placeholder for weight quantization method, to be implemented by subclasses.
        check_parameters(): Placeholder for parameter validation, ensuring correct layer configuration.
        prepare_params(): Prepares quantized parameters for the forward pass, potentially decompressing quantized values.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 a_bit: int = 16,
                 w_bit: int = 4,
                 dtype=torch.half,
                 group_size=-1,
                 use_gba_quant=True,
                 dq_group_size=-1,
                 dq_mode=2,
                 disable_bias=True,
                 asym=False,
                 requires_grad=True) -> None:
        """
        Args:
            in_channels (int): dim of input features after bit-packing
            out_channels (int): dim of hidden states
            a_bit: activation bits
            w_bit: weight bits
            dtype: data type used in this layer
            group_size: number of associated weight elements->scale and zero facter
            disable_bias: whether use bias
            use_gba_quant: True: GBA specific quantization, False: use GPTQ-compliant methods
            dq_group_size: gba specific parameter. Indicates double quantization group size.
            dq_mode: gba specific parameter. Indicates double quantization mode, which is used to adapt to multiple different LLaMA versions.
            asym: gba specific parameter. Indicates asymmetry or symmetry quantization strategies.
            requires_grad (bool): Indicates whether gradient calculation should be enabled for the parameters.
        """
        super(MPQLinearBase, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.dtype = dtype
        self.maxq = 2 ** self.w_bit - 1
        self.group_size = group_size if group_size > -1 else self.in_channels
        self.asym = asym
        self.disable_bias = disable_bias
        self.use_gba_quant = use_gba_quant
        self.dq_group_size = dq_group_size
        self.requires_grad = requires_grad
        self.dq_mode = dq_mode
        self.initialize()

    def initialize(self) -> None:
        """
        Initializes layer parameters and quantization buffers. This method sets up the infrastructure
        for either GBA or GPTQ quantization methods, based on the layer configuration. It allocates
        memory for quantized weights, scales, zero-points, and other necessary buffers, ensuring
        they are ready for the quantization process.
        """
        ## Both GPTQ and GBA methods require qweights, scales, zeros and group_index
        # trainable params
        self.qweight = MPQWeightParameter(
            data = torch.empty((self.in_channels // 32 * self.w_bit, self.out_channels),
                        dtype=torch.int32),
            requires_grad=self.requires_grad,
            w_bit=self.w_bit,
            asym=self.asym,
            group_size=self.group_size,
        )
        self.privileged_grad = torch.empty((self.in_channels, self.out_channels), dtype=self.dtype) if self.requires_grad else None

        # non-trainable params
        self.register_buffer('g_idx', torch.tensor([i // self.group_size for i in range(self.in_channels)],
                                                   dtype=torch.int32))
        # still need to initialize for the checkpoint loading. Unused params will be released in "prepare_params()"
        self.register_buffer('bias', torch.zeros((self.out_channels), dtype=self.dtype))
        self.register_buffer("wf", torch.tensor(list(range(0, 32, self.w_bit)), dtype=torch.int32).unsqueeze(0))

        # init
        if self.use_gba_quant:
            self.init_gba()
        else:
            self.init_gptq()

    def init_gptq(self) -> None:
        """
        Initializes parameters and buffers specific to the GPTQ quantization method. This includes
        setting up zero-point buffers, scale factors, and ensuring asymmetric quantization is enabled.
        GPTQ, being a more general quantization approach, requires specific buffers to hold quantization
        parameters for accurate computation and minimal precision loss.
        """
        self.register_buffer('qzeros', torch.zeros((math.ceil(self.in_channels / self.group_size),
                                                    self.out_channels // 32 * self.w_bit), dtype=torch.int32))
        self.register_buffer('scales', torch.ones((math.ceil(self.in_channels / self.group_size),
                                                   self.out_channels), dtype=self.dtype))
        self.asym = True

    def init_gba(self) -> None:
        """
        Prepares the layer for GBA-specific quantization, configuring buffers for scales, zero-points,
        and statistics for double quantization if enabled. GBA quantization allows for fine-tuned control
        over the quantization process, accommodating asymmetric quantization and providing additional
        parameters to adjust for different model versions and requirements.
        """
        # non-trainable params
        if self.dq_group_size == -1:
            self.dq_group_size = self.out_channels

        buffer_shape_1 = (
            math.ceil(self.in_channels / self.group_size),
            math.ceil(self.out_channels / self.dq_group_size),
            1
        )
        buffer_shape_2 = (
            math.ceil(self.in_channels / self.group_size),
            math.ceil(self.out_channels / self.dq_group_size),
            self.dq_group_size
        )

        if self.asym:
            self.register_buffer('qzeros', torch.zeros((math.ceil(self.in_channels / self.group_size),
                                                        self.out_channels // 32 * self.w_bit), dtype=torch.int32))
            if self.w_bit == 4:
                self.register_buffer('qscales', torch.ones(buffer_shape_2, dtype=torch.uint8))
            else:
                self.register_buffer('qscales', torch.ones((math.ceil(self.in_channels / self.group_size),
                                                            self.out_channels), dtype=torch.uint8))
        else:
            self.register_buffer('qstatistic', torch.ones(buffer_shape_2, dtype=torch.uint8))
            self.register_buffer('qzeros_zeros', torch.zeros(buffer_shape_1, dtype=self.dtype))
            self.register_buffer('qzeros_scales', torch.ones(buffer_shape_1, dtype=self.dtype))

        if self.dq_mode == 1:
            self.register_buffer('qscales_zeros', torch.zeros((1, self.out_channels, 1), dtype=self.dtype))
            self.register_buffer('qscales_scales', torch.ones((1, self.out_channels, 1), dtype=self.dtype))
        else:
            self.register_buffer('qscales_zeros', torch.zeros(buffer_shape_1, dtype=self.dtype))
            self.register_buffer('qscales_scales', torch.ones(buffer_shape_1, dtype=self.dtype))

        self.register_buffer('scales', torch.ones((math.ceil(self.in_channels / self.group_size), self.out_channels),
                                                  dtype=self.dtype))
        self.register_buffer('zeros', torch.zeros((math.ceil(self.in_channels / self.group_size), self.out_channels),
                                                  dtype=self.dtype))

    def set_qweight_data(self, data: torch.Tensor) -> None:
        """
        Updates the quantized weight tensor with new data. This method is crucial for adjusting the quantized
        weights based on training or fine-tuning processes, ensuring the layer's weights reflect the most
        recent updates.

        Args:
            data (torch.Tensor): The new quantized weight data to be set in the layer.
        """
        self.qweight.data = data

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        A placeholder method for the weight quantization process. Subclasses should implement this method
        to define how the layer's weights are quantized based on the current configuration and quantization
        method. This operation is typically executed before saving the model weights or performing inference to ensure that the weights
        are in the appropriate quantized format.

        Args:
            qweight_only (bool): A flag to indicate whether only the quantized weights need to be generated,
                                 without considering other quantization parameters like scales or zero-points.
                                 Default is False, which means all relevant quantization parameters are generated.
        """
        raise NotImplementedError("this method has not been implemented.")

    def check_parameters(self) -> None:
        """
        Validates the configuration and parameters of the layer to ensure they are set correctly for the
        quantization process. This method should check for common configuration errors and ensure that all
        required parameters for the selected quantization method are correctly initialized.

        Raises:
            NotImplementedError: Indicates that the method has not been implemented yet and needs to be
                                 provided by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def prepare_params(self) -> None:
        '''
        This method should be executed before the actual forward pass. It mainly decompress quantized parameters
        such as qscale and qzero. This step could be simplified or eliminated in the future by having a kernel
        implementation that can decompress during kernel computation.

        One can use "prepare_bie_layers" method from project_root.utils.model_helper to call this function.

        Note:
            This method should be called before executing the forward pass, especially after loading
            the model from a checkpoint or before inference to ensure that quantized parameters are
            correctly prepared.

        Raises:
            NotImplementedError: Indicates that the method has not been implemented yet and should be
                                 provided by subclasses.
        '''
        raise NotImplementedError("Subclasses should implement this method.")
