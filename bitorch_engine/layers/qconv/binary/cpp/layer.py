import torch
from torch.autograd import Function

from bitorch_engine.utils.safe_import import import_extension

binary_conv_cpp = import_extension("binary_conv_cpp")


from bitorch_engine.utils.quant_operators import get_binary_row
from ..layer import BinaryConv2dBase

class BinaryConv2dForward(Function):
    """
    A custom autograd function to perform forward pass of a 2D binary convolution.

    This class implements a static method `forward` to carry out the convolution operation
    using binary weights and activations. The operation is performed using a custom C++
    backend for efficiency.

    Attributes:
        - No class-level attributes.

    Methods:
        - forward: Performs the forward pass of the binary convolution.
    """
    @staticmethod
    def forward(ctx, activations: torch.Tensor, weights: torch.Tensor, m: int, n: int, k: int, kernel_size: int,
                stride: int, padding: int, dilation: int, output_edge: int) -> torch.Tensor:
        """
        Forward pass for the 2D binary convolution.

        Utilizes a C++ backend implemented in `binary_conv_cpp.forward` to perform the operation.
        This method is statically defined and automatically integrated with PyTorch's autograd mechanism.

        Parameters:
            - ctx (torch.autograd.function.BackwardContext): Context object that can be used to stash information
              for backward computation. You can cache arbitrary objects for use in the backward pass using
              the `save_for_backward` method.
            - activations (Tensor): The input feature map or activation tensor.
            - weights (Tensor): The binary weights tensor.
            - m, n, k (int): Dimensions of the input, specifically:
              - m: The number of output channels.
              - n: The number of input channels.
              - k: The spatial size of the output feature map.
            - kernel_size (int or tuple): Size of the conv kernel.
            - stride (int or tuple): Stride of the convolution.
            - padding (int or tuple): Zero-padding added to both sides of the input.
            - dilation (int or tuple): Spacing between kernel elements.
            - output_edge (int): The size of the output edge to ensure the output dimension matches expectations.

        Returns:
            - Tensor: The output feature map resulting from the binary convolution operation.

        Note:
            This method is part of the forward pass and needs to be paired with a corresponding backward
            method to enable gradient computation.
        """
        output = binary_conv_cpp.forward(activations, weights, m, n, k, kernel_size, stride, padding,
                                         dilation, output_edge)
        return output


class BinaryConv2dCPP(BinaryConv2dBase):
    """
    This class implements a binary convolutional layer in PyTorch, specifically optimized with C++ extensions.
    It inherits from BinaryConv2dBase to leverage common binary convolution functionalities with added
    optimizations for efficient computation.

    Attributes:
        bits_binary_word (int): Defines the size of the binary word, defaulting to 8 bits.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the BinaryConv2dCPP layer with the given arguments, which are forwarded to the base class.
        Additionally, it sets up the binary word size for quantization.

        Args:
            *args: Variable length argument list to be passed to the BinaryConv2dBase class.
            **kwargs: Arbitrary keyword arguments to be passed to the BinaryConv2dBase class.
        """
        super(BinaryConv2dCPP, self).__init__(*args, **kwargs)
        self.bits_binary_word = 8

    def prepare_params(self) -> None:
        """
        Prepares and initializes the model parameters for training.
        One can use "prepare_bie_layers" method from project_root.utils.model_helper to call this function.
        """
        pass

    def generate_quantized_weight(self, qweight_only: bool = False) -> None:
        """
        Generates and stores quantized weights based on the current weights of the layer, utilizing a binary
        quantization method. Quantized weights are stored as a torch.nn.Parameter but are not set to require gradients.

        Args:
            qweight_only (bool): If True, the original weights are discarded to save memory. Defaults to False.
        """
        w_size = self.out_channels * self.in_channels/self.bits_binary_word * self.kernel_size * self.kernel_size
        self.qweight = torch.nn.Parameter(
            get_binary_row(self.weight.reshape(-1, ),
                           torch.empty(int(w_size), dtype=torch.uint8),
                           w_size * self.bits_binary_word,
                           self.bits_binary_word),
            requires_grad=False
        )
        if qweight_only:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the binary convolution operation using the quantized weights.

        Args:
            x (torch.Tensor): The input tensor for the convolution operation with shape (N, C_in, H, W),
                              where N is the batch size, C_in is the number of channels, and H, W are the height
                              and width of the input tensor.

        Returns:
            torch.Tensor: The output tensor of the convolution operation with shape determined by the layer's
                          attributes and the input dimensions.
        """
        self._check_forward(x)
        # pass m, n, k
        m = self.out_channels                                       # number of output channel
        k = x.size(dim=1) * self.kernel_size * self.kernel_size; # number of input channels * kernel size
        # (Image_w – filter_w + 2*pad_w) / stride + 1
        output_edge = int((x.size(dim=2) - self.kernel_size + 2 * self.padding) / self.stride + 1)
        n = output_edge * output_edge                               # number of pixels of output images per channel
        return BinaryConv2dForward.apply(x, self.opt_weight, m, n, k, self.kernel_size, self.stride, self.padding,
                                         self.dilation, output_edge)