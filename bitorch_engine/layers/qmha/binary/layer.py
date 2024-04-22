from typing import Tuple

import torch
import torch.nn as nn
import math

from bitorch_engine.layers.qlinear.binary.cutlass import BinaryLinearCutlass, BinaryMatMul

class LearnableBias(nn.Module):
    """
    A module that introduces a learnable bias term to the input tensor.

    This module adds a learnable bias parameter to the input tensor along the channel dimension.
    The bias parameter is learnable and can be optimized during the training process.

    Attributes:
        bias (nn.Parameter): The learnable bias parameter initialized with zeros.

    Args:
        out_chn (int): The number of output channels. This should match the channel dimension
                       of the input tensor to which the bias will be added.
    """
    def __init__(self, out_chn: int):
        """
        Initializes the LearnableBias module.

        Args:
            out_chn (int): The number of output channels for the bias parameter.
        """
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LearnableBias module.

        Adds the learnable bias to the input tensor.

        Args:
            x (Tensor): The input tensor to which the learnable bias will be added.

        Returns:
            Tensor: The output tensor after adding the learnable bias to the input.
        """
        out = x + self.bias.expand_as(x)
        return out


class BMHA(torch.nn.Module):
    '''
    Implements a binary version of multi-head attention (MHA) where the linear transformations
    are executed using binary operations to improve efficiency. This class is designed to work
    with binary weights and can be particularly useful for deployments in resource-constrained
    environments or for models where computational efficiency is crucial.

    Attributes:
        dtype (torch.dtype): Data type for the computations, typically float or binary.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        hidden_dim (int): Dimension of the hidden layer.
        input_dim (int): Dimension of the input layer.
        q_linear (BinaryLinearCutlass): Linear transformation for the query vector using binary operations.
        v_linear (BinaryLinearCutlass): Linear transformation for the value vector using binary operations.
        k_linear (BinaryLinearCutlass): Linear transformation for the key vector using binary operations.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
        out (BinaryLinearCutlass): Final linear layer to project the attention output back to input dimensionality.

    Raises:
        ValueError: If the hidden size is not a multiple of the number of attention heads.

    Note: The implementation of this class is still in the EXPERIMENTAL STAGE!
    '''
    def __init__(self, input_dim: int, hidden_dim: int, num_heads:int, dtype: torch.Tensor=torch.float, *args, **kwargs):
        """
        Initializes the BMHA module with the specified parameters. It sets up the binary linear layers for
        the queries, keys, and values, along with the output projection layer. It also validates that the
        hidden dimension is evenly divisible by the number of heads to ensure that the dimensions align
        properly for multi-head attention.

        Args:
            input_dim (int): The size of each input vector.
            hidden_dim (int): The size of the hidden dimension. Must be divisible by num_heads.
            num_heads (int): The number of attention heads.
            dtype (torch.dtype, optional): The data type for computations. Defaults to torch.float.
            *args: Variable length argument list for the parent class.
            **kwargs: Arbitrary keyword arguments for the parent class.

        Raises:
            ValueError: If hidden_dim is not divisible by num_heads, an error is raised to alert the user.
        """
        super(BMHA, self).__init__(*args, **kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.dtype = dtype
        self.num_heads = num_heads
        self.head_dim = int(hidden_dim / num_heads)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.q_linear = BinaryLinearCutlass(input_dim, hidden_dim, dtype=dtype)
        self.v_linear = BinaryLinearCutlass(input_dim, hidden_dim, dtype=dtype)
        self.k_linear = BinaryLinearCutlass(input_dim, hidden_dim, dtype=dtype)
        self.dropout = nn.Dropout(0.1)
        self.out = BinaryLinearCutlass(hidden_dim, input_dim, dtype=dtype)

        ## uncomment the following two lines, if you want to try binary matmul.
        ##
        # self.cutlass_matmul_1 = BinaryMatMul(dtype=dtype)
        # self.cutlass_matmul_2 = BinaryMatMul(dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the binary multi-head attention layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
            mask (torch.Tensor, optional): Optional mask tensor to exclude certain positions from
                                           the attention mechanism. Shape (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Output tensor of shape (batch_size, sequence_length, input_dim).
                - Attention scores tensor of shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        # get batch size
        bs = hidden_states.size(0)
        # Linear projections -> (bs, num_head, seq_length, head_dim)
        q = self.q_linear(hidden_states).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_linear(hidden_states).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_linear(hidden_states).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        # Attention output
        attention = torch.matmul(scores, v)
        attention = attention.transpose(1, 2).contiguous().view(bs, -1, self.hidden_dim)

        # Final linear layer
        output = self.out(attention)
        return output, scores


        # ======================================================================================================= #
        # The version using binary matmul. NOTE THAT the batched_gemm of cutlass is not very efficient, we didn't #
        # observe speedup for smaller matrices. More efficient CUDA implementation for binary matmul is required  #
        # ======================================================================================================= #
        # # get batch size
        # bs = hidden_states.size(0)
        # # Linear projections -> (bs, num_head, seq_length, head_dim)
        # q = self.q_linear(hidden_states).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        # k = self.k_linear(hidden_states).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        # v = self.v_linear(hidden_states).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        #
        # # shifting with learnable bias
        # q = self.move_q(q)
        # k = self.move_k(k)
        # v = self.move_v(v)
        #
        # # Scaled Dot-Product Attention
        # # this matmul accepts x_shape:(..., m, k), y_shape(..., n, k), the function will transpose the second input
        # scores = self.cutlass_matmul_1(q, k) / math.sqrt(self.head_dim)
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        #     scores = scores.masked_fill(mask == 0, -1e9)
        # scores = nn.functional.softmax(scores, dim=-1)
        # scores = self.dropout(scores)
        #
        # # Attention output
        # attention = self.cutlass_matmul_2(v.transpose(-1, -2), scores)
        # attention = attention.transpose(1, 2).contiguous().view(bs, -1, self.hidden_dim)
        #
        # # Final linear layer
        # output = self.out(attention)
        # return output, scores