from typing import Tuple

import torch


def nv_tensor_quant(inputs, amax=None, num_bits=8, unsigned=False, narrow_range=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the given tensor using specified quantization parameters. This method supports
    both signed and unsigned quantization with an option for narrow range quantization.
    This function is shared between TensorQuantFunction and FakeTensorQuantFunction.

    Author: nv_pytorch_quantization
    Source: https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/pytorch_quantization/tensor_quant.py#L315

    Args:
        inputs (torch.Tensor): The input tensor to be quantized.
        amax (torch.Tensor or None): The maximum absolute value used for quantization scaling. If None, it will be
                                      calculated from the input tensor.
        num_bits (int): Number of bits to use for quantization, default is 8.
        unsigned (bool): Flag indicating if the quantization is unsigned, default is False.
        narrow_range (bool): Flag indicating if the quantization should use narrow range, default is True.

    Raises:
        ValueError: If `amax` has a different shape than `inputs` or contains negative values.
        TypeError: If negative values are encountered in unsigned quantization mode.

    Returns:
        torch.Tensor: The quantized tensor.
        torch.Tensor: The scale factor used for quantization.

    Note:
        - Quantization is performed in FP32 to avoid overflow.
        - If `inputs` or `amax` are in FP16 or BF16, they are converted to FP32 for calculation.
        - The quantization range is adjusted based on `unsigned` and `narrow_range` flags.
        - Special handling for `amax` values smaller than the minimum representable value of FP16.
    """
    if isinstance(amax, torch.Tensor) and inputs.dim() != amax.dim():
        raise ValueError(
            "amax %s has different shape than inputs %s. Make sure broadcast works as expected!",
            amax.size(),
            inputs.size(),
        )

    # print("{} bits quantization on shape {} tensor.".format(num_bits, inputs.size()))

    if amax == None:
        amax = torch.amax(inputs, keepdim=True)

    if unsigned:
        if inputs.min() < 0.0:
            raise TypeError("Negative values encountered in unsigned quantization.")

    # Computation must be in FP32 to prevent potential over flow.
    input_dtype = inputs.dtype
    if inputs.dtype == torch.bfloat16 or inputs.dtype == torch.float16:
        inputs = inputs.float()
    if amax.dtype == torch.bfloat16 or amax.dtype == torch.float16:
        amax = amax.float()

    min_amax = amax.min()
    if min_amax < 0:
        raise ValueError("Negative values in amax")

    max_bound = torch.tensor(
        (2.0 ** (num_bits - 1 + int(unsigned))) - 1.0, device=inputs.device
    )
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1
    scale = max_bound / amax

    outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

    epsilon = 1.0 / (1 << 24)
    if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
        zero_amax_mask = amax <= epsilon
        scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0
    if min_amax <= epsilon:
        scale[
            zero_amax_mask
        ] = 1.0  # Return 1 makes more sense for values quantized to 0 with amax=0

    if input_dtype == torch.bfloat16 or input_dtype == torch.float16:
        outputs = outputs.to(input_dtype)

    return outputs, scale


def bit_set(var, pos, val):
    """
    Sets a specific bit in an integer variable to a given value.

    This method allows you to modify a single bit within an integer by shifting the `val` (either 0 or 1) to the position `pos` and then performing a bitwise OR operation with the original variable `var`. This effectively sets the bit at position `pos` to the value specified by `val`.

    The operation performed is equivalent to:
    `var |= (val << pos)`

    Parameters:
        var (int): The original integer variable whose bit is to be modified.
        pos (int): The position of the bit to be set, starting from 0 for the least significant bit (LSB).
        val (int): The new value for the bit, either 0 or 1.

    Returns:
        int: The modified integer with the bit at position `pos` set to `val`.

    Example:
        >>> bit_set(0b0010, 1, 1)
        6  # The binary representation is 0b0110
    """
    var |= val << pos
    return var


def get_binary_row(nd_row, binary_row, nd_size, bits_per_binary_word):
    """
    Binarizes an input NDArray (nd_row) into a binary representation (binary_row) based on the specified number of bits per binary word (bits_per_binary_word).

    This function iteratively processes each segment of the input array with the length of 'bits_per_binary_word', converting each segment into a binary word.
    Each bit in the binary word represents the sign (positive or negative) of the corresponding element in the input array segment.

    Specifically, for each segment:
        - A binary word ('rvalue') is initialized to 0.
        - For each element in the segment, if the element is non-negative, the corresponding bit in 'rvalue' is set to 1; otherwise, it remains 0.
        - The binary word is then stored in 'binary_row' at the position corresponding to the segment index.

    Parameters:
        nd_row (array-like): The input array to be binarized.
        binary_row (array-like): The output array where each element is a binary word representing a segment of 'nd_row'.
        nd_size (int): The size of the 'nd_row' array.
        bits_per_binary_word (int): The number of bits in each binary word, determining the segment size for binarization.

    Returns:
        array-like: The binarized representation of 'nd_row' stored in 'binary_row'.

    Example of equivalent C++ logic:

    .. code-block::

        for (int i = 0; i < size; i+=BITS_PER_BINARY_WORD) {
          BINARY_WORD rvalue=0;
          BINARY_WORD sign;
          for (int j = 0;j < BITS_PER_BINARY_WORD; ++j) {
            sign = (row[i+j]>=0);
            BIT_SET(rvalue, j, sign);
          }
          b_row[i/BITS_PER_BINARY_WORD] = rvalue;
        }
    """
    i = 0
    while i < nd_size:
        rvalue = 0
        j = 0
        while j < bits_per_binary_word:
            sign = 0
            if nd_row[i + j] >= 0:
                sign = 1
            rvalue = bit_set(rvalue, j, sign)
            j += 1

        # print('{0:64b}'.format(rvalue))

        binary_row[int(i / bits_per_binary_word)] = rvalue

        # print('{0:64b}'.format(binary_row[int(i/bits_per_binary_word)]))
        # testing stuff
        # d = mx.nd.array(binary_row, dtype="float64")
        # print('{0:64b}'.format(int(d.asnumpy()[int(i/bits_per_binary_word)])))
        i += bits_per_binary_word
    return binary_row


def get_binary_col(nd_col, binary_col, dim_n, dim_k, bits_per_binary_word):
    """
    Binarizes an array column-wise, transforming each element into a binary representation.

    This function is a Python re-implementation of an equivalent C++ version. It operates on
    a columnar slice of an array, encoding each segment of BITS_PER_BINARY_WORD bits into
    a binary word, where each bit is determined by the sign (positive or non-negative vs. negative)
    of the corresponding element in the input array.

    The binarization process proceeds by iterating over the array in blocks of BITS_PER_BINARY_WORD,
    setting each bit based on the sign of the corresponding element. The result is stored in a
    pre-allocated array for binary representations.

    Args:
        nd_col (array-like): The input array containing numerical values to be binarized.
        binary_col (array-like): Pre-allocated array where the binary representations are stored.
        dim_n (int): The size of the dimension over which to iterate, typically the number of rows in the array.
        dim_k (int): The size of the second dimension, typically the number of columns.
        bits_per_binary_word (int): The number of bits in each binary word, determining the block size for binarization.

    Returns:
        array-like: The modified binary_col array containing the binary representations of the input array, column-wise.

    Example of equivalent C++ logic:

    .. code-block::

        for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
            for(int x=0; x < k; ++x){
                BINARY_WORD rvalue=0;
                BINARY_WORD sign;
                for(int b=0; b<BITS_PER_BINARY_WORD; ++b){
                    sign = (col[(y*BITS_PER_BINARY_WORD+b)*k + x]>=0);
                    BIT_SET(rvalue, b, sign);
                }
                b_col[y*k + x] = rvalue;
            }
        }
    """
    y = 0
    while y < int(dim_n / bits_per_binary_word):
        x = 0
        while x < dim_k:
            rvalue = 0
            b = 0
            while b < bits_per_binary_word:
                sign = 0
                if nd_col[(y * bits_per_binary_word + b) * dim_k + x] >= 0:
                    sign = 1
                rvalue = bit_set(rvalue, b, sign)
                b += 1
            binary_col[y * dim_k + x] = rvalue
            x += 1
        y += 1

    return binary_col


def q8_quantization(input: torch.Tensor, scale_a: torch.Tensor=None, eps: torch.Tensor=None) -> torch.Tensor:
    """
    Quantizes an input tensor to 8-bit integers using uniform quantization.

    The function first ensures that the input tensor is of floating-point type.
    It then adjusts the scale factor `scale_a` to avoid division by values too close to zero,
    applying a lower threshold defined by `eps`. The quantization process scales the input tensor
    by the inverse of `scale_a`, rounds the result to the nearest integer, and clamps the values
    to the 8-bit range [-128, 127].

    Args:
        input (torch.Tensor): The input tensor to be quantized. Should ideally be of floating-point type.
        scale_a (torch.Tensor): The scale factor for quantization. Each element in `scale_a`
                                scales the corresponding element in `input`.
        eps (torch.Tensor): A small positive tensor used to prevent division by zero or values
                            too close to zero in the scale factor.

    Returns:
        torch.Tensor: The quantized tensor, with values rounded and clamped to fit within
                      the 8-bit integer range.
    """
    is_scale_none = scale_a is None
    if input.dtype != torch.float:
        input = input.to(torch.float)
    if scale_a is None:
        scale_a = 2 * input.abs().mean() / 11.269
    if eps is None:
        eps = torch.tensor(0.00001).type(input.dtype).device(input.device)

    scale_a = torch.where(scale_a > eps, scale_a, eps)
    Qn = -128
    Qp = 127
    if is_scale_none:
        return (input / scale_a).round().clamp(Qn, Qp), scale_a
    else:
        return (input / scale_a).round().clamp(Qn, Qp)


def q4_quantization(input: torch.Tensor, scale_a: torch.Tensor=None, eps: torch.Tensor=None) -> torch.Tensor:
    """
    Quantizes an input tensor to 4-bit integers using uniform quantization.

    The function first ensures that the input tensor is of floating-point type.
    It then adjusts the scale factor `scale_a` to avoid division by values too close to zero,
    applying a lower threshold defined by `eps`. The quantization process scales the input tensor
    by the inverse of `scale_a`, rounds the result to the nearest integer, and clamps the values
    to the 4-bit range [-8, 7].

    Args:
        input (torch.Tensor): The input tensor to be quantized. Should ideally be of floating-point type.
        scale_a (torch.Tensor): The scale factor for quantization. Each element in `scale_a`
                                scales the corresponding element in `input`.
        eps (torch.Tensor): A small positive tensor used to prevent division by zero or values
                            too close to zero in the scale factor.

    Returns:
        torch.Tensor: The quantized tensor, with values rounded and clamped to fit within
                      the 4-bit integer range.
    """
    is_scale_none = scale_a is None
    if input.dtype != torch.float:
        input = input.to(torch.float)
    if scale_a is None:
        scale_a = 2 * input.abs().mean() / 5.6345  # Adjusted scale calculation for 4-bit
    if eps is None:
        eps = torch.tensor(0.00001).type(input.dtype).device(input.device)

    scale_a = torch.where(scale_a > eps, scale_a, eps)
    Qn = -8
    Qp = 7
    if is_scale_none:
        return (input / scale_a).round().clamp(Qn, Qp), scale_a
    else:
        return (input / scale_a).round().clamp(Qn, Qp)


def gptq_stype_unpacking(qweight) -> torch.Tensor:
    """
    Reconstructs the fp16 weight tensor from the input quantized weight parameter in GPTQ style.

    Parameters:
        qweight: The quantized weight parameter object containing all necessary quantization information.

    Returns:
        torch.Tensor: The reconstructed weight tensor in fp16 format.
    """

    wf = torch.tensor(list(range(0, 32, qweight.w_bit)), dtype=torch.int32, device=qweight.device).unsqueeze(0)
    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // qweight.w_bit, -1),
                   wf.unsqueeze(-1)).to(torch.int16 if qweight.w_bit == 8 else torch.int8).view(-1, qweight.size(-1))
    torch.bitwise_and(weight, (2 ** qweight.w_bit) - 1, out=weight)

    if qweight.asym:
        zeros_unpack = torch.bitwise_right_shift(torch.unsqueeze(qweight.zeros, 2).expand(-1, -1, 32 // qweight.w_bit),
                             wf.unsqueeze(0)).to(torch.int16 if qweight.w_bit == 8 else torch.int8)
        torch.bitwise_and(zeros_unpack, (2 ** qweight.w_bit) - 1, out=zeros_unpack)
        zeros_unpack = zeros_unpack + 1
        zeros = zeros_unpack.reshape(-1, qweight.size(-1))

        weights = qweight.scales[qweight.g_idx.long()] * (weight - zeros[qweight.g_idx.long()])
    else:
        # 2. GPTQ style without g_index.
        if qweight.g_idx is None:
            scales = qweight.scales.unsqueeze(1).repeat(1, weight.size(0)//qweight.scales.size(0), 1).view(-1, qweight.scales.size(-1))
            zeros = qweight.zeros.unsqueeze(1).repeat(1, weight.size(0) // qweight.zeros.size(0), 1).view(-1, qweight.zeros.size(-1))
            weights = weight.mul(scales) - zeros
            q_perm = qweight.q_perm.unsqueeze(1).repeat(1, weights.size(1)).long()
            weights.scatter_(dim=0, index=q_perm, src=weights.clone())
        else:
            weights = weight * qweight.scales[qweight.g_idx.long()] - qweight.zeros[qweight.g_idx.long()]

    return weights