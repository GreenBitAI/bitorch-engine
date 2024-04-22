import torch

from bitorch_engine.utils.safe_import import import_extension

functions_cuda = import_extension("functions_cuda")


def fp32toint4(input: torch.Tensor) -> torch.Tensor:
    """
    Converts a 32-bit floating point tensor to a 4-bit integer representation.

    This function takes an input tensor of floating point numbers and compresses it
    into a tensor of 4-bit integers, effectively reducing the memory footprint by a factor of 8.
    The conversion process involves finding the minimum and maximum values of the input
    to normalize the data range, and then quantizing the normalized values into 4-bit integers.

    Parameters:
        - input (Tensor): A tensor of 32-bit floating point numbers that we want to compress.

    Returns:
        - Tensor: A tensor of 4-bit integers representing the quantized version of the input tensor.
          The output tensor uses a 64-bit integer data type to store the 4-bit values,
          with each 64-bit integer holding sixteen 4-bit values.

    Note:
        - The input tensor is assumed to be a flat 1D tensor, and the output tensor will also be a 1D tensor.
        - This function is designed to be executed on CUDA-enabled devices and utilizes custom CUDA kernels
          for the quantization process.
        - The function allocates temporary memory on the GPU for intermediate computations, which is
          freed before returning the output tensor.
    """
    return functions_cuda.fp32toint4(input)


def tensor_to_packed_uint8(input: torch.Tensor) -> torch.Tensor:
    """
    Packs the given tensor into an 8-bit unsigned integer tensor representation on CUDA.

    This function converts a tensor of various data types (int8, float32, bfloat16, half)
    to an 8-bit unsigned integer tensor using CUDA kernels for efficient computation.
    It ensures that the operation is performed on the correct CUDA device by employing
    a device guard based on the input tensor's device. If the data type of the input tensor
    is not supported, the function will terminate the program with an error message.

    Parameters:
        input (torch.Tensor): The input tensor to be packed. Can be of type int8, float32,
                             bfloat16, or half.

    Returns:
        torch.Tensor: An 8-bit unsigned integer representation of the input tensor.

    Note:
        The input tensor must be on a CUDA device.
        The function terminates the program if the input tensor's type is not supported.
    """
    return functions_cuda.tensor_pack_to_uint8(input)


def unpack_uint8_tensor(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Unpacks an 8-bit unsigned integer tensor into a floating-point tensor using CUDA,
    scaling the unpacked values by a provided scale tensor.

    This function is a Python interface to the CUDA backend that performs the unpacking
    and scaling operation. It is designed to work with tensors that represent compressed
    data in 8-bit unsigned integer format and converts them into a higher precision
    floating-point format. This is useful in scenarios where compact data representations
    need to be expanded and processed in their original or higher precision for further
    computations or analysis.

    Parameters:
        - input (torch.Tensor): A tensor of 8-bit unsigned integers with shape
          (batch_size, sequence_length, packed_embedding_dimension). This tensor represents
          the compressed embeddings or data to be unpacked.
        - scale (torch.Tensor): A tensor with shape (batch_size, sequence_length, 1) that
          contains scaling factors for each sequence in the batch. These factors are applied
          to the unpacked floating-point values.

    Returns:
        - torch.Tensor: A tensor of floating-point (-1.0, 1.0) values with shape
          (batch_size, sequence_length, packed_embedding_dimension * 8), where the unpacked
          and scaled embeddings or data are stored.

    The function directly interfaces with a CUDA implementation for efficient processing
    on GPUs, offering significant speedups compared to CPU-based operations.
    """
    return functions_cuda.uint8_to_unpacked_tensor(input, scale)


def q4_pack_tensor(input: torch.Tensor, is_transpose: bool = False) -> torch.Tensor:
    """
    Packs a tensor into a 4-bit packed format using CUDA accelerated functions.

    This function takes an input tensor and optionally transposes it before packing.
    The packing process reduces the storage requirement by representing each value
    in the tensor with only 4 bits. This is particularly useful for quantized neural
    network weights and other scenarios where precision can be traded for storage
    efficiency without significantly affecting the application's performance.

    The actual packing is performed by a CUDA-accelerated function for efficiency,
    making this function suitable for large tensors.

    Args:
        input (torch.Tensor): The input tensor to be packed. This tensor should be
                              in a compatible format (int32) where each value
                              can be represented in 4 bits.
        is_transpose (bool): If True, the tensor will be transposed before packing.
                             This is useful if the packed tensor needs to be in a
                             specific orientation for subsequent operations.

    Returns:
        torch.Tensor: A tensor containing the 4-bit packed representation of the
                      input tensor. The returned tensor will have a dtype of int8
                      and potentially half the number of elements in the last
                      dimension of the input tensor if `is_transpose` is False.
                      If `is_transpose` is True and the transposition changes the
                      tensor's shape, the returned tensor's shape will be adjusted
                      accordingly.
    """
    assert input.dtype == torch.int32, "Error: input tensor dtype should be int32"
    return functions_cuda.q4_pack(input, is_transpose)


def q4_unpack_tensor(input: torch.Tensor, is_transpose: bool = False) -> torch.Tensor:
    """
    Unpacks a tensor that has been previously packed using 4-bit quantization into its original format.

    This function is designed to work with tensors that have been quantized and packed,
    reducing their bit representation from a standard format (int32) down to 4-bit representations,
    and then packed two values into a single int8 type. This unpacking function reverses that process,
    reconstructing the original quantized values as a new tensor.

    Parameters:
        input (torch.Tensor): The input tensor that contains packed 4-bit quantized values.
                            It must be of dtype int8.
        is_transpose (bool, optional): Indicates whether the unpacked tensor should be transposed.
                                      The default is False, meaning no transposition will occur.

    Returns:
        torch.Tensor: A tensor containing the unpacked quantized values. The dtype of this tensor
                    will depend on the implementation of the `q4_unpack` function in the `functions_cuda` module,
                    typically returning values in a format suitable for further processing or analysis.

    Raises:
        AssertionError: If the input tensor's dtype is not int8, an assertion error is raised to
                      ensure the unpacking process is applied to a correctly formatted tensor.
    """
    assert input.dtype == torch.int8, "Error: input tensor dtype should be int8."
    return functions_cuda.q4_unpack(input, is_transpose)


def q4_unpack_and_scaling_tensor(input: torch.Tensor, scale:float, is_transpose: bool = False) -> torch.Tensor:
    """
    Unpacks a tensor that has been previously packed using 4-bit quantization into its original format.

    This function is designed to work with tensors that have been quantized and packed,
    reducing their bit representation from a standard format (int32) down to 4-bit representations,
    and then packed two values into a single int8 type. This unpacking function reverses that process,
    reconstructing the original quantized values as a new tensor.

    Parameters:
        input (torch.Tensor): The input tensor that contains packed 4-bit quantized values.
                            It must be of dtype int8.
        scale (float): a scaling factor will be multiplied to the unpacked values.
        is_transpose (bool, optional): Indicates whether the unpacked tensor should be transposed.
                                      The default is False, meaning no transposition will occur.

    Returns:
        torch.Tensor (float): A tensor containing the unpacked quantized values. The dtype of this tensor
                        will depend on the implementation of the `q4_unpack` function in the `functions_cuda` module,
                        typically returning values in a format suitable for further processing or analysis.

    Raises:
        AssertionError: If the input tensor's dtype is not int8, an assertion error is raised to
                      ensure the unpacking process is applied to a correctly formatted tensor.
    """
    assert input.dtype == torch.int8, "Error: input tensor dtype should be int8."
    return functions_cuda.q4_unpack_and_scaling(input, scale, is_transpose)