#include <torch/extension.h>
#include <vector>

// CUDA declarations

/**
 * Converts a 32-bit floating point tensor to a 4-bit integer representation.
 *
 * This function takes an input tensor of floating point numbers and compresses it
 * into a tensor of 4-bit integers, effectively reducing the memory footprint by a factor of 8.
 * The conversion process involves finding the minimum and maximum values of the input
 * to normalize the data range, and then quantizing the normalized values into 4-bit integers.
 *
 * Parameters:
 *  - input: A tensor of 32-bit floating point numbers that we want to compress.
 *
 * Returns:
 *  - A tensor of 4-bit integers representing the quantized version of the input tensor.
 *    The output tensor uses a 64-bit integer data type to store the 4-bit values,
 *    with each 64-bit integer holding sixteen 4-bit values.
 *
 * Note:
 *  - The input tensor is assumed to be a flat 1D tensor, and the output tensor will also be a 1D tensor.
 *  - This function is designed to be executed on CUDA-enabled devices and utilizes custom CUDA kernels
 *    for the quantization process.
 *  - The function allocates temporary memory on the GPU for intermediate computations, which is
 *    freed before returning the output tensor.
 */
torch::Tensor fp32_to_int4_cuda(
    torch::Tensor input);


/**
 * Packs the given tensor into an 8-bit unsigned integer tensor representation on CUDA.
 *
 * This function converts a tensor of various data types (int8, float32, bfloat16, half)
 * to an 8-bit unsigned integer tensor using CUDA kernels for efficient computation.
 * It ensures that the operation is performed on the correct CUDA device by employing
 * a device guard based on the input tensor's device. If the data type of the input tensor
 * is not supported, the function will terminate the program with an error message.
 *
 * Parameters:
 *  - data: The input tensor to be packed. Can be of type int8, float32, bfloat16, or half.
 *
 * Returns:
 *  - torch::Tensor: An 8-bit unsigned integer representation of the input tensor.
 *
 * Note:
 *  - The input tensor must be on a CUDA device.
 *  - The function terminates the program if the input tensor's type is not supported.
 */
torch::Tensor tensor_pack_to_uint8_cuda(
    torch::Tensor input);


/**
 * Converts an 8-bit unsigned integer tensor into an unpacked float tensor on CUDA.
 *
 * This function unpacks an 8-bit integer tensor (`emd`) into a float tensor (`out`)
 * while scaling it with another tensor (`scl`). The unpacking process converts
 * each 8-bit integer into 8 separate float values, effectively increasing the
 * dimensionality of the embedding space by a factor of 8. This is particularly
 * useful for operations that require higher precision representations of compressed
 * embeddings.
 *
 * Parameters:
 *  emd A torch::Tensor containing 8-bit unsigned integers with shape
 *            (batch_size, sequence_length, packed_embedding_dimension), representing
 *            the compressed embeddings.
 *  scl A torch::Tensor with shape (batch_size, sequence_length, 1), used for
 *            scaling the unpacked embeddings.
 *
 * return
    A torch::Tensor of floats with shape
 *         (batch_size, sequence_length, packed_embedding_dimension * 8), representing
 *         the unpacked and scaled embeddings.
 *
 * This function calculates the size of the
 * output tensor based on the input dimensions, and initializes the output tensor with
 * zeros. It then launches a CUDA kernel (`unpack_uint8_to_float`) to perform the unpacking
 * and scaling in parallel on the GPU.
 */
torch::Tensor uint8_to_unpacked_tensor_cuda(
    torch::Tensor input,
    torch::Tensor scale);


/**
 * Function to perform 4-bit packing on a tensor.
 * This function packs the input data tensor into 4-bit integers,
 * storing every two quantized values into one int8 variable.
 *
 * Parameters:
 *     data: The input tensor to be quantized and packed. The tensor can have 2 or 3 dimensions,
 *           where the last two dimensions are considered as out_channels and in_channels, respectively.
 *     is_transpose: If true, the packed tensor will be transposed on the last two dimensions.
 *
 * Returns:
 *     A packed tensor with int8 data type, where every two 4-bit quantized values are stored in one int8 variable.
 */
torch::Tensor q4_pack_cuda(
    const torch::Tensor data,
    bool is_transpose
);


/**
 * Function to unpack a 4-bit packed tensor into its original int values.
 * Parameters:
 * packed_data: The tensor containing packed 4-bit values in int8 format.
 * is_transpose: Boolean flag indicating whether the unpacked tensor should be transposed.
 * Returns:
 * A tensor of int32 values, unpacked from the input tensor.
 */
torch::Tensor q4_unpack_cuda(
    const torch::Tensor packed_data,
    bool is_transpose
);


/**
 * Function to unpack a 4-bit packed tensor into its original int values.
 * Parameters:
 * packed_data: The tensor containing packed 4-bit values in int8 format.
 * scale: a scaling factor will be applied to each unpacked value.
 * is_transpose: Boolean flag indicating whether the unpacked tensor should be transposed.
 * Returns:
 * A tensor of int32 values, unpacked from the input tensor.
 */
torch::Tensor q4_unpack_and_scaling_cuda(
    const torch::Tensor packed_data,
    float scale,
    bool is_transpose
);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);


torch::Tensor fp32_to_int4(
    torch::Tensor input) {
    CHECK_INPUT(input);
    return fp32_to_int4_cuda(input);
}


torch::Tensor tensor_pack_to_uint8(
    torch::Tensor input) {
    CHECK_INPUT(input);
    return tensor_pack_to_uint8_cuda(input);
}


torch::Tensor uint8_to_unpacked_tensor(
    torch::Tensor input,
    torch::Tensor scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(scale);
    return uint8_to_unpacked_tensor_cuda(input, scale);
}


torch::Tensor q4_pack(
    const torch::Tensor data,
    bool is_transpose
){
    CHECK_INPUT(data);
    return q4_pack_cuda(data, is_transpose);
}


torch::Tensor q4_unpack(
    const torch::Tensor packed_data,
    bool is_transpose
){
    CHECK_INPUT(packed_data);
    return q4_unpack_cuda(packed_data, is_transpose);
}


torch::Tensor q4_unpack_and_scaling(
    const torch::Tensor packed_data,
    float scale,
    bool is_transpose
){
    CHECK_INPUT(packed_data);
    return q4_unpack_and_scaling_cuda(packed_data, scale, is_transpose);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp32toint4", &fp32_to_int4, "fp32 to int4 tensor (CUDA)");
	m.def("tensor_pack_to_uint8", &tensor_pack_to_uint8, "get packed uint8 tensor (CUDA)");
	m.def("uint8_to_unpacked_tensor", &uint8_to_unpacked_tensor, "get unpacked tensor (CUDA)");
	m.def("q4_pack", &q4_pack, "pack a int32 tensor to 4-bit packed tensor (CUDA)");
	m.def("q4_unpack", &q4_unpack, "unpack 4-bit packed tensor to int32 tensor (CUDA)");
	m.def("q4_unpack_and_scaling", &q4_unpack_and_scaling, "unpack 4-bit packed tensor and scaling (CUDA)");
}
