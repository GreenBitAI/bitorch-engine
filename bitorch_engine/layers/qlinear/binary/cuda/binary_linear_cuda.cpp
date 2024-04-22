#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

/**
 * Performs a forward pass of the binary linear layer using CUDA.
 *
 * This function facilitates binary linear operations with support for different data types
 * for input and weight tensors, including floating point and quantized types. It leverages
 * CUDA for efficient computation, especially suited for deep learning models running on GPU.
 *
 * @param input The input tensor, which can be of type torch::kFloat32 (float), torch::kBFloat16 (bfloat16),
 *              or torch::kHalf (half).
 * @param weight The weight tensor, which supports torch::kInt8 (int8), torch::kFloat32 (float),
 *               and torch::kUInt8 (uint8) data types.
 * @param bmm_type An integer specifying the type of binary matrix multiplication to perform.
 *                 This parameter allows for customization of the operation based on the model's requirements.
 * @param transpose A boolean indicating whether the weight matrix should be transposed during the operation.
 *
 * @return A tensor containing the result of the binary linear operation.
 *
 * @note This function dynamically dispatches to specialized template functions based on the data types of
 *       the input and weight tensors. It supports a combination of float, bfloat16, half, int8, and uint8
 *       types, ensuring flexibility in handling various neural network architectures.
 *       If the data type of the input or weight tensor is not supported, the function will terminate the
 *       program and print an error message.
 */
torch::Tensor binary_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int bmm_type,
    bool transpose);


/**
 * Converts a given weight tensor to its binary representation based on the specified data type.
 *
 * This function supports weight tensors of different data types (int8, float, bfloat16, and half)
 * and converts them to a binary format suitable for certain binary matrix multiplication (BMM) operations.
 * The conversion process is dependent on the data type of the input tensor and whether the tensor
 * should be transposed as part of the conversion.
 *
 * @param weight The input weight tensor to be converted to binary format.
 * @param bmm_type An integer specifying the type of binary matrix multiplication operation.
 *                 This parameter can influence how the binary conversion is performed.
 * @param transpose A boolean indicating whether the weight tensor should be transposed
 *                  as part of the conversion process.
 * @return torch::Tensor A tensor containing the binary representation of the input weight tensor.
 *                       The specific format of the binary representation is determined by the
 *                       data type of the input tensor.
 *
 * @note This function is templated to handle different data types of the input tensor by
 *       calling the appropriate specialized version of the _get_binary_weight_cuda function.
 *       If the data type of the input tensor is not supported, the function prints an error message
 *       and exits the program.
 */
torch::Tensor get_binary_weight_cuda(
    torch::Tensor weights,
    int bmm_type,
    bool transpose);


/**
 * Performs binary matrix multiplication on CUDA using specified data types.
 *
 * This function dispatches the binary matrix multiplication operation to specialized
 * CUDA kernels based on the data type of the input tensors. It supports int8, float32,
 * bfloat16, and half (float16) data types. The function checks if the data types of both
 * input tensors match and then calls the appropriate templated CUDA kernel function.
 *
 * @param x A torch::Tensor representing the first matrix in the multiplication.
 * @param y A torch::Tensor representing the second matrix in the multiplication.
 * @param bmm_type An integer indicating the type of binary matrix multiplication to perform.
 *
 * @return A torch::Tensor containing the result of the binary matrix multiplication.
 *
 * @throws std::runtime_error If the input tensors have different data types or if an unsupported
 *         data type is provided.
 */
torch::Tensor binary_mm_cuda(
    torch::Tensor x,
    torch::Tensor y,
    int bmm_type);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);

torch::Tensor binary_linear_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int bmm_type,
    bool transpose) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return binary_linear_cuda_forward(input, weights, bmm_type, transpose);
}

torch::Tensor get_binary_weight(
    torch::Tensor weights,
    int bmm_type,
    bool transpose) {
    CHECK_INPUT(weights);
    return get_binary_weight_cuda(weights, bmm_type, transpose);
}

torch::Tensor binary_linear_mm(
    torch::Tensor x,
    torch::Tensor y,
    int bmm_type) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    return binary_mm_cuda(x, y, bmm_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &binary_linear_forward, "binary linear forward (CUDA)");
  m.def("w_pack", &get_binary_weight, "get linear binary weight (CUDA)");
  m.def("mm", &binary_linear_mm, "binary linear mm (CUDA)");
}
