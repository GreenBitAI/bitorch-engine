#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

/**
 * Performs a forward pass of binary linear operation using CUTLASS.
 *
 * This function executes a binary linear operation (e.g., binary matrix multiplication) on the given input tensor
 * and weight tensor, utilizing CUTLASS kernels for optimized computation on CUDA devices. It supports both
 * pre-packed and unpacked (requiring packing) weight tensors for flexibility in handling different data formats.
 *
 * @param input A torch::Tensor representing the input matrix with shape [m, k], where m is the batch size
 *              and k is the dimensionality of the input features. The input tensor can be either in packed
 *              binary format (torch::kUInt8) or in a format that requires packing.
 * @param weight A torch::Tensor representing the weight matrix with shape [n, k] if not transposed,
 *               and [k, n] if transposed, where n is the output dimensionality. The weight tensor can be
 *               in packed binary format (torch::kUInt8) or in a floating-point format that requires packing.
 * @param transpose A boolean flag indicating whether the weight matrix is transposed. If true, the weight
 *                  tensor is assumed to be in [k, n] shape and will be transposed internally if not already
 *                  in packed format.
 * @param kernel_id An integer specifying the ID of the CUTLASS kernel to be used for the computation. Different
 *                  kernels may be optimized for specific shapes or hardware configurations.
 *
 * @return A torch::Tensor representing the output matrix with shape [m, n], where m is the batch size
 *         and n is the output dimensionality. The output tensor contains the result of the binary linear
 *         operation performed on the input tensor and the weight tensor.
 *
 * @note This function requires the input and/or weight tensors to be in packed binary format (torch::kUInt8)
 *       for efficient computation. If tensors are not in this format, they will be packed internally.
 *       The packing process converts floating-point or integer tensors into a compact binary representation,
 *       reducing memory footprint and computation time on supported CUDA devices.
 */
torch::Tensor binary_linear_cutlass_forward(
    torch::Tensor input,
    torch::Tensor weight,
    bool transpose);


/**
 * Retrieves a packed data tensor, optionally transposing it.
 *
 * This function takes an input tensor and, based on its data type,
 * applies a specific packing operation using CUDA kernels. Supported
 * data types include int8, float32, bfloat16, and half. The function
 * can also transpose the packed tensor if requested.
 *
 * @param data The input tensor to be packed. It supports tensors of
 *             type int8, float32, bfloat16, and half.
 * @param transpose A boolean flag indicating whether to transpose the
 *                  packed tensor. If true, the tensor is transposed
 *                  along its last two dimensions.
 * @return torch::Tensor The packed (and optionally transposed) tensor.
 *
 * Note: If the tensor's data type is not supported, the function will
 *       output an error message and terminate the execution.
 */
torch::Tensor get_packed_data_tensor(
    const torch::Tensor data,
    bool transpose);


/**
 * Performs binary matrix multiplication using CUTLASS kernels.
 * This function takes two input tensors `x` and `y`, along with a `kernel_id`
 * to select the specific CUTLASS kernel for the operation. It first ensures
 * that the operation is carried out on the same CUDA device as `x`. The inputs
 * are then packed to reduce their bit-width, according to the specified kernel's
 * requirements, before being passed to a binary forward operation implemented by CUTLASS.
 *
 * @param x A torch::Tensor representing the left matrix in the multiplication.
 *          It should have a shape of (m, k), where `m` is the number of rows,
 *          and `k` is the number of columns (which matches the dimension of `y`).
 * @param y A torch::Tensor representing the right matrix in the multiplication.
 *          Its shape should be (n, k), aligning with `x`'s second dimension.
 * @param kernel_id An integer specifying which CUTLASS kernel to use for the operation.
 *                  Different kernels may be optimized for specific sizes or types of operations.
 *
 * @return torch::Tensor The result of the binary matrix multiplication, with a shape of (m, n).
 *
 * Note: This function assumes both input tensors are on the same CUDA device and
 *       uses CUDA's guard mechanism to ensure the computation is executed on the correct device.
 *       The tensors `x` and `y` are first packed to reduce their size according to the bit-width
 *       expected by the binary forward operation, which is a key step in optimizing performance
 *       for binary operations.
 */
torch::Tensor binary_mm_function(
    torch::Tensor x,
    torch::Tensor y,
    int kernel_id);


/**
 * Performs binary matrix multiplication with optional scaling.
 * This function executes a binary matrix multiplication between tensors `x` and `y`,
 * followed by scaling the result by a specified factor. It is designed to work with CUDA tensors.
 *
 * The inputs `x` and `y` are expected to be 3-dimensional where the last dimension of `x` and `y`
 * should match. The first dimension is treated as the batch size, allowing for batched matrix multiplication.
 *
 * After the computation, the output tensor is scaled by the provided `scale` factor. This scaling
 * is applied differently based on the output tensor's data type.
 *
 * @param x The first input tensor, expected to be of shape (batch_size, m, k).
 * @param y The second input tensor, expected to be of shape (batch_size, n, k).
 * @param scale A float value used to scale the output tensor.
 * @return torch::Tensor The result of the binary matrix multiplication, scaled by `scale`,
 *         with shape (batch_size, m, n).
 *
 * Note: This function is designed to work on CUDA devices and requires the inputs `x` and `y`
 *       to be CUDA tensors. It supports `torch::kFloat32` and `torch::kBFloat16` data types for the output tensor.
 *       If the output tensor's data type is not supported, the function will terminate the program.
 */
torch::Tensor binary_matmul_function(
    torch::Tensor x,
    torch::Tensor y,
    float scale);


/**
 * Performs a binary linear operation using CUTLASS kernels, followed by scaling the output.
 *
 * This function applies a binary linear operation (such as binary matrix multiplication) on the input tensor and weight tensor,
 * and then scales the result according to a given scale factor. It supports operations on both float32 and bfloat16 data types,
 * with the option to perform the operation on transposed weight matrices.
 *
 * @param input A tensor representing the input data. The last dimension size should match the weight tensor's relevant dimension.
 * @param weight A tensor representing the binary weights. Its dimensions should be compatible with the input tensor.
 * @param scale A float value representing the scale factor to apply to the output tensor.
 * @param transpose A boolean indicating whether the weight tensor should be transposed before the operation.
 * @param kernel_id An integer specifying which CUTLASS kernel to use for the operation. Different kernels may be optimized for different scenarios.
 * @return torch::Tensor The result of the binary linear operation, scaled by the specified factor. The output tensor's data type matches the input tensor's data type.
 *
 * Note:
 * - This function automatically selects the appropriate scaling kernel based on the output tensor's data type.
 * - If the output tensor's data type is not float32 or bfloat16, the function will terminate the program with an error message.
 */
torch::Tensor binary_linear_cutlass_scaled_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float scale,
    bool transpose,
    int kernel_id);


/**
 * Selects the optimal kernel ID for binary linear operations using CUTLASS based on device and matrix dimensions.
 *
 * This function determines the best kernel to use for binary linear operations (such as matrix multiplication) on a given CUDA device,
 * considering the dimensions of the input matrices. It ensures that the selected kernel ID is within a valid range. If an invalid kernel ID
 * is found, the program will report an error and exit.
 *
 * @param device_id An integer representing the CUDA device ID.
 * @param m The number of rows in the first matrix.
 * @param n The number of columns in the second matrix.
 * @param k The common dimension of the matrices (i.e., the number of columns in the first matrix and the number of rows in the second matrix).
 * @return An integer representing the ID of the selected kernel, guaranteed to be within the valid range.
 */
int get_selected_kernel_id(
    int device_id, int m, int n, int k);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);

torch::Tensor get_packed_tensor(
    const torch::Tensor data,
    bool transpose){
    CHECK_INPUT(data);
    return get_packed_data_tensor(data, transpose);
}

torch::Tensor binary_mm(
    torch::Tensor x,
    torch::Tensor y,
    int kernel_id) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
  return binary_mm_function(x, y, kernel_id);
}

torch::Tensor binary_matmul(
    torch::Tensor x,
    torch::Tensor y,
    float scale) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
  return binary_matmul_function(x, y, scale);
}

torch::Tensor binary_linear_scaled_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float scale,
    bool transpose,
    int kernel_id) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
  return binary_linear_cutlass_scaled_forward(input, weight, scale, transpose, kernel_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &binary_linear_scaled_forward, "scaled binary linear forward (CUTLASS)");
    m.def("w_pack", &get_packed_tensor, "packing binary weight (CUTLASS)");
    m.def("mm", &binary_mm, "binary mm function (CUTLASS)");
    m.def("matmul", &binary_matmul, "binary matmul function for High-dimensional matrix multiplication (CUTLASS)");
    m.def("kernel_eval", &get_selected_kernel_id, "get selected kernel id (CUTLASS)");
}
