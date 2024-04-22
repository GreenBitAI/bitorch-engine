#include <torch/extension.h>
#include <vector>


// CUDA declarations
/*
 * Args:
 * transpose: True: The second tensor arg "weight" will be transposed.
 */

 /**
 * Performs a forward pass using a 4-bit quantized linear layer with CUTLASS.
 *
 * This function executes a quantized matrix multiplication (GEMM) operation, using
 * CUTLASS templates for 4-bit computation. It supports training and inference modes,
 * with optional weight quantization and packing during training.
 *
 * @param input The input tensor, expected to be a 2D tensor where the first dimension is the batch size.
 * @param weight The weight tensor, expected to be a 2D tensor where the dimensions are output features by input features.
 * @param scale_a The quantization scale for the input tensor.
 * @param scale_w The quantization scale for the weight tensor.
 * @param transpose Indicates whether the weight tensor should be transposed.
 * @param is_train Indicates if the operation is performed in training mode. In training mode, weights are quantized and packed.
 * @return A vector of tensors containing the output tensor, the packed input tensor, and the packed or original weight tensor.
 *
 * The function begins by setting up a CUDA guard for the device associated with the input tensor.
 * It then calculates the dimensions for the matrix multiplication operation based on the input and weight tensor sizes.
 * Depending on the training mode, it may pack and quantize the weight tensor. The input tensor is always packed and quantized.
 * The 4-bit GEMM operation is performed using CUTLASS, and the result is scaled back to the original data range.
 * The function supports output tensors in float32 and bfloat16 data types, with appropriate scaling applied.
 * Finally, it returns the output tensor along with the packed input and weight tensors.
 *
 * Note: If the input tensor's data type is not supported, the function will print an error message and terminate the program.
 */
std::vector<torch::Tensor> q4_linear_cutlass_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float scale_a,
    float scale_w,
    bool transpose,
    bool is_train);


/**
 * Performs the backward pass for a 4-bit quantized operation, calculating gradients for both activations and weights.
 *
 * This function computes the gradients of the activations (q4_a) and weights (q4_w) in a 4-bit quantized format,
 * using the gradient of the output (output_gradient) and scales for activations, weights, and gradient. The gradients
 * are computed using a 4-bit generalized matrix multiplication (gemm) operation, tailored for quantized tensors.
 *
 * The function uses CUTLASS to perform the quantized operations, ensuring efficiency on GPU-accelerated hardware.
 *
 * @param output_gradient The gradient of the output tensor, in a floating-point format.
 * @param q4_a The 4-bit quantized activations tensor from the forward pass.
 * @param q4_w The 4-bit quantized weights tensor.
 * @param scale_a The scale factor used to quantize the activations in the forward pass.
 * @param scale_w The scale factor used to quantize the weights in the forward pass.
 * @param scale_grad The scale factor used to quantize the output gradient.
 *
 * @return A pair of tensors representing the gradients of the activations and weights, respectively. These tensors
 *         are scaled according to the provided scale factors for activations and weights.
 *
 * Note: The dimensions m, k, and n are derived from the shapes of the input tensors, considering the specific
 *       quantization scheme (4-bit) used here. The variable `k` is adjusted to account for the packing in the forward pass.
 */
std::pair<torch::Tensor, torch::Tensor> q4_linear_cutlass_backward(
    torch::Tensor output_gradient,
    torch::Tensor input_q4,
    torch::Tensor weight_q4,
    float scale_a,
    float scale_w,
    float scale_grad);


/**
 * Performs quantization and packing of data tensor into 4-bit representation.
 *
 * This function quantizes a given data tensor using the provided scale factor
 * and packs the quantized values into 4-bit integers to reduce memory usage.
 * The function supports tensors with 2 or 3 dimensions, where the last two dimensions
 * are considered as out_channels and in_channels respectively. The function also
 * supports optional transposition of the resulting tensor.
 *
 * @param data The input tensor to be quantized and packed. It can have 2 or 3 dimensions.
 * @param scale The scale factor to be used for quantization. All tensor elements
 *              are scaled by this factor during the quantization process.
 * @param is_transpose A boolean flag indicating whether the resulting packed tensor
 *                     should be transposed (last two dimensions swapped).
 *
 * @return A tensor with the quantized and packed data. The resulting tensor will have
 *         type torch::kInt8 to represent the 4-bit packed values and it will have the same
 *         device as the input tensor. The shape of the output tensor depends on the input
 *         tensor's dimensions and the is_transpose flag.
 *
 * Note: The function exits with an error message if the input tensor has dimensions other
 *       than 2 or 3. For a 3-dimensional input tensor, the first dimension is considered
 *       as a batch dimension and is multiplied with out_channels dimension for packing.
 *       If is_transpose is true, the last two dimensions of the resulting tensor are swapped.
 */
torch::Tensor get_q4_packed_data_tensor_cuda(
    torch::Tensor data,
    float scale,
    bool transpose);


/**
 * Forward pass for a linear layer using CUTLASS for quantized 8-bit GEMM.
 *
 * This function implements the forward pass of a linear (fully connected) layer
 * by performing quantized 8-bit matrix multiplication using the CUTLASS library.
 * The function supports optional transposition of the weight matrix for flexibility
 * in defining the layer's parameters.
 *
 * @param input The input tensor, expected to be a 2D tensor where the first dimension
 *              is the batch size and the second dimension is the feature size.
 * @param weight The weight matrix for the linear layer, quantized to 8-bit integers.
 * @param transpose A boolean flag indicating whether the weight matrix should be
 *                  transposed before the GEMM operation.
 * @param scale_a activation scaling factor.
 * @param scale_w weight scaling factor.
 * @return A tensor containing the result of the linear layer's forward pass.
 */
torch::Tensor q8_linear_cutlass_forward(
    torch::Tensor input,
    torch::Tensor weight,
    bool transpose,
    float scale_a,
    float scale_w);


/**
 * Performs the backward pass for an 8-bit quantized linear (fully connected) layer using CUTLASS.
 * This function computes the gradients for both the activations (inputs) and the weights, based on the output gradient.
 *
 * The function assumes the input activations and weights are already quantized to 8-bit integers (int8).
 * It performs two 8-bit GEMM (General Matrix Multiply) operations to compute the gradients:
 * - One GEMM operation computes the gradient with respect to the activations (grad_a).
 * - Another GEMM operation computes the gradient with respect to the weights (grad_w).
 *
 * Both gradients are computed in 8-bit precision to maintain consistency with the forward pass and to exploit
 * the efficiency of low-precision arithmetic.
 *
 * @param output_gradient The gradient of the loss with respect to the output of the linear layer.
 *                        This tensor should have the same shape as the layer output and must be of type int8.
 * @param q8_a The quantized activations (inputs) to the linear layer. Must be a 2D tensor of type int8.
 * @param q8_w The quantized weights of the linear layer. Must be a 2D tensor of type int8.
 *
 * @return A pair of tensors:
 *         - The first element is the gradient with respect to the activations (inputs) of the linear layer.
 *         - The second element is the gradient with respect to the weights of the linear layer.
 *         Both elements are tensors of type int8.
 *
 * @note This function checks the data types of the input tensors and will terminate the program with an error message
 *       if the activations or weights are not of type int8. This strict check ensures that the function operates
 *       under the expected conditions for quantized tensors.
 */
std::pair<torch::Tensor, torch::Tensor> q8_linear_cutlass_backward(
    torch::Tensor output_gradient,
    torch::Tensor input_q8,
    torch::Tensor weight_q8);


/**
 * Performs a 4-bit matrix multiplication (GEMM) using CUTLASS templates.
 *
 * This function executes a quantized matrix multiplication where both input matrices x and y
 * are quantized to 4-bit precision. The quantization scales for each matrix are provided
 * by scale_x and scale_y respectively. This operation is useful for efficient computation
 * on hardware accelerators by reducing the precision of the data.
 *
 * @param x The first input tensor with shape (m, k), where 'm' is the number of rows
 *          and 'k' is the number of columns. This tensor represents the left matrix in the multiplication.
 * @param y The second input tensor with shape (n, k), where 'n' is the number of rows
 *          and 'k' matches the number of columns in 'x'. This tensor represents the right matrix in the multiplication.
 * @param scale_x The quantization scale factor for the first input tensor 'x'. This scale is used to quantize the data to 4-bit.
 * @param scale_y The quantization scale factor for the second input tensor 'y'. Similarly, it's used for quantizing 'y' to 4-bit.
 *
 * @return A std::vector containing three torch::Tensor elements:
 *         1. The result of the 4-bit GEMM operation,
 *         2. The packed and quantized version of the first input tensor 'x',
 *         3. The packed and quantized version of the second input tensor 'y'.
 *
 * @note The function uses CUTLASS templates to perform the 4-bit GEMM operation efficiently.
 *       The inputs are first packed and quantized to 4-bit representations using the provided scale factors.
 *       The CUDA OptionalCUDAGuard is used to ensure that the operation is executed on the correct device.
 */
std::vector<torch::Tensor> q4_mm_cutlass(
    torch::Tensor x,
    torch::Tensor y,
    float scale_x,
    float scale_y);


/**
 * Performs quantized matrix multiplication using CUTLASS for 4-bit quantized tensors.
 * This function takes two input tensors `x` and `y`, along with their respective scales
 * `scale_x` and `scale_y`, to perform the quantized matrix multiplication operation.
 * The function is designed to handle inputs with a batch dimension.
 *
 * @param x The input tensor representing the left matrix in the multiplication.
 *          Expected shape: (batch_size, m, k), where 'm' is the number of rows in each matrix,
 *          'k' is the common dimension, and 'batch_size' is the size of the batch.
 * @param y The input tensor representing the right matrix in the multiplication.
 *          Expected shape: (batch_size, n, k), where 'n' is the number of columns in each matrix,
 *          and 'k' matches the last dimension of tensor 'x'.
 * @param scale_x The scaling factor applied to tensor 'x' for quantization.
 * @param scale_y The scaling factor applied to tensor 'y' for quantization.
 *
 * @return A vector of torch::Tensor, which includes the result of the batched GEMM operation
 *         on the quantized inputs, along with the packed versions of 'x' and 'y'.
 *         The first element in the vector is the GEMM result, the second is the packed 'x',
 *         and the third is the packed 'y'.
 *
 * Note: This function utilizes the CUTLASS library for efficient GPU-accelerated
 *       quantized matrix multiplications. It assumes the input tensors are already
 *       quantized and requires them to be reshaped into 3D tensors if not in that format.
 *       The function also handles the packing of input tensors into 4-bit representations
 *       before performing the multiplication.
 */
std::vector<torch::Tensor> q4_matmul_cutlass(
    torch::Tensor x,
    torch::Tensor y,
    float scale_x,
    float scale_y);


/**
 * Performs backward pass for 4-bit quantized matrix multiplication using CUTLASS, returning gradients for inputs.
 * This function computes gradients for both input matrices (`q4_x` and `q4_y`) given the gradient of the output (`output_gradient`)
 * and scales associated with both inputs and the gradient. It leverages 4-bit quantized GEMM (General Matrix Multiply)
 * provided by CUTLASS for efficient computation on CUDA-enabled devices.
 *
 * @param output_gradient The gradient of the output tensor from the forward pass, with shape (bs, m, n).
 * @param q4_x The first input tensor (4-bit quantized) involved in the original matrix multiplication, with shape (bs, m, k/2).
 * @param q4_y The second input tensor (4-bit quantized) involved in the original matrix multiplication, with shape (bs, n, k/2).
 * @param scale_x Scale factor used for quantization of `q4_x`.
 * @param scale_y Scale factor used for quantization of `q4_y`.
 * @param scale_grad Scale factor used for quantization of `output_gradient`.
 *
 * @details
 * - `output_gradient` is the gradient with respect to the output of the 4-bit quantized matrix multiplication.
 * - `q4_x` and `q4_y` are the original 4-bit quantized inputs to the matrix multiplication.
 * - The function calculates gradients for both `q4_x` and `q4_y` by performing quantized GEMM operations.
 * - `scale_x`, `scale_y`, and `scale_grad` are scale factors that were applied during forward pass quantization,
 *    which need to be considered during the gradient computation to ensure correctness.
 *
 * @return A pair of tensors:
 *    - The first tensor is the gradient with respect to `q4_x`, scaled by `scale_x`.
 *    - The second tensor is the gradient with respect to `q4_y`, scaled by `scale_y`.
 *
 * @note
 * - The inputs `q4_x` and `q4_y` are expected to be in 4-bit packed format, which effectively halves the last dimension size.
 * - The function internally reshapes the inputs and `output_gradient` to 3D (if not already) for batched GEMM operations.
 * - This function is optimized for CUDA and uses CUTLASS for the underlying 4-bit GEMM computations.
 */
std::pair<torch::Tensor, torch::Tensor> q4_matmul_backward_cutlass(
    torch::Tensor output_gradient,
    torch::Tensor x,
    torch::Tensor y,
    float scale_x,
    float scale_y,
    float scale_grad);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);

std::vector<torch::Tensor> q4_linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float scale_a,
    float scale_w,
    bool transpose,
    bool is_train) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
  return q4_linear_cutlass_forward(input, weight, scale_a, scale_w, transpose, is_train);
}


std::pair<torch::Tensor, torch::Tensor> q4_linear_backward(
    torch::Tensor output_gradient,
    torch::Tensor input_q4,
    torch::Tensor weight_q4,
    float scale_a,
    float scale_w,
    float scale_grad) {
    CHECK_INPUT(input_q4);
    CHECK_INPUT(weight_q4);
    CHECK_INPUT(output_gradient);
  return q4_linear_cutlass_backward(output_gradient, input_q4,
            weight_q4, scale_a, scale_w, scale_grad);
}


std::vector<torch::Tensor> q4_mm(
    torch::Tensor x,
    torch::Tensor y,
    float scale_x,
    float scale_y) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
  return q4_mm_cutlass(x, y, scale_x, scale_y);
}


std::vector<torch::Tensor> q4_matmul(
    torch::Tensor x,
    torch::Tensor y,
    float scale_x,
    float scale_y) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
  return q4_matmul_cutlass(x, y, scale_x, scale_y);
}


std::pair<torch::Tensor, torch::Tensor> q4_matmul_backward(
    torch::Tensor output_gradient,
    torch::Tensor x,
    torch::Tensor y,
    float scale_x,
    float scale_y,
    float scale_grad) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(output_gradient);
	return q4_matmul_backward_cutlass(output_gradient, x, y, scale_x, scale_y, scale_grad);
}


torch::Tensor get_q4_weight(
    torch::Tensor weight,
    float scale_w) {
    CHECK_INPUT(weight);
    return get_q4_packed_data_tensor_cuda(weight, scale_w, false);
}


// ========== Q8 ========== //
torch::Tensor q8_linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    bool transpose,
    float scale_a,
    float scale_w) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
  return q8_linear_cutlass_forward(input, weight, transpose, scale_a, scale_w);
}


std::pair<torch::Tensor, torch::Tensor> q8_linear_backward(
    torch::Tensor output_gradient,
    torch::Tensor input_q8,
    torch::Tensor weight_q8) {
    CHECK_INPUT(input_q8);
    CHECK_INPUT(weight_q8);
    CHECK_INPUT(output_gradient);
  return q8_linear_cutlass_backward(output_gradient, input_q8, weight_q8);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q4_forward", &q4_linear_forward, "4-bit linear forward (CUTLASS)");
    m.def("q4_backward", &q4_linear_backward, "4-bit linear backward (CUTLASS)");
    m.def("q4_w_pack", &get_q4_weight, "4-bit weight packing");
    m.def("q4_mm", &q4_mm, "4-bit 2D matrix multiplication");
    m.def("q4_matmul", &q4_matmul, "4-bit higher dimension matrix multiplication");
    m.def("q4_matmul_backward", &q4_matmul_backward, "4-bit higher dimension matrix multiplication");
    m.def("q8_forward", &q8_linear_forward, "8-bit linear forward (CUTLASS)");
    m.def("q8_backward", &q8_linear_backward, "8-bit linear backward (CUTLASS)");
}
