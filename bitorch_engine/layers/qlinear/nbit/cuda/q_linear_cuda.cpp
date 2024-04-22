#include <torch/extension.h>
#include <vector>


// CUDA declarations


/**
 * Performs a forward pass of a quantized matrix multiplication using CUDA.
 *
 * This function conducts a forward pass of a linear layer with mixed precision quantized weights.
 * It supports efficient computation on CUDA devices by leveraging optimized CUDA kernels, specifically
 * designed for quantized neural network operations.
 *
 * @param x The input tensor with shape (m, k), where m is the batch size and k is the feature dimension.
 * @param qweight The quantized weight tensor, stored in a compact format to save memory and improve computational efficiency.
 * @param scales The scaling factors for the quantized weights to convert them back to their floating-point equivalents.
 * @param qzeros The zero-point values for the quantized weights, used in asymmetric quantization schemes to adjust the zero level.
 * @param g_idx An index tensor that maps the groupings in quantized matrix multiplication, facilitating grouped or depthwise operations.
 * @param a_bit The bit width used for the input tensor quantization. Currently, only 16-bit quantization is supported.
 * @param w_bit The bit width used for the weight tensor quantization.
 * @param asym A boolean flag indicating whether asymmetric quantization is used for weights. If true, asymmetric quantization is applied.
 *
 * @return A tensor containing the result of the quantized matrix multiplication with shape (m, n),
 *         where n is the output feature dimension, derived from the shape of qweight.
 *
 * Note: This function currently only supports 16-bit quantization for the input tensor (`a_bit` == 16).
 *       If a different bit width is specified, the function will terminate the program with an error message.
 */
torch::Tensor mpq_linear_cuda_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    int a_bit,
    int w_bit,
    bool asym);


/**
 * Computes the gradient with respect to the input of a mixed-precision quantized linear layer on CUDA.
 *
 * This function calculates the gradient of the input tensor based on the provided gradients of the output,
 * quantized weights, scales, and zero points. It supports asymmetric quantization and is optimized for CUDA devices.
 *
 * @param qweight Quantized weights tensor, which holds the weights of the linear layer in a quantized format.
 * @param scales A tensor containing the scale factors used for quantization of the weights.
 * @param qzeros A tensor containing the zero points used for asymmetric quantization of the weights.
 * @param g_idx A tensor containing group indices for the quantized matrix multiplication.
 * @param output_gradient The gradient of the output tensor from the subsequent layer.
 * @param a_bit Bit width for the activation quantization.
 * @param w_bit Bit width for the weight quantization.
 * @param asym A boolean flag indicating whether asymmetric quantization is used.
 *
 * @return A tensor containing the gradient with respect to the input of the linear layer.
 *
 * Note: Currently, only 16-bit activation quantization (a_bit == 16) is supported. If a different bit width
 * is provided, the function will print an error message and terminate the execution.
 */
torch::Tensor mpq_linear_cuda_grad_input(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    torch::Tensor output_gradient,
    int a_bit,
    int w_bit,
    bool asym);


/**
 * Performs a mixed bit-width quantized linear transformation on quantized weights using CUDA.
 *
 * This function applies a permutation based on the bit width of each quantized group within the
 * weight tensor, supporting mixed bit-width configurations. It operates directly on CUDA tensors
 * and utilizes CUDA kernels for efficient computation. The function is designed to work with
 * quantized neural network models where different parts of the network might operate at different
 * quantization levels.
 *
 * @param qweight A CUDA tensor representing the quantized weights of a linear layer.
 * @param cuda_q_groups A CUDA tensor containing the group information, including the bit width
 *                      and the start row for each group within the `qweight` tensor.
 * @param use_mbw A boolean flag indicating whether mixed bit-width (MBW) configuration is used.
 *                If `false`, a default bit-width is assumed for all weights.
 * @param height The height (number of rows) in the `qweight` tensor.
 * @param groups The number of distinct quantization groups within the weights, each possibly
 *               having a different bit width.
 * @param bits The bit-width required when use_mbw=False.
 *
 * @return A pair consisting of the transformed `qweight` tensor and a vector of integers. The
 *         vector contains the start row number for each bit width, beginning from 8-bits down to
 *         2-bits, followed by a bit pattern indicating the presence of different bit widths.
 *         For example, `0b00000010` indicates 2-bit, `0b00001110` indicates 2, 3, and 4-bit, etc.
 *
 *         The transformation rearranges the quantized weights according to their bit widths,
 *         facilitating efficient processing by subsequent CUDA kernels that leverage the mixed
 *         bit-width configuration.
 */
std::pair<torch::Tensor, std::vector<int>>  mbwq_linear_trans_qweight_cuda(
    torch::Tensor qweight,
    torch::Tensor cuda_q_groups,
    bool use_mbw,
    int height,
    int groups,
    int bits);


/**
 * Performs the conversion of GPTQ-like 4-bit quantized weights to full precision using CUDA.
 *
 * This function converts 4-bit quantized weights (`qweight`) back to full-precision weights.
 * It utilizes CUDA for efficient computation, suitable for neural network operations,
 * especially in linear layers where quantization is applied. The conversion considers
 * scale factors (`scales`) and zero points (`zeros`) for accurate reconstruction.
 *
 * @param qweight A torch::Tensor containing the 4-bit quantized weights.
 * @param scales A torch::Tensor containing the scale factors for each quantization group.
 * @param zeros A torch::Tensor containing the zero points for each quantization group,
 *              used to correctly shift the quantized values.
 * @param group_size An integer specifying the size of the quantization group,
 *                   which is used to determine the number of input channels per group.
 * @param bits An integer specifying the size of the bit width
 * @param q_perm A torch::Tensor representing permutation indices for quantized weight groups.
 *
 * @return A torch::Tensor of the reconstructed full-precision weights.
 *
 * Detailed operation:
 * 1. Initializes CUDA guard for the device associated with `qweight`.
 * 2. Calculates the dimensions for the output tensor based on `qweight` dimensions and `scales` count.
 * 3. Sets up CUDA kernel dimensions for efficient parallel computation.
 * 4. Calls the CUDA kernel `reconstruct_gptq_kernel` to perform the actual reconstruction.
 *    The kernel converts 4-bit quantized values back to half-precision floating-point format (`half` type),
 *    considering the provided scales and zero points.
 */
torch::Tensor mbwq_linear_q42fp_weight_cuda(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size,
    int bits,
    torch::Tensor q_perm);


/**
 * Performs a forward pass of the quantized linear (fully connected) layer using 4-bit quantization on CUDA.
 *
 * This function executes a matrix multiplication between the input tensor `x` and the quantized weight matrix `qweight`,
 * with additional scaling factors applied from `scales`. The computation is performed on the GPU and is optimized for
 * 4-bit quantization, making it suitable for models where memory and computational efficiency are crucial.
 *
 * @param x The input tensor with a shape of [batch_size, in_features], expected to be in half precision (torch::kHalf).
 * @param qweight The quantized weight matrix, packed into 32-bit integers, with a shape of [in_features/8, out_features].
 *                Each 32-bit block represents 8 consecutive 4-bit quantized values.
 * @param scales The scaling factors for the quantization, typically one per output channel, with a shape of [out_channels].
 * @param zeros Placeholder tensor for potential zero-point adjustments or other quantization parameters, not used in this implementation.
 * @param group_size The size of the group for grouped convolution operations, affecting how input channels are divided.
 *                   For standard (non-grouped) operations, this is typically set to the number of input channels.
 * @param q_perm Permutation indices for quantized weight matrix, supporting optimized memory access patterns.
 * @param bits q_weight's bitwidth.
 *
 * @return A tensor containing the result of the quantized linear operation, with a shape of [batch_size, out_features].
 *         The output tensor retains the same precision (half precision) and device as the input tensor.
 *
 * Notes:
 * - The function utilizes custom CUDA kernels optimized for half precision and 4-bit quantization, aiming to leverage
 *   the computational efficiency of modern NVIDIA GPUs.
 * - The quantization scheme assumes that the weights are pre-quantized and packed into 32-bit integers, where each integer
 *   contains 8 values of 4 bits each.
 * - The `scales` tensor provides a per-channel scaling factor necessary for quantization, allowing the restoration of
 *   the tensor to a representation closer to its full-precision counterpart.
 * - This function includes checks to ensure that the input tensor `x` is of the correct data type and that the dimensions
 *   of `x` and `qweight` align according to the expected matrix multiplication rules.
 */
torch::Tensor mbwq_linear_q4_forward_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size,
    torch::Tensor q_perm,
    int bits);


/**
 * Converts exl2 format quantized weights into half-precision floating point format using a custom linear reconstruction kernel.
 * This function is designed for CUDA execution, utilizing specific memory layouts and quantization schemes.
 *
 * @param qweight Tensor representing quantized weights, typically stored in an integer format.
 * @param scales Tensor containing scales for quantization levels.
 * @param zeros Tensor representing quantized zero points.
 * @param q_perm Tensor representing permutation indices for quantized weight groups.
 * @param q_group_map Tensor mapping each weight to its corresponding quantization group.
 * @param rows Vector<int> specifying the distribution of rows across different quantization precisions.
 *
 * The function orchestrates the execution of a CUDA kernel to perform the linear reconstruction of quantized weights
 * back into a half-precision floating-point format. This operation is crucial for models that utilize mixed precision
 * to balance performance and memory usage while ensuring the precision requirements of the computation are met.
 *
 * The CUDA kernel, `reconstruct_exl2_kernel`, is called with dynamically calculated block and grid dimensions based on
 * the input tensor sizes and the specified quantization group structure.
 *
 * @return A Tensor of half-precision floating-point numbers representing the reconstructed weights, ready for use
 *         in further computation or storage.
 */
torch::Tensor mbwq_linear_exl2fp_weight_cuda(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor q_perm,
    torch::Tensor q_group_map,
    std::vector<int> rows);


/**
 * Performs a forward pass of a mixed-precision, quantized linear layer on CUDA.
 *
 * This function applies a linear transformation to the input data `x` using quantized weights `qweight`
 * and a set of quantization parameters (scales and zeros for both weights and inputs) to produce the output tensor.
 * The computation is optimized for CUDA and uses half precision for inputs and outputs, with quantization
 * parameters allowing for efficient mixed-precision computation.
 *
 * @param x Input tensor of shape (m, k) in half precision.
 * @param qweight Quantized weight tensor, packed into 32-bit integers.
 * @param scales Scales for quantizing the weights.
 * @param zeros Scales for quantizing the inputs.
 * @param q_perm Permutation indices for quantized weight matrix, supporting optimized memory access patterns.
 * @param q_group_map Mapping tensor that associates groups with quantization parameters.
 * @param rows Vector of integers specifying the row counts for different kernel configurations.
 * @param use_cublas indicates whether use cublas for matmul computation
 *
 * @return A tensor containing the result of the linear transformation applied to `x`, in half precision.
 *
 * Note: The function requires CUDA and is designed to be called within a CUDA context. It leverages specialized
 * CUDA kernels for efficient execution, and the choice of kernel is determined by the `rows` parameter, which
 * specifies the configuration for different block sizes and quantization strategies.
 *
 * The implementation uses CUDA's OptionalCUDAGuard to ensure the operation is performed on the correct device,
 * and checks are performed to ensure the input tensor `x` is of the correct data type (half precision).
 */
torch::Tensor mbwq_linear_exl2_forward_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor q_perm,
    torch::Tensor q_group_map,
    std::vector<int> rows,
    bool use_cublas = false);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);

torch::Tensor mpq_linear_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    int a_bit,
    int w_bit,
    bool asym) {
    CHECK_INPUT(x);
    CHECK_INPUT(qweight);
  return mpq_linear_cuda_forward(x, qweight, scales, qzeros, g_idx, a_bit, w_bit, asym);
}

torch::Tensor mpq_linear_grad_input(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    torch::Tensor output_gradient,
    int a_bit,
    int w_bit,
    bool asym) {
    CHECK_INPUT(output_gradient);
    CHECK_INPUT(qweight);
  return mpq_linear_cuda_grad_input(qweight, scales, qzeros, g_idx, output_gradient, a_bit, w_bit, asym);
}

std::pair<torch::Tensor, std::vector<int>>  mbwq_linear_trans_qweight(
    torch::Tensor qweight,
    torch::Tensor cuda_q_groups,
    bool use_mbw,
    int height,
    int groups,
    int bits) {
    CHECK_INPUT(qweight);
    CHECK_INPUT(cuda_q_groups);
  return mbwq_linear_trans_qweight_cuda(qweight, cuda_q_groups, use_mbw, height, groups, bits);
}

torch::Tensor mbwq_linear_q42fp_weight(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size,
    int bits,
    torch::Tensor q_perm) {
    CHECK_INPUT(qweight);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);
  return mbwq_linear_q42fp_weight_cuda(qweight, scales, zeros, group_size, bits, q_perm);
}

torch::Tensor mbwq_linear_q4_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size,
    torch::Tensor q_perm,
    int bits) {
    CHECK_INPUT(x);
    CHECK_INPUT(qweight);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);
  return mbwq_linear_q4_forward_cuda(x, qweight, scales, zeros, group_size, q_perm, bits);
}


torch::Tensor mbwq_linear_exl2fp_weight(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor q_perm,
    torch::Tensor q_group_map,
    std::vector<int> rows
    ) {
    CHECK_INPUT(qweight);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);
  return mbwq_linear_exl2fp_weight_cuda(qweight, scales, zeros, q_perm, q_group_map, rows);
}


torch::Tensor mbwq_linear_exl2_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor q_perm,
    torch::Tensor q_group_map,
    std::vector<int> rows,
    bool use_cublas = false) {
    CHECK_INPUT(x);
    CHECK_INPUT(qweight);
  return mbwq_linear_exl2_forward_cuda(x, qweight, scales, zeros, q_perm, q_group_map, rows, use_cublas);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // mpq linear layer
    m.def("mpq_forward", &mpq_linear_forward, "mixed-precision quantized linear forward (CUDA)");
    m.def("mpq_grad_input", &mpq_linear_grad_input, "mixed-precision quantized linear backward (CUDA)");

    // mbwq layer weight operation
    m.def("mbwq_trans_qweight", &mbwq_linear_trans_qweight, "transform qweight layout, preparation for forward (CUDA)");
    m.def("mbwq_q42fp_weight", &mbwq_linear_q42fp_weight, "dequantize q4-weight to fp weight (CUDA)");
    m.def("mbwq_exl2fp_weight", &mbwq_linear_exl2fp_weight, "dequantize exl2-qweight to fp weight (CUDA)");
	// mbwq layer forward function
	m.def("mbwq_q4_forward", &mbwq_linear_q4_forward, "mbwq-q4 forward (CUDA)");
    m.def("mbwq_exl2_forward", &mbwq_linear_exl2_forward, "mbwq-exl2 forward (CUDA)");
}