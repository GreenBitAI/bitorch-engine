#include <torch/extension.h>
#include <vector>


// CUTLASS forward declarations

/**
 * Performs a forward pass of the quantized 4-bit convolution (q4_conv2d) using the CUTLASS library.
 * This function takes a 4-bit quantized input and weight tensors, along with convolution parameters
 * like scale factors, kernel size, stride, padding, and dilation, to perform the convolution operation
 * optimized for CUDA. It's designed to work with NHWC tensor format for efficient computation.
 *
 * Parameters:
 *   input - The input tensor in NCHW format that will be converted to NHWC internally.
 *   weight - The weight tensor, which can be either pre-packed (in inference mode) or will be packed during training.
 *   scale_a - The scale factor for the input tensor quantization.
 *   scale_w - The scale factor for the weight tensor quantization.
 *   is_train - A boolean flag indicating whether the operation is for training. Affects weight processing.
 *   kernel_size - The size of the convolution kernel.
 *   stride - The stride of the convolution.
 *   padding - The padding added to both sides of the input tensor.
 *   dilation - The spacing between kernel elements.
 *
 * Returns:
 *   A tensor containing the result of the quantized convolution operation, scaled by the product of input
 *   and weight scale factors.
 */
std::vector<torch::Tensor> q4_conv2d_cutlass_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float scale_a,
    float scale_w,
    bool is_train,
    int kernel_size,
    int stride,
    int padding,
    int dilation);


/**
 *
 * This function prepares the weight tensor for a quantized 4-bit convolution operation.
 * It takes a weight tensor and a scale factor as inputs, restructures the weight tensor for the
 * convolution operation, and quantizes it to 4 bits. This preparation is crucial for leveraging
 * CUTLASS's efficient low-bit computation capabilities.
 *
 * Parameters:
 * - weight: The original weight tensor of the convolutional layer.
 * - scale: The scaling factor used for quantization of the weights to 4-bit precision.
 *
 * Returns:
 * - A tensor representing the packed and quantized weights, ready for use in a 4-bit convolution operation.
 */
torch::Tensor q4_conv2d_w_pack_cutlass(
    torch::Tensor weight,
    float scale);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);


std::vector<torch::Tensor> q4_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float scale_a,
    float scale_w,
    bool is_train,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
	return q4_conv2d_cutlass_forward(input, weight, scale_a, scale_w, is_train,
										kernel_size, stride, padding, dilation);
}


torch::Tensor q4_conv2d_w_pack(
    torch::Tensor weight,
    float scale
    ) {
    CHECK_INPUT(weight);
	return q4_conv2d_w_pack_cutlass(weight, scale);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &q4_conv2d_forward, "4-bit conv forward (CUTLASS)");
  m.def("w_pack", &q4_conv2d_w_pack, "pack q4 weight");
}
