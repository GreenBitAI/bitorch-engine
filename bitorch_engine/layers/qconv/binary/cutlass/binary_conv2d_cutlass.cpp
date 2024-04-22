#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

/**
 * Performs a forward pass of binary convolution using the CUTLASS library.
 * This function is optimized for binary convolutions, leveraging the efficiency of CUTLASS kernels.
 *
 * Args:
 *   input (torch::Tensor): The input tensor with shape [batch_size, in_channels, in_height, in_width].
 *   weight (torch::Tensor): The filter weights tensor with shape [out_channels, kernel_size, kernel_size, in_channels].
 *   scale (float): A scaling factor applied to the output tensor.
 *   is_train (bool): A flag indicating whether the operation is being performed during training.
 *                    This influences the processing of the weight tensor.
 *   kernel_size (int): The size of the convolution kernel.
 *   stride (int): The stride of the convolution.
 *   padding (int): The padding added to the input tensor.
 *   dilation (int): The dilation factor for the convolution.
 *   device_id (int): The ID of the CUDA device on which to perform the operation.
 *
 * Returns:
 *   torch::Tensor: The output tensor of the convolution, scaled by the 'scale' parameter.
 *                  The output tensor has shape [batch_size, out_edge, out_edge, out_channels],
 *                  where 'out_edge' is computed based on the input dimensions, padding, and stride.
 *
 * Notes:
 * - The function sets the CUDA device to 'device_id' at the beginning.
 * - It calculates the output tensor dimensions based on the input size, kernel size, stride, and padding.
 * - The weights are optionally preprocessed (viewed and packed) based on the training mode.
 * - The input tensor is reshaped and packed for efficient processing.
 * - The actual convolution operation is performed by a call to the 'xnor_cutlass::_impl_conv_forward' function,
 *   which utilizes CUTLASS kernels optimized for binary convolutions.
 * - Finally, the output tensor is scaled by the 'scale' parameter before being returned.
 */
torch::Tensor binary_conv2d_cutlass_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float scale,
    bool is_train,
    int kernel_size,
    int stride,
    int padding,
    int dilation);


/**
 * Performs binary convolution operation with weight packing using CUTLASS.
 *
 * This function adapts the input data tensor for binary convolution by rearranging its dimensions to match
 * the expected format {OHWC} (Output Channels, Height, Width, Input Channels) and then packs the data to optimize
 * the convolution operation. It leverages CUTLASS kernels for efficient computation.
 *
 * Args:
 *    data (torch::Tensor): The input tensor for the convolution operation. Expected to have dimensions
 *                          {Output Channels, Input Channels, Kernel Height, Kernel Width}.
 *
 * Returns:
 *    torch::Tensor: A tensor containing the packed data, ready for efficient binary convolution with CUTLASS.
 */
torch::Tensor binary_conv2d_w_pack_cutlass(
	torch::Tensor data);


// C++ interface


// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);


torch::Tensor binary_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float scale,
    bool is_train,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
  return binary_conv2d_cutlass_forward(input, weight, scale, is_train, kernel_size, stride, padding, dilation);
}


torch::Tensor binary_conv2d_w_pack(
    torch::Tensor data
){
    CHECK_INPUT(data);
    return binary_conv2d_w_pack_cutlass(data);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &binary_conv2d_forward, "scaled binary conv2d forward (CUTLASS)");
    m.def("w_pack", &binary_conv2d_w_pack, "packing binary weight (CUTLASS)");
}
