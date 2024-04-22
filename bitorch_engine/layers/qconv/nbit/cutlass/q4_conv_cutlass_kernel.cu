#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/tensor_view.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>

#define NUM_THREADS 256

#define CUTLASS_CHECK(status)                                                                    \
{                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
}

#define CUDA_CHECK(status)                                              \
{                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
}


template <typename T>
__global__ void output_scaling_kernel(T* input, T scale, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        input[i] = input[i] * scale;
    }
}


__global__ void output_scaling_kernel(__nv_bfloat16* input, __nv_bfloat16 scale, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        input[i] = __hmul_rn(input[i], scale); // a * b
    }
}


__global__ void output_scaling_kernel(__half* input, __half scale, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        input[i] = __hmul_rn(input[i], scale); // a * b
    }
}


__global__ void q4_quantization_and_bit_packing_kernel(float *input, float scale_a, int8_t *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx * 2; // Index for the pair of inputs
    float eps = 0.00001f;

    if (pair_idx < N) {
        unsigned char packedValue = 0;

        // Handle the first of the pair
        if (pair_idx < N) {
            float scale = max(scale_a, eps); // Avoid division by zero or values too close to zero
            float quantizedValueFirst = round(input[pair_idx] / scale); // Scale and round
            quantizedValueFirst = fmin(fmax(quantizedValueFirst, -8.0), 7.0); // Clamp to the range [-8, 7]
            int qValueFirst = static_cast<int>(quantizedValueFirst) & 0xF; // Keep only the lower 4 bits
            packedValue |= qValueFirst << 4; // Place it in the high 4 bits
        }

        // Handle the second of the pair
        if (pair_idx + 1 < N) {
            float scale = max(scale_a, eps); // Avoid division by zero or values too close to zero
            float quantizedValueSecond = round(input[pair_idx + 1] / scale); // Scale and round
            quantizedValueSecond = fmin(fmax(quantizedValueSecond, -8.0), 7.0); // Clamp to the range [-8, 7]
            int qValueSecond = static_cast<int>(quantizedValueSecond) & 0xF; // Keep only the lower 4 bits
            packedValue |= qValueSecond; // Place it in the low 4 bits
        }

        // Write the packed value directly to output, no need for atomicExch
        output[idx] = packedValue;
    }
}


__global__ void q4_quantization_and_bit_packing_kernel(__half *input, __half scale_a, int8_t *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx * 2; // Index for the pair of inputs
    __half eps = __float2half(0.00001f);

    if (pair_idx < N) {
        unsigned char packedValue = 0;

        // Handle the first of the pair
        if (pair_idx < N) {
            __half scale = __hmax(scale_a, eps); // Avoid division by zero or values too close to zero
            float quantizedValueFirst = roundf(__half2float(__hdiv(input[pair_idx], scale)));
            quantizedValueFirst = fminf(fmaxf(quantizedValueFirst, -8.0f), 7.0f); // Clamp to the range [-8, 7]
            int qValueFirst = static_cast<int>(quantizedValueFirst) & 0xF; // Keep only the lower 4 bits
            packedValue |= qValueFirst << 4; // Place it in the high 4 bits
        }

        // Handle the second of the pair
        if (pair_idx + 1 < N) {
            __half scale = __hmax(scale_a, eps); // Avoid division by zero or values too close to zero
            float quantizedValueSecond = roundf(__half2float(__hdiv(input[pair_idx + 1], scale)));
            quantizedValueSecond = fminf(fmaxf(quantizedValueSecond, -8.0f), 7.0f); // Clamp to the range [-8, 7]
            int qValueSecond = static_cast<int>(quantizedValueSecond) & 0xF; // Keep only the lower 4 bits
            packedValue |= qValueSecond; // Place it in the low 4 bits
        }

        // Write the packed value directly to output, no need for atomicExch
        output[idx] = packedValue;
    }
}


__global__ void q4_quantization_and_bit_packing_kernel(__nv_bfloat16 *input, __nv_bfloat16 scale_a, int8_t *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx * 2; // Index for the pair of inputs

    __nv_bfloat16 eps = __float2bfloat16(0.00001f);

    if (pair_idx < N) {
        unsigned char packedValue = 0;

        // Handle the first of the pair
        if (pair_idx < N) {
            __nv_bfloat16 scale = __hmax(scale_a, eps); // Avoid division by zero or values too close to zero
            float quantizedValueFirst = roundf(__bfloat162float(input[pair_idx]) / __bfloat162float(scale)); // Scale and round
            quantizedValueFirst = fminf(fmaxf(quantizedValueFirst, -8.0f), 7.0f); // Clamp to the range [-8, 7]
            int qValueFirst = static_cast<int>(quantizedValueFirst) & 0xF; // Keep only the lower 4 bits
            packedValue |= qValueFirst << 4; // Place it in the high 4 bits
        }

        // Handle the second of the pair
        if (pair_idx + 1 < N) {
            __nv_bfloat16 scale = __hmax(scale_a, eps); // Avoid division by zero or values too close to zero
            float quantizedValueSecond = roundf(__bfloat162float(input[pair_idx + 1]) / __bfloat162float(scale)); // Scale and round
            quantizedValueSecond = fminf(fmaxf(quantizedValueSecond, -8.0f), 7.0f); // Clamp to the range [-8, 7]
            int qValueSecond = static_cast<int>(quantizedValueSecond) & 0xF; // Keep only the lower 4 bits
            packedValue |= qValueSecond; // Place it in the low 4 bits
        }

        // Write the packed value directly to output, no need for atomicExch
        output[idx] = packedValue;
    }
}


/// The code section below describes datatype for input, output tensors and computation between
/// elements
using ElementAccumulator = int32_t;                 // Data type of accumulator
using ElementComputeEpilogue = int32_t;               // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::int4b_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::int4b_t;             // Data type of elements in input tensor
using ElementOutput = int32_t;                      // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;
// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;
// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput,                                     // Data type of output matrix.
	128 / cutlass::sizeof_bits<ElementOutput>::value,   // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

#ifdef ARCH_SM_80 // sm_80
	using SmArch = cutlass::arch::Sm80;
	using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
	using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
	using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
	const int NumStages = 3;
#endif // end of ifdef ARCH_SM_80

#ifdef ARCH_SM_75 // sm_75
	using SmArch = cutlass::arch::Sm75;
	using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;  // Threadblock tile shape
	using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;         // Warp tile shape
	using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;    // TensorCore instruction shape
	const int NumStages = 2;
#endif // end of ifdef ARCH_SM_75

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAddSaturate,
  IteratorAlgorithm
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


/////////////////////////////////////////////////////////////////////////////////////////////////



// C++-CUDA methods
//
cudaError_t _4b_conv_forward(
        torch::Tensor  input,
        torch::Tensor  weight,
        torch::Tensor  output,
        int kernel_size,
        int stride,
        int pad,
        int dila,
        int batch_size,
        int out_edge,
        int out_C
){
    cutlass::int4b_t* pA = reinterpret_cast<cutlass::int4b_t *>(input.data_ptr<int8_t>());
    cutlass::int4b_t* pB = reinterpret_cast<cutlass::int4b_t *>(weight.data_ptr<int8_t>());

    cutlass::Tensor4DCoord input_size(input.sizes()[0], input.sizes()[1], input.sizes()[2], input.sizes()[3]);
    cutlass::Tensor4DCoord filter_size(cutlass::Tensor4DCoord(weight.sizes()[0], weight.sizes()[1],
                                                            weight.sizes()[2], weight.sizes()[3]));
    cutlass::Tensor4DCoord output_size(cutlass::Tensor4DCoord(output.sizes()[0],output.sizes()[1], output.sizes()[2],
                                                            output.sizes()[3]));
    cutlass::Tensor4DCoord padding(pad, pad, pad, pad);
    cutlass::MatrixCoord conv_stride(stride, stride);
    cutlass::MatrixCoord dilation(dila, dila);

    cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a( pA,
                                                            cutlass::layout::TensorNHWC::packed(input_size)); ///NHWC
    cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b( pB,
                                                            cutlass::layout::TensorNHWC::packed(filter_size)); ///NHWC
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c( (int32_t*)output.data_ptr<int32_t>(),
                                                        cutlass::layout::TensorNHWC::packed(output_size)); ///NHWC
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c( (int32_t*)output.data_ptr<int32_t>(),
                                                        cutlass::layout::TensorNHWC::packed(output_size)); ///NHWC

//     /// update padding
//     if (filter_size.h() == 3 && filter_size.w() == 3) {
//         padding = {1, 1, 1, 1};
//     }
//     else {
//         filter_size.h() = 1;
//         filter_size.w() = 1;
//         padding = {0, 0, 0, 0};
//     }
//     padding.n() = filter_size.h() / 2;
//     padding.h() = filter_size.h() / 2;
//     padding.w() = filter_size.w() / 2;
//     padding.c() = filter_size.w() / 2;

    //
    // Define arguments for CUTLASS Convolution
    //
    // mode (kCrossCorrelation or kConvolution)
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0f);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0.0f);

    // Construct Conv2dProblemSize with user defined output size
    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        mode,
        split_k_slices);

    // Construct ImplicitGemm::Argument structure with conv2d
    // problem size, data pointers, and epilogue values
    typename ImplicitGemm::Arguments arguments{
        problem_size,
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_c,
        {alpha, beta},
    };

    //
    // Initialize CUTLASS Convolution
    //
    ImplicitGemm implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    //
    // Launch initialized CUTLASS kernel
    //
    status = implicit_gemm_op();
    CUTLASS_CHECK(status);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}


torch::Tensor get_q4_packed_data_tensor_cuda(
    const torch::Tensor data,
    float scale
){

	auto input_size = data.sizes().size();
	if (input_size != 4){
        std::cerr << "tensor sizes not supported: " << input_size << std::endl;
        exit(EXIT_FAILURE);
	}

    auto option_quantize = torch::TensorOptions().dtype(torch::kInt8).device(data.device());
    // shape: {NHWC}
    torch::Tensor pack_qw =
        torch::empty(
            {data.sizes()[0], data.sizes()[1], data.sizes()[2], data.sizes()[3]>>1},
            option_quantize
        );

    int w_quant_size = data.numel();
    dim3 block(NUM_THREADS);
    dim3 grid((w_quant_size-1)/(block.x)+1);


    // quantization and pack into 4-bit
    if(data.dtype() == torch::kFloat32){
	    q4_quantization_and_bit_packing_kernel<<<grid, block>>>(
	        data.data_ptr<float>(),
	        scale,
	        pack_qw.data_ptr<int8_t>(),
	        w_quant_size);
	} else if(data.dtype() == torch::kBFloat16){
	    q4_quantization_and_bit_packing_kernel<<<grid, block>>>(
	        reinterpret_cast<__nv_bfloat16 *>(data.data_ptr()),
	        __float2bfloat16(scale),
	        pack_qw.data_ptr<int8_t>(),
	        w_quant_size);
	} else if(data.dtype() == torch::kHalf){
	    q4_quantization_and_bit_packing_kernel<<<grid, block>>>(
	        reinterpret_cast<__half *>(data.data_ptr()),
	        __float2half(scale),
	        pack_qw.data_ptr<int8_t>(),
	        w_quant_size);
	} else {
        std::cerr << "tensor type not supported: " << data.dtype() << std::endl;
        exit(EXIT_FAILURE);
	}

    return pack_qw;
}


template <typename T>
torch::Tensor get_scaled_output(
	torch::Tensor output,
	T scale
){
	// 1. multiply scaling factor: out = out * scale_a
    int size = output.numel();
    dim3 block(256);
    dim3 grid((size - 1) / 256 + 1);

    output_scaling_kernel<<<grid, block>>>(
        reinterpret_cast<T *>(output.data_ptr()),
        scale,
        size);
	return output;
}


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
    int dilation
) {
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    int batch_size = input.sizes()[0];
    int in_c = input.sizes()[1];
    int in_h = input.sizes()[2];
    int in_w = input.sizes()[3];
    //(Image_w – filter_w + 2*pad_w) / stride + 1
    int out_edge = (in_w - kernel_size + 2 * padding) / stride + 1;
    int out_C = weight.sizes()[0];

    ///NHWC
    auto output =
        torch::empty(
            {batch_size, out_edge, out_edge, out_C},
            torch::TensorOptions().dtype(torch::kInt32).device(input.device())
        );

    // handle input and weight tensor shapes.
    // the cutlass conv implmentation only supports the tensor shape:{NHWC}
    // we need to convert from {NCHW} -> {NHWC}
    torch::Tensor packed_w;
    if(is_train){
        weight = weight.view({out_C, kernel_size, kernel_size, in_c});
        packed_w = get_q4_packed_data_tensor_cuda(weight, scale_w);
    } else {
        packed_w = weight;
	}

    input = input.view({batch_size, in_h, in_w, in_c});
	torch::Tensor packed_a = get_q4_packed_data_tensor_cuda(input, scale_a);

    _4b_conv_forward(
        packed_a,
	    packed_w,
	    output,
	    kernel_size,
	    stride,
	    padding,
	    dilation,
	    batch_size,
	    out_edge,
	    out_C
    );

    if(input.dtype() == torch::kFloat32){
        output = get_scaled_output<float>(output.to(input.dtype()), scale_a*scale_w);
    }else if(input.dtype() == torch::kBFloat16){
        output = get_scaled_output<__nv_bfloat16>(output.to(input.dtype()), __float2bfloat16(scale_a*scale_w));
    }else if(input.dtype() == torch::kHalf){
        output = get_scaled_output<__half>(output.to(input.dtype()), __float2half(scale_a*scale_w));
    }else{
        std::cerr << "tensor type not supported: " << input.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<torch::Tensor> outputs;
    outputs.push_back(output);
    outputs.push_back(packed_a);
    outputs.push_back(packed_w);
    return outputs;
}


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
    float scale
){
	const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));

    int out_C = weight.sizes()[0];
    int in_c = weight.sizes()[1];
    int kernel_size = weight.sizes()[2];

    // adapt dimension
    weight = weight.view({out_C, kernel_size, kernel_size, in_c});
	auto packed_w = get_q4_packed_data_tensor_cuda(weight, scale);

	return packed_w;
}