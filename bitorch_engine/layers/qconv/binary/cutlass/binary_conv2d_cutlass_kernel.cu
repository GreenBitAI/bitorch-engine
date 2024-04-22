#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/tensor_view.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>

typedef uint8_t BINARY_WORD; // 8-bit binary word
const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);
const int THREADS_PER_BLOCK = 512;

#define CUTLASS_CHECK(status)                                                                          \
	{                                                                                                  \
		cutlass::Status error = status;                                                                \
		if (error != cutlass::Status::kSuccess) {                                                      \
			std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
			        << std::endl;                                                                      \
			exit(EXIT_FAILURE);                                                                        \
		}                                                                                              \
	}

#define CUDA_CHECK(status)                                                    \
	{                                                                         \
		cudaError_t error = status;                                           \
		if (error != cudaSuccess) {                                           \
			std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
			        << " at line: " << __LINE__ << std::endl;                 \
			exit(EXIT_FAILURE);                                               \
		}                                                                     \
	}


namespace xnor_cuda {

__device__ BINARY_WORD bit_packing(__nv_bfloat16* array){
    BINARY_WORD rvalue=0;
    BINARY_WORD sign;
#pragma unroll
    for (int i = 0; i < BITS_PER_BINARY_WORD; i++){
        sign = (array[i] >= __float2bfloat16(0.0f));
        rvalue |= (sign << i);
    }
    return rvalue;
}

__device__ BINARY_WORD bit_packing(__half* array){
    BINARY_WORD rvalue=0;
    BINARY_WORD sign;
#pragma unroll
    for (int i = 0; i < BITS_PER_BINARY_WORD; i++){
        sign = (__hge(array[i], __float2half(0.0f)));
        rvalue |= (sign << i);
    }
    return rvalue;
}

// 32 single float array ->  8 bits BINARY_WORD
template <typename DstScalar>
__device__ BINARY_WORD bit_packing(DstScalar* array)
{
    BINARY_WORD rvalue=0;
    BINARY_WORD sign;
    #pragma unroll
    for (int i = 0; i < BITS_PER_BINARY_WORD; i++){
        sign = (array[i]>=0);
        rvalue |= (sign << i);
    }
    return rvalue;
}

template <typename DstScalar>
__global__ void _to_uint8_array_baseline(DstScalar *a, uint8_t *b, int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x)
    {
        BINARY_WORD bw = bit_packing(&a[i*BITS_PER_BINARY_WORD]); //BITS_PER_BINARY_WORD=8 for uint8_t
        b[i] = bw;
    }
}

template <typename DstScalar>
__global__ void _to_uint8_array(DstScalar *a, uint8_t *b, int size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    __shared__ DstScalar shared_a[THREADS_PER_BLOCK * BITS_PER_BINARY_WORD];

    for (int i = tid; i < size; i += stride)
    {
        // Load data into shared memory in a coalesced manner
        #pragma unroll
        for (int j = 0; j < BITS_PER_BINARY_WORD; j++)
        {
            shared_a[threadIdx.x * BITS_PER_BINARY_WORD + j] = a[i * BITS_PER_BINARY_WORD + j];
        }
        __syncthreads(); // Ensure all data is loaded

        // Perform bit packing using shared memory
        BINARY_WORD bw = bit_packing(&shared_a[threadIdx.x * BITS_PER_BINARY_WORD]);

        // Store the result
        b[i] = bw;
    }
}

} // namespace cuda


namespace xnor_cutlass {

/// The code section below describes datatype for input, output tensors and computation between
/// elements
using ElementAccumulator = int32_t;                 // Data type of accumulator
using ElementComputeEpilogue = float;               // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::uint1b_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::uint1b_t;             // Data type of elements in input tensor
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
	const int NumStages = 3;
	using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 512>;  // Threadblock tile shape
	using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;         // Warp tile shape
	using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;    // TensorCore instruction shape
#endif // end of ifdef ARCH_SM_80
#ifdef ARCH_SM_75 // sm_75
	using SmArch = cutlass::arch::Sm75;
	const int NumStages = 2;
	using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 512>;  // Threadblock tile shape
	using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;         // Warp tile shape
	using InstructionShape = cutlass::gemm::GemmShape<8, 8, 128>;    // TensorCore instruction shape
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
	cutlass::arch::OpXorPopc,
	IteratorAlgorithm
>::Kernel;
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

//============= end cutlass kernel configuration ===============//

cudaError_t _impl_conv_forward(
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

    cutlass::uint1b_t* pA = reinterpret_cast<cutlass::uint1b_t *>(input.data_ptr<uint8_t>());
    cutlass::uint1b_t* pB = reinterpret_cast<cutlass::uint1b_t *>(weight.data_ptr<uint8_t>());

    cutlass::Tensor4DCoord
        input_size(
            input.sizes()[0],
            input.sizes()[1],
            input.sizes()[2],
            input.sizes()[3]
        );
    cutlass::Tensor4DCoord
        filter_size(
            cutlass::Tensor4DCoord(
                weight.sizes()[0],
                weight.sizes()[1],
                weight.sizes()[2],
                weight.sizes()[3]
            )
        );
    cutlass::Tensor4DCoord
        output_size(
            cutlass::Tensor4DCoord(
                output.sizes()[0],
                output.sizes()[1],
                output.sizes()[2],
                output.sizes()[3]
            )
        );
    cutlass::Tensor4DCoord padding(pad, pad, pad, pad);
    cutlass::MatrixCoord conv_stride(stride, stride);
    cutlass::MatrixCoord dilation(dila, dila);

    cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a(
        pA,
        cutlass::layout::TensorNHWC::packed(input_size)
    ); ///NHWC
    cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b(
        pB,
        cutlass::layout::TensorNHWC::packed(filter_size)
    ); ///NHWC
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c(
        (int32_t*)output.data_ptr<int32_t>(),
        cutlass::layout::TensorNHWC::packed(output_size)
    ); ///NHWC
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c(
        (int32_t*)output.data_ptr<int32_t>(),
        cutlass::layout::TensorNHWC::packed(output_size)
    ); ///NHWC

    //
    // Define arguments for CUTLASS Convolution
    //
    // mode (kCrossCorrelation or kConvolution)
    cutlass::conv::Mode mode = cutlass::conv::Mode::kConvolution;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0f);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0.0f);

    // Construct Conv2dProblemSize with user defined output size
    cutlass::conv::Conv2dProblemSize
        problem_size(
	        input_size,
	        filter_size,
	        padding,
	        conv_stride,
	        dilation,
	        output_size,
	        mode,
	        split_k_slices
        );

    // Construct ImplicitGemm::Argument structure with conv2d
    // problem size, data pointers, and epilogue values
    typename ImplicitGemm::Arguments
        arguments{
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
    cutlass::Status status;
    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    status = implicit_gemm_op.can_implement(arguments);
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
    return cudaSuccess;
}

} // namespace xnor_cutlass


// C++-CUDA methods

template <typename DstScalar>
torch::Tensor get_pack_cuda_kernel(
    torch::Tensor input
){
	auto input_size = input.sizes().size();
	if (input_size != 4){
        std::cerr << "tensor sizes not supported: " << input_size << std::endl;
        exit(EXIT_FAILURE);
	}

    // shape: {NHWC}
    auto option_quantize = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    torch::Tensor w_quant =
        torch::empty(
            {
                input.sizes()[0],
                input.sizes()[1],
                input.sizes()[2],
                input.sizes()[3]/BITS_PER_BINARY_WORD
            },
            option_quantize
        );

    int w_quant_size = input.numel()/BITS_PER_BINARY_WORD;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
	dim3 grid_dim( input.numel() / (BITS_PER_BINARY_WORD * THREADS_PER_BLOCK) + 1, 1);
	xnor_cuda::_to_uint8_array<DstScalar><<<grid_dim, block_dim>>>(
	                        reinterpret_cast<DstScalar *>(input.data_ptr()),
	                        w_quant.data_ptr<uint8_t>(),
	                        w_quant_size);
    return w_quant;
}


torch::Tensor get_packed_data_tensor(
    const torch::Tensor data
){
    torch::Tensor pack_qw;
    if(data.dtype() == torch::kInt8){
        pack_qw = get_pack_cuda_kernel<int8_t>(data);
    }else if(data.dtype() == torch::kFloat32){
        pack_qw = get_pack_cuda_kernel<float>(data);
    }else if(data.dtype() == torch::kBFloat16){
        pack_qw = get_pack_cuda_kernel<__nv_bfloat16>(data);
    }else if(data.dtype() == torch::kHalf){
        pack_qw = get_pack_cuda_kernel<__half>(data);
    }else{
        std::cerr << "tensor type not supported: " << data.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }
    return pack_qw;
}


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
    int dilation
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    int batch_size = input.sizes()[0];
    int in_c = input.sizes()[1];
    int in_h = input.sizes()[2];
    int in_w = input.sizes()[3];

    ///(Image_w – filter_w + 2*pad_w) / stride + 1
    int out_edge = (in_w - kernel_size + 2 * padding) / stride + 1;
    int out_C = weight.sizes()[0];

    ///NHWC
    auto output =
        torch::empty(
            {batch_size, out_edge, out_edge, out_C},
            torch::TensorOptions().dtype(torch::kInt32).device(input.device())
        );

    torch::Tensor packed_w;
    if(is_train){
        weight =
            weight.view(
                {out_C, kernel_size, kernel_size, in_c}
            );
        packed_w = get_packed_data_tensor(weight);
    } else {
        packed_w = weight;
	}

	input = input.view({batch_size, in_h, in_w, in_c});
	torch::Tensor packed_a = get_packed_data_tensor(input);

    xnor_cutlass::_impl_conv_forward(
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
    return output*scale;
}


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
    torch::Tensor data
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));

    // adapt dimension to {OHWC}
    int out_C = data.sizes()[0];
    int in_c = data.sizes()[1];
    int kernel_size = data.sizes()[2];
    data = data.view({out_C, kernel_size, kernel_size, in_c});
    return get_packed_data_tensor(data);
}