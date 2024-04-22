#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <string.h>
#include <iostream>
#include <algorithm>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/tensor_view.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>

#include <vector_types.h>
#include <cmath>
#include <curand_kernel.h>
#include <tuple>
#include <bits/stdc++.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <torch/script.h>
using namespace torch::indexing;

// standard number of threads per block to use
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


//=================== Cutlass configuration ==================//
// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::int4b_t;                       // <- data type of elements in input matrix A
using ElementInputB = cutlass::int4b_t;                       // <- data type of elements in input matrix B
using ElementOutput = int32_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
using SwizzleThreadBlockBatched = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    64 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function
// ===============================================================//

cudaError_t CutlassSgemmNN(
    const int M,
    const int N,
    const int K,
    const cutlass::int4b_t *A,
    int lda,
    const cutlass::int4b_t *B,
    int ldb,
    int32_t *C,
    int ldc) {
#ifdef ARCH_SM_75 // sm_75
	const int NumStages = 2;
	using SmArch = cutlass::arch::Sm75;
	using ShapeMMAThreadBlock =
	    cutlass::gemm::GemmShape<256, 128, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
	using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 64
	using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 32>;  // <- MMA Op tile M = 8, N = 8, K = 16
#endif // end of ifdef ARCH_SM_75

#ifdef ARCH_SM_80 // sm_80
	const int NumStages = 3;
	using SmArch = cutlass::arch::Sm80;
	using ShapeMMAThreadBlock =
	    cutlass::gemm::GemmShape<128, 128, 256>;  // <- threadblock tile M = 256, N = 128, K = 128
	using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 256>;  // <- warp tile M = 64, N = 64, K = 128
	using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;  // <- MMA Op tile M = 16, N = 8, K = 64
#endif // end of ifdef ARCH_SM_80

using Gemm =
	cutlass::gemm::device::Gemm<
		ElementInputA,
		LayoutInputA,
		ElementInputB,
		LayoutInputB,
		ElementOutput,
		LayoutOutput,
		ElementAccumulator,
		MMAOp,
		SmArch,
		ShapeMMAThreadBlock,
		ShapeMMAWarp,
		ShapeMMAOp,
		EpilogueOp,
		SwizzleThreadBlock,
		NumStages>;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments
        arguments{
            problem_size,  // <- problem size of matrix multiplication
			{A, lda},  // <- reference to matrix A on device
			{B, ldb},  // <- reference to matrix B on device
			{C, ldc},  // <- reference to matrix C on device
			{C, ldc},  // <- reference to matrix D on device
			{alpha, beta},          // <- tuple of alpha and beta
			split_k_slices};        // <- k-dimension split factor

    Gemm gemm_op;
    cutlass::Status status = gemm_op(arguments);
    CUTLASS_CHECK(status);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}


cudaError_t BatchedCutlassSgemmNN(
    const int M,
    const int N,
    const int K,
    const cutlass::int4b_t *A,
    int lda,
    const cutlass::int4b_t *B,
    int ldb,
    int32_t *C,
    int ldc,
    long long int batch_stride_A,
	long long int batch_stride_B,
	long long int batch_stride_C,
	int bs) {

#ifdef ARCH_SM_75 // sm_75
	// Number of pipelines you want to use
	const int NumStages = 2;
	// This code section describes CUDA SM architecture number
	using SmArch = cutlass::arch::Sm75;

	// This code section describes the tile size a thread block will compute
	using ShapeMMAThreadBlock =
	    cutlass::gemm::GemmShape<256, 128, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
	// This code section describes tile size a warp will compute
	using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 64
	// This code section describes the size of MMA op
	using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 32>;  // <- MMA Op tile M = 8, N = 8, K = 16

#endif // end of ifdef ARCH_SM_75

#ifdef ARCH_SM_80 // sm_80
	const int NumStages = 3;
	using SmArch = cutlass::arch::Sm80;

	// This code section describes the tile size a thread block will compute
	using ShapeMMAThreadBlock =
	    cutlass::gemm::GemmShape<64, 64, 256>;  // <- threadblock tile M = 256, N = 128, K = 128
	// This code section describes tile size a warp will compute
	using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 256>;  // <- warp tile M = 64, N = 64, K = 128
	// This code section describes the size of MMA op
	using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;  // <- MMA Op tile M = 16, N = 8, K = 64

#endif // end of ifdef ARCH_SM_80

using Gemm =
	cutlass::gemm::device::GemmBatched<
		ElementInputA,
		LayoutInputA,
		ElementInputB,
		LayoutInputB,
		ElementOutput,
		LayoutOutput,
		ElementAccumulator,
		MMAOp,
		SmArch,
		ShapeMMAThreadBlock,
		ShapeMMAWarp,
		ShapeMMAOp,
		EpilogueOp,
		SwizzleThreadBlockBatched,
		NumStages>;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

	typename Gemm::Arguments arguments{
		problem_size,  // <- problem size of matrix multiplication
		{A, lda},  // <- reference to matrix A on device
		batch_stride_A,
		{B, ldb},  // <- reference to matrix B on device
		batch_stride_B,
		{C, ldc},// <- reference to matrix C on device
		batch_stride_C,
		{C, ldc},  // <- reference to matrix C on device
		batch_stride_C,
		{alpha, beta},      // <- tuple of alpha and beta
		bs
	};        // <- batch size

    Gemm gemm_op;
    cutlass::Status status = gemm_op(arguments);
    CUTLASS_CHECK(status);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}


// C++-CUDA methods


/**
 * Scales the output tensor by a given factor.
 *
 * This function applies a scaling factor to each element of the provided output tensor.
 * It is templated to support different data types for the scaling factor and output tensor.
 * The scaling operation is performed in parallel using CUDA kernels for efficiency.
 *
 * Template Parameters:
 *   T - The data type of the scaling factor (e.g., float, double).
 *
 * Parameters:
 *   output - A torch::Tensor representing the output to be scaled. The tensor is modified in-place.
 *   scale - The scaling factor of type T to be applied to each element of the output tensor.
 *
 * Returns:
 *   The scaled torch::Tensor, with each element multiplied by the scaling factor.
 *
 * Note:
 *   This function assumes that a CUDA kernel named `output_scaling_kernel` is defined elsewhere.
 *   The kernel is responsible for performing the actual scaling operation on the GPU.
 *   The `reinterpret_cast<T *>(output.data_ptr())` is used to ensure that the data pointer
 *   passed to the CUDA kernel matches the expected data type.
 */
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
    const torch::Tensor data,
    float scale,
    bool is_transpose
){
    auto option_quantize = torch::TensorOptions().dtype(torch::kInt8).device(data.device());
    auto input_size = data.sizes().size();
    auto n = data.sizes()[input_size-2]; // out_channels
    auto k = data.sizes()[input_size-1]; // in_channels

    if (input_size == 3){
        n = n * data.size(0);
	} else if (input_size != 2){
        std::cerr << "tensor sizes not supported: " << input_size << std::endl;
        exit(EXIT_FAILURE);
	}

	// every two adjusted input values will be quantized into 4bit and saved into one int8 variable.
    torch::Tensor pack_qw = torch::empty({n, k >> 1}, option_quantize);

    int N = n*k;

    dim3 block(NUM_THREADS);
    dim3 grid((N-1)/(block.x)+1);

    // quantization and pack into 4-bit
    if(data.dtype() == torch::kFloat32){
	    q4_quantization_and_bit_packing_kernel<<<grid, block>>>(
	        data.data_ptr<float>(),
	        scale,
	        pack_qw.data_ptr<int8_t>(),
	        N);
	} else if(data.dtype() == torch::kBFloat16){
	    q4_quantization_and_bit_packing_kernel<<<grid, block>>>(
	        reinterpret_cast<__nv_bfloat16 *>(data.data_ptr()),
	        __float2bfloat16(scale),
	        pack_qw.data_ptr<int8_t>(),
	        N);
	} else if(data.dtype() == torch::kHalf){
	    q4_quantization_and_bit_packing_kernel<<<grid, block>>>(
	        reinterpret_cast<__half *>(data.data_ptr()),
	        __float2half(scale),
	        pack_qw.data_ptr<int8_t>(),
	        N);
	} else {
        std::cerr << "tensor type not supported: " << data.dtype() << std::endl;
        exit(EXIT_FAILURE);
	}

	if (input_size == 3) pack_qw = pack_qw.view({data.size(0), -1, k>>1});
    if (is_transpose) pack_qw = pack_qw.transpose(-1, -2).contiguous();

    return pack_qw;
}


/**
 * Performs quantized 4-bit GEMM (General Matrix Multiply) using CUTLASS library.
 * This function is designed for operations on quantized matrices where each element is represented by 4 bits.
 * It leverages CUTLASS's efficient GEMM computation kernels for accelerated matrix multiplication on GPU.
 *
 * @param m The number of rows in the matrix A and the resulting matrix C.
 * @param n The number of columns in the matrix B and the resulting matrix C.
 * @param k The number of columns in matrix A and rows in matrix B.
 * @param q4_a The quantized 4-bit tensor representing matrix A, stored as int8_t but interpreted as 4-bit values.
 * @param q4_w The quantized 4-bit tensor representing matrix B (weights), stored as int8_t but interpreted as 4-bit values.
 *
 * @return A tensor of type int32_t containing the result of matrix multiplication. Each element in the resulting matrix
 *         is the sum of products of corresponding elements in matrices A and B, accumulated in 32-bit integer format.
 *         This accumulation in higher precision helps in maintaining the precision of quantized computations.
 *
 * Note: This function assumes that the matrices are stored in row-major format and performs the multiplication accordingly.
 *       The tensors `q4_a` and `q4_w` are expected to be on the same device (preferably GPU) for CUDA computation.
 *       The function uses CUTLASS's `CutlassSgemmNN` kernel under the hood, which performs the computation in NN (no transpose) mode.
 */
torch::Tensor q4_gemm(
    int m,
    int n,
    int k,
    const torch::Tensor q4_a,
    const torch::Tensor q4_w
){
    // cutlass GEMM
    int lda = k;
    int ldb = k;
    int ldc = n;

    auto option_gemm = torch::TensorOptions().dtype(torch::kInt32).device(q4_a.device());
    torch::Tensor gemm = torch::empty({m, n}, option_gemm);
    cudaError_t result;

    result =
        CutlassSgemmNN(
            m,
            n,
            k,
            reinterpret_cast<cutlass::int4b_t *>(q4_a.data_ptr<int8_t>()),
            lda,
            reinterpret_cast<cutlass::int4b_t *>(q4_w.data_ptr<int8_t>()),
            ldb,
            gemm.data_ptr<int32_t>(),
            ldc);

    return gemm;
}


/**
 * Performs a batched GEMM (General Matrix Multiply) operation using quantized 4-bit integers.
 * This function leverages the CUTLASS library to perform the matrix multiplication efficiently
 * on CUDA-enabled devices. It is optimized for operations where the memory is batched along the K dimension.
 *
 * @param m The number of rows in matrices A and C.
 * @param n The number of columns in matrices B and C.
 * @param k The number of columns in matrix A and rows in matrix B.
 * @param q4_a The input tensor A with quantized 4-bit integers, expected to be in the shape [m, k].
 * @param q4_w The weight matrix B (also called W), with quantized 4-bit integers, expected to be in the shape [k, n].
 * @param bs The batch size, indicating how many instances of the GEMM operation to perform.
 *
 * @return A torch::Tensor of shape [bs, m, n] containing the result of the batched GEMM operation,
 *         with each element being a 32-bit integer.
 *
 * Note:
 * - The function is specifically designed for use with CUDA-enabled devices and requires tensors to be on a CUDA device.
 * - The memory layout for the batch operation is designed along the K dimension, with specific stride calculations
 *   to accommodate this layout.
 * - This function wraps a call to the CUTLASS library's BatchedCutlassSgemmNN function, handling low-precision
 *   integer multiplication with special attention to memory strides for efficient batch processing.
 * - In case of a CUDA error during the GEMM operation, the error message is printed to standard error output.
 */
torch::Tensor batched_q4_gemm(
    int m,
    int n,
    int k,
    const torch::Tensor q4_a,
    const torch::Tensor q4_w,
    int bs
){
    // cutlass GEMM
    int lda = k;
    int ldb = k * bs;
    int ldc = n;

	// the memory is batched along K dimension
	long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(m);
	long long int batch_stride_B = static_cast<long long int>(k);
	long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(m);

    auto option_gemm = torch::TensorOptions().dtype(torch::kInt32).device(q4_a.device());
    torch::Tensor gemm = torch::empty({bs, m, n}, option_gemm);
    cudaError_t result;

    result =
        BatchedCutlassSgemmNN(
	        m,
	        n,
	        k,
	        reinterpret_cast<cutlass::int4b_t *>(q4_a.data_ptr<int8_t>()),
	        lda,
	        reinterpret_cast<cutlass::int4b_t *>(q4_w.data_ptr<int8_t>()),
	        ldb,
	        gemm.data_ptr<int32_t>(),
	        ldc,
	        batch_stride_A,
			batch_stride_B,
			batch_stride_C,
			bs);
    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
    }
    return gemm;
}


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
    bool is_train
) {
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    // get m, k, n
    int m = input.size(0);
    int k = input.size(1);
    int n;
    torch::Tensor packed_w;
    torch::Tensor output;

    n = weight.size(0); // shape(n, k)
    if(!is_train){ // packing already done
        packed_w = weight;
    }else{
        packed_w = get_q4_packed_data_tensor_cuda(weight, scale_w, transpose); // weight needs to be transposed
    }

    torch::Tensor packed_a = get_q4_packed_data_tensor_cuda(input, scale_a, false);

    // 4-bit gemm using cutlass template
    auto gemm = q4_gemm(m, n, k, packed_a, packed_w);

    if(input.dtype() == torch::kFloat32){
        output = get_scaled_output<float>(gemm.to(input.dtype()), scale_a*scale_w);
    }else if(input.dtype() == torch::kBFloat16){
        output = get_scaled_output<__nv_bfloat16>(gemm.to(input.dtype()), __float2bfloat16(scale_a*scale_w));
    }else if(input.dtype() == torch::kHalf){
        output = get_scaled_output<__half>(gemm.to(input.dtype()), __float2half(scale_a*scale_w));
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
std::pair<torch::Tensor, torch::Tensor> q4_backward(
    torch::Tensor output_gradient,
    torch::Tensor q4_a,
    torch::Tensor q4_w,
    float scale_a,
    float scale_w,
    float scale_grad
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(output_gradient));
    auto option_quantize = torch::TensorOptions().dtype(torch::kInt8).device(output_gradient.device());
    // get m, k, n
    int m = q4_a.size(0);
    int k = q4_a.size(1) * 2; // note that here the k after packing in forward is actually k/2
    int n = q4_w.size(0); // (n, k)

    // calculate 4-bit packed gradient (m,n)
    torch::Tensor packed_output_grad = get_q4_packed_data_tensor_cuda(output_gradient, scale_grad, false);

    // 4-bit gemm for grad_a
    auto gemm_grad_a = q4_gemm(m, k, n, packed_output_grad, q4_w); // (m, k)
    // 4-bit gemm for grad_w (n,m)*(m,k)=(n,k)
    auto gemm_grad_w = q4_gemm(n, k, m, packed_output_grad.t().contiguous(), q4_a);

    return std::make_pair(gemm_grad_a*scale_a, gemm_grad_w*scale_w);
}

std::pair<torch::Tensor, torch::Tensor> q4_linear_cutlass_backward(
    torch::Tensor output_gradient,
    torch::Tensor q4_a,
    torch::Tensor q4_w,
    float scale_a,
    float scale_w,
    float scale_grad
){
    return q4_backward(output_gradient, q4_a, q4_w, scale_a, scale_w, scale_grad);
}


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
    float scale_y
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    // get m, k, n
    int m = x.size(0);
    int k = x.size(1);
    int n = y.size(0); // shape(n, k)

	torch::Tensor packed_x = get_q4_packed_data_tensor_cuda(x, scale_x, false);
    // (n, k/bit_width)
    torch::Tensor packed_y = get_q4_packed_data_tensor_cuda(y, scale_y, false);

    // 4-bit gemm using cutlass template
    auto gemm = q4_gemm(m, n, k, packed_x, packed_y);

    std::vector<torch::Tensor> outputs;
    outputs.push_back(gemm);
    outputs.push_back(packed_x);
    outputs.push_back(packed_y);
    return outputs;
}


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
    float scale_y
){
	const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
	// x_shape: bs, m, k
	// y_shape: bs, n, k
	// bs denotes the batch size
	auto sizes_x = x.sizes().size();
	auto sizes_y = y.sizes().size();
	// Get the last two dimensions
	int k = x.sizes()[sizes_x - 1]; // last dim * batch size
	int m = x.sizes()[sizes_x - 2]; // second_last_dim is seq_length
	int n = y.sizes()[sizes_y - 2]; // y_shape: bs, n, k
	// reshaping the input tensor to 3D
	x = x.view({-1, m, k});
	y = y.view({-1, n, k});
	// batch dim after reshaping
	int bs = x.size(0);

	// (bs, m, k/bit_width)
	torch::Tensor packed_x = get_q4_packed_data_tensor_cuda(x, scale_x, false);
    // (bs, n, k/bit_width)
    torch::Tensor packed_y = get_q4_packed_data_tensor_cuda(y, scale_y, false);

    auto gemm = batched_q4_gemm(m, n, k, packed_x, packed_y, bs);

    std::vector<torch::Tensor> outputs;
    outputs.push_back(gemm);
    outputs.push_back(packed_x);
    outputs.push_back(packed_y);
    return outputs;
}


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
    torch::Tensor q4_x,
    torch::Tensor q4_y,
    float scale_x,
    float scale_y,
    float scale_grad
){
	const at::cuda::OptionalCUDAGuard device_guard(device_of(output_gradient));

    auto option_quantize = torch::TensorOptions().dtype(torch::kInt8).device(output_gradient.device());
    // x_shape: bs, m, k/2
	// y_shape: bs, n，k/2
	// bs denotes the batch size
	auto sizes_x = q4_x.sizes().size();
	auto sizes_y = q4_y.sizes().size();
	// Get the last two dimensions
	int k = q4_x.sizes()[sizes_x - 1] * 2; // note that here the k after packing in forward is actually k/2
	int m = q4_x.sizes()[sizes_x - 2]; // second_last_dim is seq_length
	int n = q4_y.sizes()[sizes_y - 2]; // y_shape: bs, n, k
	// reshaping the input tensor to 3D
	q4_x = q4_x.view({-1, m, k>>1});
	q4_y = q4_y.view({-1, k>>1, n});
	output_gradient = output_gradient.view({-1, m, n});
	// batch dim after reshaping
	int bs = q4_x.size(0);

    // calculate 4-bit packed gradient (bs,m,n)
    torch::Tensor packed_output_grad = get_q4_packed_data_tensor_cuda(output_gradient, scale_grad, false);

    // 4-bit gemm for grad_a
    // (bs, m, n) * (bs, n, k) = (bs, m, k)
    auto gemm_grad_x =
        batched_q4_gemm(m, k, n, packed_output_grad, q4_y, bs);
    // 4-bit gemm for grad_y (bs,n,m)*(bs,m,k)=(bs,n,k)
    auto gemm_grad_y =
        batched_q4_gemm(n, k, m,
            packed_output_grad.transpose(-1, -2).contiguous(), q4_x, bs);

    return std::make_pair(gemm_grad_x*scale_x, gemm_grad_y*scale_y);
}