#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string.h>
#include <iostream>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/tensor_view.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>

#include <cuda_fp16.h>
#include <vector_types.h>
#include <cmath>
#include <curand_kernel.h>
#include <tuple>
#include <bits/stdc++.h>

// standard number of threads per block to use
#define NUM_THREADS 512

#define CUTLASS_CHECK(status)                                                                    \
{                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
}

//=================== Cutlass configuration ==================//
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = int8_t;                       // <- data type of elements in input matrix A
using ElementInputB = int8_t;                       // <- data type of elements in input matrix B
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
// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

#ifdef ARCH_SM_75 // sm_75
	const int NumStages = 2;
	using SmArch = cutlass::arch::Sm75;
	using ShapeMMAThreadBlock =
	    cutlass::gemm::GemmShape<256, 128, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
	using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64
	using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 16
#endif // end of ifdef ARCH_SM_75

#ifdef ARCH_SM_80 // sm_80
	const int NumStages = 3;
	using SmArch = cutlass::arch::Sm80;
	using ShapeMMAThreadBlock =
	    cutlass::gemm::GemmShape<128, 128, 128>;  // <- threadblock tile M = 256, N = 128, K = 128
	using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 128
	using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M = 16, N = 8, K = 64
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
// ===============================================================//


/**
 * Executes a single-precision general matrix multiplication (SGEMM) using the CUTLASS library.
 * This function performs the operation C = alpha * A * B + beta * C, where A, B, and C are matrices,
 * and alpha and beta are scalar values. The matrices A and B are assumed to be in row-major order.
 *
 * @param M The number of rows in matrices A and C.
 * @param N The number of columns in matrices B and C.
 * @param K The number of columns in matrix A and rows in matrix B.
 * @param A Pointer to the first matrix operand (matrix A) in device memory.
 * @param lda Leading dimension of matrix A, specifying the stride between consecutive rows.
 * @param B Pointer to the second matrix operand (matrix B) in device memory.
 * @param ldb Leading dimension of matrix B, specifying the stride between consecutive rows.
 * @param C Pointer to the output matrix (matrix C) in device memory. This matrix is also used as the third operand in the computation for beta != 0.
 * @param ldc Leading dimension of matrix C, specifying the stride between consecutive rows.
 *
 * @return cudaError_t Status of the SGEMM operation. Returns cudaSuccess on successful execution.
 *
 * The function initializes the CUTLASS GEMM operation with the specified problem size and leading dimensions
 * for each matrix. It then configures the alpha and beta coefficients for the GEMM computation.
 * The GEMM operation is executed with a single partition along the K dimension.
 * The function checks for errors during the GEMM operation and returns an appropriate error code if encountered.
 */
cudaError_t CutlassSgemmNN(
    const int M,
    const int N,
    const int K,
    const int8_t *A,
    int lda,
    const int8_t *B,
    int ldb,
    int32_t *C,
    int ldc
) {
    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
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

// C++-CUDA methods


/**
 * Performs quantized 8-bit General Matrix Multiply (GEMM) using CUTLASS.
 *
 * This function computes the matrix product of two quantized 8-bit tensors
 * using the CUTLASS library. It is designed for performing efficient GEMM operations
 * on CUDA-enabled devices.
 *
 * @param m The number of rows in the matrix A and the resulting matrix.
 * @param n The number of columns in the matrix B and the resulting matrix.
 * @param k The common dimension of matrices A and B.
 * @param q8_a A quantized 8-bit integer tensor representing matrix A.
 * @param q8_w A quantized 8-bit integer tensor representing matrix B.
 * @return A 32-bit integer tensor containing the result of the matrix multiplication.
 */
torch::Tensor q8_gemm(
    int m,
    int n,
    int k,
    const torch::Tensor q8_a,
    const torch::Tensor q8_w
){
    // cutlass GEMM
    int lda = k;
    int ldb = k;
    int ldc = n;

    auto option_gemm = torch::TensorOptions().dtype(torch::kInt32).device(q8_a.device());
    torch::Tensor gemm = torch::empty({m, n}, option_gemm);
    cudaError_t result;

    result =
        CutlassSgemmNN(
            m,
            n,
            k,
            q8_a.data_ptr<int8_t>(),
            lda,
            q8_w.data_ptr<int8_t>(),
            ldb,
            gemm.data_ptr<int32_t>(),
            ldc);

    return gemm;
}


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
    float scale_w
) {
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    // get m, k, n
    int m = input.size(0);
    int k = input.size(1);
    int n = weight.size(0);

    if(transpose) weight = weight.t().contiguous();
    // 8-bit gemm using cutlass template
    auto result = q8_gemm(m, n, k, input, weight);
    return result*scale_a*scale_w;
}


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
    torch::Tensor q8_a,
    torch::Tensor q8_w
){

    // convert grad to int8
    if(output_gradient.dtype() != torch::kInt8)
        output_gradient = output_gradient.to(torch::kInt8);
    if(q8_a.dtype() != torch::kInt8 || q8_w.dtype() != torch::kInt8){
        std::cerr << "Error: the dtype of activation or weight tensor must be int8!" << std::endl;
        exit(1);
    }

    // get m, k, n
    int m = q8_a.size(0);
    int k = q8_a.size(1);
    int n = q8_w.size(0);

    // 8-bit gemm for grad_a
    auto gemm_grad_a = q8_gemm(m, k, n, output_gradient, q8_w); // (m, k)
    // 8-bit gemm for grad_w (n,m)*(m,k)=(n,k)
    auto gemm_grad_w = q8_gemm(n, k, m, output_gradient.t().contiguous(), q8_a);

    return std::make_pair(gemm_grad_a, gemm_grad_w);
}