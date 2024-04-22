#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "binary_linear_cutlass_kernel.h"
#include "kernel_selection.h"

typedef uint8_t BINARY_WORD; // 8-bit binary word
const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);
const int THREADS_PER_BLOCK = 512;

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
__device__ BINARY_WORD bit_packing(DstScalar* array){
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

template <typename T>
__global__ void output_scaling_converting(T* input, T k, T scale, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
    	// 1. alignment from (0,1) to (-1, +1) : out = k - 2 * out
		// 2. multiply scaling factor: out = out * scale_a
        input[i] = (k - input[i] * 2.0f) * scale;
    }
}

__global__ void output_scaling_converting(__nv_bfloat16* input, __nv_bfloat16 k, __nv_bfloat16 scale, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
    	// 1. alignment from (0,1) to (-1, +1) : out = k - 2 * out
		// 2. multiply scaling factor: out = out * scale_a
        __nv_bfloat16 aligned = __hfma(input[i], __float2bfloat16(-2.0f), k); // a * b + c
        input[i] = __hmul_rn(aligned, scale); // a * b
    }
}

} // namespace cuda

namespace xnor_cutlass {

cudaError_t cutlass_binary_gemm(
	int M,
	int N,
	int K,
	float alpha,
	cutlass::uint1b_t *A,
	int lda,
	cutlass::uint1b_t *B,
	int ldb,
	float beta,
	int32_t *C,
	int ldc,
    int kernel_id) {
	// arguments definition
	int length_m = M;
	int length_n = N;
	int length_k = K;
	// Create a tuple of problem size for matrix multiplication
	cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
	// Initialize tensors using CUTLASS helper functions
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a(A, lda);  // <- Create matrix A with dimensions M x K
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b(B, ldb);  // <- Create matrix B with dimensions K x N
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c(C, ldc);     // <- Create matrix C with dimensions M x N
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c(C, ldc);  // <- Create matrix C with dimensions M x N used to store output from reference kernel
	// Initialize alpha and beta for dot product computation
	ElementComputeEpilogue alpha_e = ElementComputeEpilogue(alpha);
	ElementComputeEpilogue beta_e = ElementComputeEpilogue(beta);
	// Split K dimension into 1 partitions
	int split_k_slices = 1;

	// call selected kernel function
	cudaError_t result;

#ifdef ARCH_SM_80 // sm_80
    switch (kernel_id) {
        case 1:
            result = sm80_gemm_7(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        case 2:
            result = sm80_gemm_8(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        case 3:
            result = sm80_gemm_9(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        case 4:
            result = sm80_gemm_4(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        case 5:
            result = sm80_gemm_5(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        case 6:
            result = sm80_gemm_6(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        case 7:
            result = sm80_gemm_1(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        case 8:
            result = sm80_gemm_2(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        case 9:
            result = sm80_gemm_3(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
        default:
            result = sm80_gemm_9(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
            break;
    }
#endif // end of ifdef ARCH_SM_80

#ifdef ARCH_SM_75 // sm_75
	result = sm75_gemm_1(problem_size, tensor_a, tensor_b, tensor_c, tensor_ref_c, alpha_e, beta_e, split_k_slices);
#endif // end of ifdef ARCH_SM_75

	// Return success, if no errors were encountered.
	return result;
}



/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t cutlass_binary_batched_gemm(
	int M,
	int N,
	int K,
	float alpha,
	cutlass::uint1b_t *A,
	int lda,
	cutlass::uint1b_t *B,
	int ldb,
	float beta,
	int32_t *C,
	int ldc,
	long long int batch_stride_A,
	long long int batch_stride_B,
	long long int batch_stride_C,
	int bs
) {

#ifdef ARCH_SM_80 // sm_80
	//TODO: will use adaptive selection function
	using CutlassGemm = sm80_batched_gemm_1;
#endif // end of ifdef ARCH_SM_80

#ifdef ARCH_SM_75 // sm_75
	using CutlassGemm = sm75_batched_gemm_1;
#endif // end of ifdef ARCH_SM_75

	int length_m = M;
	int length_n = N;
	int length_k = K;

	// Create a tuple of problem size for matrix multiplication
	cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
	// Initialize tensors using CUTLASS helper functions
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a(A, lda);  // <- Create matrix A with dimensions M x K
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b(B, ldb);  // <- Create matrix B with dimensions K x N
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c(C, ldc);     // <- Create matrix C with dimensions M x N
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c(C, ldc);  // <- Create matrix C with dimensions M x N used to store output from reference kernel
	// Initialize alpha and beta for dot product computation
	ElementComputeEpilogue alpha_e = ElementComputeEpilogue(alpha);
	ElementComputeEpilogue beta_e = ElementComputeEpilogue(beta);

	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments
		args{
			problem_size,  // <- problem size of matrix multiplication
			tensor_a,  // <- reference to matrix A on device
			batch_stride_A,
			tensor_b,  // <- reference to matrix B on device
			batch_stride_B,
			tensor_c,// <- reference to matrix C on device
			batch_stride_C,
			tensor_ref_c,  // <- reference to matrix C on device
			batch_stride_C,
			{alpha_e, beta_e},      // <- tuple of alpha and beta
			bs
		};        // <- batch size

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);
	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
	}
	// Return success, if no errors were encountered.
	return cudaSuccess;
}


} // namespace xnor_cutlass


// C++-CUDA methods


torch::Tensor binary_forward_cutlass(
    torch::Tensor packed_a,
    torch::Tensor packed_w,
    int m,
    int n,
    int k,
    int kernel_id){

    auto options =
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(packed_a.device());
    auto output = torch::zeros({m, n}, options);

    /// cutlass GEMM
    float alpha = 1.0f, beta = 0.0f;
    int lda = k;
    int ldb = k;
    int ldc = n;

    cudaError_t result =
        xnor_cutlass::cutlass_binary_gemm(
			m,
			n,
			k,
			alpha,
			reinterpret_cast<cutlass::uint1b_t *>(packed_a.data_ptr<uint8_t>()),
			lda,
			reinterpret_cast<cutlass::uint1b_t *>(packed_w.data_ptr<uint8_t>()),
			ldb,
			beta,
			output.data_ptr<int32_t>(),
			ldc,
			kernel_id);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
    }
    return output;
}


torch::Tensor binary_batched_forward_cutlass(
    torch::Tensor packed_a,
    torch::Tensor packed_w,
    int m,
    int n,
    int k,
    int bs){

    auto options =
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(packed_a.device());
    auto output = torch::zeros({bs, m, n}, options);

    /// cutlass GEMM
    float alpha = 1.0f, beta = 0.0f;
    int const lda = k;
    int const ldb = k * bs;
    int const ldc = n;

	// the memory is batched along K dimension
	long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(m);
	long long int batch_stride_B = static_cast<long long int>(k);
	long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(m);

    cudaError_t result =
        xnor_cutlass::cutlass_binary_batched_gemm(
			m,
			n,
			k,
			alpha,
			reinterpret_cast<cutlass::uint1b_t *>(packed_a.data_ptr<uint8_t>()),
			lda,
			reinterpret_cast<cutlass::uint1b_t *>(packed_w.data_ptr<uint8_t>()),
			ldb,
			beta,
			output.data_ptr<int32_t>(),
			ldc,
			batch_stride_A,
			batch_stride_B,
			batch_stride_C,
			bs);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
    }
    return output;
}


/**
 * Quantizes and packs the input tensor into a uint8_t representation using CUDA.
 *
 * This function is designed to work with tensors that have either 2 or 3 dimensions.
 * It quantizes the input tensor, `input`, by packing multiple binary bits into each byte of the output tensor, `w_quant`.
 * The packing process is dependent on the template type `DstScalar`, which dictates the binary representation.
 *
 * @tparam DstScalar The data type of the elements in the output packed tensor.
 * @param input The input tensor to be quantized and packed. It can be either 2D [MxK] or 3D [N, M, K].
 *              The last two dimensions are considered as the dimensions to be packed.
 *              In case of a 3D tensor, the function treats it as a batch of 2D matrices [N, MxK].
 *
 * @return A tensor of quantized and packed values in uint8 format. For a 2D input [MxK], the output
 *         tensor will have dimensions [M, K/BITS_PER_BINARY_WORD]. For a 3D input [N, M, K], the output
 *         tensor will be [N, M, K/BITS_PER_BINARY_WORD], effectively treating the first dimension as a batch size.
 *
 * Note:
 * - `BITS_PER_BINARY_WORD` is assumed to be a constant that defines how many bits are packed into a single byte.
 * - `THREADS_PER_BLOCK` is a predefined constant that determines the number of CUDA threads per block for the kernel execution.
 * - This function checks the dimensionality of the input tensor and adjusts the computation accordingly.
 * - The function exits with an error message if the input tensor has unsupported dimensions (not 2D or 3D).
 * - The CUDA kernel `xnor_cuda::_to_uint8_array` is called to perform the actual quantization and packing operation.
 * - The output tensor, `w_quant`, is created with the appropriate size and data type (`torch::kUInt8`) and is filled by the kernel.
 * - For 3D inputs, the output tensor is reshaped to preserve the batch dimension.
 */
template <typename DstScalar>
torch::Tensor get_pack_cuda_kernel(
    torch::Tensor input){
    auto option_quantize = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
	auto input_size = input.sizes().size();
    auto m = input.sizes()[input_size-2];
    auto k = input.sizes()[input_size-1];
	torch::Tensor w_quant;

    if (input_size == 3){
        m = m * input.size(0);
	} else if (input_size != 2){
        std::cerr << "tensor sizes not supported: " << input_size << std::endl;
        exit(EXIT_FAILURE);
	}
	w_quant = torch::empty({m, k/BITS_PER_BINARY_WORD}, option_quantize);

	dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
	dim3 grid_dim((m * k) / (THREADS_PER_BLOCK * BITS_PER_BINARY_WORD) + 1, 1);
	xnor_cuda::_to_uint8_array<DstScalar><<<grid_dim, block_dim>>>(
	                        reinterpret_cast<DstScalar *>(input.data_ptr()),
	                        w_quant.data_ptr<uint8_t>(),
	                        m*k/BITS_PER_BINARY_WORD);

	if(input_size == 3)
		w_quant = w_quant.view({input.size(0), -1, k/BITS_PER_BINARY_WORD});
    return w_quant;
}


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
    bool transpose){
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
    if(transpose) {
        pack_qw = pack_qw.transpose(-1, -2).contiguous();
	}
    return pack_qw;
}


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
    bool transpose,
    int kernel_id
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // get m, k, n
    int m = input.size(0);
    int k = input.size(1);
    int n;
    torch::Tensor packed_w;

    if(weight.dtype() == torch::kUInt8){
        n = weight.size(0); // shape(n, k)
        packed_w = weight;
    }else{
        n = weight.size(0); // shape(n, k)
		packed_w = get_packed_data_tensor(weight, transpose); // weight needs to be transposed
    }

	if (input.dtype() != torch::kUInt8){
        input = get_packed_data_tensor(input, false); // weight needs to be transposed
    } else {
        k *= BITS_PER_BINARY_WORD;
    }

    torch::Tensor output = binary_forward_cutlass(input, packed_w, m, n, k, kernel_id);
    return output;
}


/**
 * Applies output scaling and range conversion to a tensor using a CUDA kernel.
 *
 * This function performs two main operations on the given output tensor:
 * 1. Range alignment: Adjusts the tensor's values from the (0,1) range to (-1, +1) range.
 *    This is achieved by the formula: out = k - 2 * out.
 * 2. Scaling: Multiplies each element in the tensor by a scaling factor.
 *    The operation is: out = out * scale.
 *
 * The above transformations are applied in-place on the 'output' tensor.
 *
 * @param output The output tensor to be transformed. The modifications are done in-place.
 * @param k The offset value used for the range alignment transformation.
 * @param scale The scaling factor to be applied to each element of the tensor.
 * @return The transformed output tensor with applied range alignment and scaling.
 *
 * Note: This function launches a CUDA kernel to perform the operations efficiently on the GPU.
 * The kernel is configured with a fixed block size of 256 and a grid size calculated based on
 * the total number of elements in the 'output' tensor.
 */
template <typename T>
torch::Tensor get_output_scaling_kernel(
	torch::Tensor output,
	T k,
	T scale){
	// use the following cuda kernel to do two tasks:
	// 1. alignment from (0,1) to (-1, +1) : out = k - 2 * out
	// 2. multiply scaling factor: out = out * scale_a
    int size = output.numel();
    dim3 block(256);
    dim3 grid((size - 1) / 256 + 1);

    xnor_cuda::output_scaling_converting<<<grid, block>>>(
        reinterpret_cast<T *>(output.data_ptr()),
        k,
        scale,
        size);
	return output;
}


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
    int kernel_id
){
	float k = (float)input.size(-1);
	auto output = binary_linear_cutlass_forward(input, weight, transpose, kernel_id).to(input.dtype());

    if(output.dtype() == torch::kFloat32){
        output = get_output_scaling_kernel<float>(output, k, scale);
    }else if(output.dtype() == torch::kBFloat16){
        output = get_output_scaling_kernel<__nv_bfloat16>(output, __float2bfloat16(k), __float2bfloat16(scale));
    }else{
        std::cerr << "tensor type not supported: " << input.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }
	return output;
}


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
    int kernel_id
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    // get m, k, n
    int m = x.size(0); // x shape(m, k)
    int k = x.size(1);
    int n = y.size(0); // shape(n, k)

    auto packed_x = get_packed_data_tensor(x, false); // (m, k/bit_width)
    // y.shape:(n, k)
	auto packed_y = get_packed_data_tensor(y, false); // (n, k/bit_width)
    torch::Tensor output = binary_forward_cutlass(packed_x, packed_y, m, n, k, kernel_id);
    return output;
}


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
    float scale
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
	// x_shape: bs, m, k;
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

    auto packed_x = get_packed_data_tensor(x, false); // (bs, m, k/bit_width)
	auto packed_y = get_packed_data_tensor(y, false); // (bs, n, k/bit_width) -> (bs, k/bit_width, n)

	// output.shape: (bs, m, n)
    torch::Tensor output = binary_batched_forward_cutlass(packed_x, packed_y, m, n, k, bs).to(x.dtype());

    if(output.dtype() == torch::kFloat32){
        output = get_output_scaling_kernel<float>(output, k, scale);
    }else if(output.dtype() == torch::kBFloat16){
        output = get_output_scaling_kernel<__nv_bfloat16>(output, __float2bfloat16(k), __float2bfloat16(scale));
    }else{
        std::cerr << "tensor type not supported: " << x.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }

    return output;
}


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
int get_selected_kernel_id(int device_id, int m, int n, int k){
 	int kernel_id = find_best_kernel(device_id, m, n, k);
 	if (kernel_id < 1 || kernel_id > 9){
        std::cerr << "Got invalid kernel_id for binary linear cutlass: " << kernel_id << std::endl;
        exit(EXIT_FAILURE);
 	}
 	return kernel_id;
}