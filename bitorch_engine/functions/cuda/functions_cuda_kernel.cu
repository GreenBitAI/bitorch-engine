#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

typedef uint8_t BINARY_WORD; // 8-bit binary word
const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);
const int kMaxThreadsPerBlock (256);

namespace cuda{

//
// CUDA methods
//

__global__ void getmaxmin_kernel(float *a, float *c, int n)
{
    int tid = threadIdx.x;
    __shared__ float maxa[1024];
    __shared__ float mina[1024];

    maxa[tid] = a[tid];
    mina[tid] = a[tid];
    __syncthreads();
    for (int i=tid; i<n; i+=1024)
    {
        if (maxa[tid] < a[i]) maxa[tid] = a[i];
        if (mina[tid] > a[i]) mina[tid] = a[i];
    }
    __syncthreads();
    for (int k=512; k>0; k>>=1)
    {
        if (tid < k)
        {
            if (maxa[tid] < maxa[tid+k]) maxa[tid] = maxa[tid+k];
            if (mina[tid] > mina[tid+k]) mina[tid] = mina[tid+k];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        c[0] = maxa[0];
        c[1] = mina[0];
    }
}

__global__ void fp32toint4_kernel(float* a, unsigned* b, float* c, int m)
{
    float maxv = c[0];
    float minv = c[1];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < m)
    {
        unsigned outv = 0;
#pragma unroll
        for (int i=0; i<7; i++)
        {
            outv = outv | ((static_cast<int>((a[id*4+i] - minv)*15.0/(maxv-minv)))<<(28-i*4));
        }
        b[id] = outv;
    }
}


//============ general bit-packing =============//
template <typename T>
__device__ BINARY_WORD bit_packing(T* array){
    BINARY_WORD rvalue=0;
    BINARY_WORD sign;
#pragma unroll
    for (int i = 0; i < BITS_PER_BINARY_WORD; i++){
        sign = (array[i] >= 0);
        rvalue |= (sign << i);
    }
    return rvalue;
}

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

template <typename T>
__global__ void _to_uint8_array(T *a, uint8_t *b, int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x)
    {
        BINARY_WORD bw = bit_packing(&a[i*BITS_PER_BINARY_WORD]); //BITS_PER_BINARY_WORD=8 for uint8_t
        b[i] = bw;
    }
}
//===========================================//


__global__ void unpack_uint8_to_float(const uint8_t* in, const float * scl, const size_t n, const size_t expand_dim, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint8_t binary_word = in[i];
        // divide by expand_dim to adjust for the unexpanded final axis:
        float sc = scl[i / expand_dim];
        #pragma unroll
	    for(int bit_p=0; bit_p<8; bit_p++){
	        int sign = ((binary_word >> bit_p) & 0x1); // read bit value
	        if (sign == 0) sign = -1;
	        out[i*8+bit_p] = (float)sign*sc;
	    }
    }
}


__global__ void q4_bit_packing_kernel(int *input, int8_t *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx * 2; // Each pair of inputs shares one output index

    if (pair_idx < N) {
        unsigned char packedValue = 0;

        // Handle the first of the pair
        if (pair_idx < N) {
            int qValueFirst = input[pair_idx] & 0xF; // Keep only the lower 4 bits
            packedValue |= qValueFirst << 4; // Place it in the high 4 bits
        }

        // Handle the second of the pair
        if (pair_idx + 1 < N) {
            int qValueSecond = input[pair_idx + 1] & 0xF; // Keep only the lower 4 bits
            packedValue |= qValueSecond; // Place it in the low 4 bits
        }

        // Write the packed value directly to output
        output[idx] = packedValue;
    }
}


__global__ void q4_bit_unpacking_kernel(int8_t *input, int *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int unpacked_idx = idx * 2; // Each input index maps to two output indices

    if (idx < N) {
        unsigned char packedValue = static_cast<unsigned char>(input[idx]);

        // Extract high and low 4-bit values
        int highValue = (packedValue >> 4) & 0xF;
        int lowValue = packedValue & 0xF;

        // Store the unpacked values
        if (unpacked_idx < 2*N) { // Check bounds for safety
            output[unpacked_idx] = highValue;
            if (unpacked_idx + 1 < 2*N) {
                output[unpacked_idx + 1] = lowValue;
            }
        }
    }
}


__global__ void q4_bit_unpacking_scaling_kernel(int8_t *input, float scale, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int unpacked_idx = idx * 2; // Each input index maps to two output indices

    if (idx < N) {
        unsigned char packedValue = static_cast<unsigned char>(input[idx]);

        // Extract high and low 4-bit values
        int highValue = (packedValue >> 4) & 0xF;
        int lowValue = packedValue & 0xF;

        // Convert 4-bit unsigned to 4-bit signed (-8 to 7 range)
        if (highValue > 7) highValue -= 16;
        if (lowValue > 7) lowValue -= 16;

        // Store the unpacked and scaled values
        if (unpacked_idx < 2*N) { // Check bounds for safety
            output[unpacked_idx] = highValue * scale;
            if (unpacked_idx + 1 < 2*N) {
                output[unpacked_idx + 1] = lowValue * scale;
            }
        }
    }
}


} // end of cuda methods


//
// C++-CUDA methods
//

/*
 * Converts a 32-bit floating point tensor to a 4-bit integer representation.
 *
 * This function takes an input tensor of floating point numbers and compresses it
 * into a tensor of 4-bit integers, effectively reducing the memory footprint by a factor of 8.
 * The conversion process involves finding the minimum and maximum values of the input
 * to normalize the data range, and then quantizing the normalized values into 4-bit integers.
 *
 * Parameters:
 *  - input: A tensor of 32-bit floating point numbers that we want to compress.
 *
 * Returns:
 *  - A tensor of 4-bit integers representing the quantized version of the input tensor.
 *    The output tensor uses a 64-bit integer data type to store the 4-bit values,
 *    with each 64-bit integer holding sixteen 4-bit values.
 *
 * Note:
 *  - The input tensor is assumed to be a flat 1D tensor, and the output tensor will also be a 1D tensor.
 *  - This function is designed to be executed on CUDA-enabled devices and utilizes custom CUDA kernels
 *    for the quantization process.
 *  - The function allocates temporary memory on the GPU for intermediate computations, which is
 *    freed before returning the output tensor.
 */
torch::Tensor fp32_to_int4_cuda(
    torch::Tensor input
) {
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    int N = input.numel();
    int M = N/8;
    auto options = torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(input.device())
                        .requires_grad(false);
    auto output = torch::empty({M}, options);

	float *fA = input.data_ptr<float>();
    unsigned *fB = (unsigned *)output.data_ptr();
    float *fC = NULL;
    cudaMalloc(&fC, 2 * sizeof(float));

    // launch kernel
    cuda::getmaxmin_kernel<<<1,kMaxThreadsPerBlock>>>(fA, fC, N);
    cuda::fp32toint4_kernel<<<dim3((M+1023)/kMaxThreadsPerBlock), kMaxThreadsPerBlock>>>(fA, fB, fC, M);

    cudaFree(fC);
    return output;
}


/**
 * Template function for packing a CUDA kernel tensor into a quantized format.
 * This function takes a tensor as input and returns a new tensor where the input
 * tensor's elements are packed and quantized into 8-bit unsigned integers.
 * This is particularly useful for optimizing memory usage and computational efficiency
 * in neural network operations that can benefit from quantization.
 *
 * Template Parameters:
 * - T: The data type of the elements in the input tensor.
 *
 * Parameters:
 * - input: A torch::Tensor representing the input data to be packed and quantized.
 *
 * Returns:
 * - A torch::Tensor containing the packed and quantized representation of the input tensor.
 */
template <typename T>
torch::Tensor get_pack_cuda_kernel(
    torch::Tensor input){
    auto option_quantize = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    auto m = input.size(0); // out_channels
    auto k = input.size(1); // in_channels
    torch::Tensor w_quant = torch::empty({m, k/BITS_PER_BINARY_WORD}, option_quantize);

    int threads_per_block = 256;
	dim3 block_row(threads_per_block, 1, 1);
	dim3 grid_row(m*k/threads_per_block+1, 1);

	cuda::_to_uint8_array<T><<<grid_row, block_row>>>(
	                        reinterpret_cast<T *>(input.data_ptr()),
	                        w_quant.data_ptr<uint8_t>(),
	                        m*k/BITS_PER_BINARY_WORD);

    return w_quant;
}


/**
 * Packs the given tensor into an 8-bit unsigned integer tensor representation on CUDA.
 *
 * This function converts a tensor of various data types (int8, float32, bfloat16, half)
 * to an 8-bit unsigned integer tensor using CUDA kernels for efficient computation.
 * It ensures that the operation is performed on the correct CUDA device by employing
 * a device guard based on the input tensor's device. If the data type of the input tensor
 * is not supported, the function will terminate the program with an error message.
 *
 * Parameters:
 *  - data: The input tensor to be packed. Can be of type int8, float32, bfloat16, or half.
 *
 * Returns:
 *  - torch::Tensor: An 8-bit unsigned integer representation of the input tensor.
 *
 * Note:
 *  - The input tensor must be on a CUDA device.
 *  - The function terminates the program if the input tensor's type is not supported.
 */
torch::Tensor tensor_pack_to_uint8_cuda(
    torch::Tensor data
){
	const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
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
 * Converts an 8-bit unsigned integer tensor into an unpacked float tensor on CUDA.
 *
 * This function unpacks an 8-bit integer tensor (`emd`) into a float tensor (`out`)
 * while scaling it with another tensor (`scl`). The unpacking process converts
 * each 8-bit integer into 8 separate float values, effectively increasing the
 * dimensionality of the embedding space by a factor of 8. This is particularly
 * useful for operations that require higher precision representations of compressed
 * embeddings.
 *
 * Parameters:
 *  emd A torch::Tensor containing 8-bit unsigned integers with shape
 *            (batch_size, sequence_length, packed_embedding_dimension), representing
 *            the compressed embeddings.
 *  scl A torch::Tensor with shape (batch_size, sequence_length, 1), used for
 *            scaling the unpacked embeddings.
 *
 * return
    A torch::Tensor of floats with shape
 *         (batch_size, sequence_length, packed_embedding_dimension * 8), representing
 *         the unpacked and scaled embeddings.
 *
 * This function calculates the size of the
 * output tensor based on the input dimensions, and initializes the output tensor with
 * zeros. It then launches a CUDA kernel (`unpack_uint8_to_float`) to perform the unpacking
 * and scaling in parallel on the GPU.
 */
torch::Tensor uint8_to_unpacked_tensor_cuda(
    torch::Tensor emd,
    torch::Tensor scl
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(emd));
    int bs = emd.size(0);
    int seq_length = emd.size(1);
    int packed_embed_dim = emd.size(2);
    int unpacked_embed_dim = packed_embed_dim*8;
    // emd has shape (batch_size, seq_length, packed_embedding_dim)
    // scl has shape (batch_size, seq_length, 1)
    auto options = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(scl.device());
    // output (batch_size, seq_length, packed_embedding_dim * 8)
    torch::Tensor out = torch::zeros({bs, seq_length, unpacked_embed_dim}, options);

	// we do not expand scl, we consider it in the kernel
    float * out_ptr = out.data_ptr<float>();
    uint8_t * emd_ptr = emd.data_ptr<uint8_t>();
    float * scl_ptr = scl.data_ptr<float>();
    size_t n = out.numel();
    int threads_per_block = 256;
	dim3 block_row(threads_per_block, 1, 1);
	dim3 grid_row(n/threads_per_block+1, 1);

	cuda::unpack_uint8_to_float<<<grid_row, block_row>>>
	(
        emd_ptr,
        scl_ptr,
        n/8,
        packed_embed_dim,
        out_ptr
    );

    return out;
}


/**
 * Function to perform 4-bit packing on a tensor.
 * This function quantizes and packs the input data tensor into 4-bit integers,
 * storing every two quantized values into one int8 variable.
 *
 * Parameters:
 *     data: The input tensor to be quantized and packed. The tensor can have 2 or 3 dimensions,
 *           where the last two dimensions are considered as out_channels and in_channels, respectively.
 *     is_transpose: If true, the packed tensor will be transposed on the last two dimensions.
 *
 * Returns:
 *     A packed tensor with int8 data type, where every two 4-bit quantized values are stored in one int8 variable.
 */
torch::Tensor q4_pack_cuda(
    const torch::Tensor data,
    bool is_transpose
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
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

    dim3 block(kMaxThreadsPerBlock);
    dim3 grid((N-1)/(block.x)+1);

    cuda::q4_bit_packing_kernel<<<grid, block>>>(
        data.data_ptr<int>(),
        pack_qw.data_ptr<int8_t>(),
        N);

	if (input_size == 3) pack_qw = pack_qw.view({data.size(0), -1, k>>1});
    if (is_transpose) pack_qw = pack_qw.transpose(-1, -2).contiguous();

    return pack_qw;
}


/**
 * Function to unpack a 4-bit packed tensor into its original int values.
 * Parameters:
 * packed_data: The tensor containing packed 4-bit values in int8 format.
 * is_transpose: Boolean flag indicating whether the unpacked tensor should be transposed.
 * Returns:
 * A tensor of int32 values, unpacked from the input tensor.
 */
torch::Tensor q4_unpack_cuda(
    const torch::Tensor packed_data,
    bool is_transpose
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(packed_data));
    auto option_unquantize = torch::TensorOptions().dtype(torch::kInt32).device(packed_data.device());

    auto input_size = packed_data.sizes().size();
    // Calculate the size for the unpacked tensor
    auto n = packed_data.sizes()[input_size-2]; // out_channels
    auto k = packed_data.sizes()[input_size-1] * 2; // in_channels

    torch::Tensor unpacked_qw = torch::empty({n, k}, option_unquantize);

    int packed_size = packed_data.numel();

    dim3 block(kMaxThreadsPerBlock);
    dim3 grid((packed_size-1)/(block.x)+1);

    cuda::q4_bit_unpacking_kernel<<<grid, block>>>(
        packed_data.data_ptr<int8_t>(),
        unpacked_qw.data_ptr<int>(),
        packed_size);

    if (input_size == 3) unpacked_qw = unpacked_qw.view({packed_data.size(0), -1, k});
    if (is_transpose) unpacked_qw = unpacked_qw.transpose(-1, -2).contiguous();

    return unpacked_qw;
}


/**
 *  Unpacks and scales a 4-bit packed tensor to a float32 tensor. This function is designed to work with CUDA tensors.
 * Parameters:
 *      packed_data: The tensor containing packed 4-bit values in int8 format.
 *      scale: a scaling factor will be applied to each unpacked value.
 *      is_transpose: Boolean flag indicating whether the unpacked tensor should be transposed.
 * Returns:
 *       A tensor of int32 values, unpacked from the input tensor.
 */
torch::Tensor q4_unpack_and_scaling_cuda(
    const torch::Tensor packed_data,
    float scale,
    bool is_transpose
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(packed_data));
    auto option_unquantize = torch::TensorOptions().dtype(torch::kFloat32).device(packed_data.device());

    auto input_size = packed_data.sizes().size();
    int k;
    torch::Tensor unpacked_qw;

    if (input_size == 4){
        // shape: {NHWC}
        unpacked_qw =
	        torch::empty(
	            {packed_data.sizes()[0], packed_data.sizes()[1], packed_data.sizes()[2], packed_data.sizes()[3] * 2},
	            option_unquantize
	        );
    }else{
	    // Calculate the size for the unpacked tensor
	    auto n = packed_data.sizes()[input_size-2]; // out_channels
	    auto k = packed_data.sizes()[input_size-1] * 2; // in_channels

	    unpacked_qw = torch::empty({n, k}, option_unquantize);
	}

    int packed_size = packed_data.numel();

    dim3 block(kMaxThreadsPerBlock);
    dim3 grid((packed_size-1)/(block.x)+1);

    cuda::q4_bit_unpacking_scaling_kernel<<<grid, block>>>(
        packed_data.data_ptr<int8_t>(),
        scale,
        unpacked_qw.data_ptr<float>(),
        packed_size);

    if (input_size == 3) unpacked_qw = unpacked_qw.view({packed_data.size(0), -1, k});
    if (is_transpose) unpacked_qw = unpacked_qw.transpose(-1, -2).contiguous();

    return unpacked_qw;
}