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
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <curand_kernel.h>
#include <tuple>
#include <bits/stdc++.h>

// standard number of threads per block to use
const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT1 = 8;
const int BLOCKHEIGHT2 = 16;
const int BLOCKHEIGHT4 = 32;
const int BLOCKHEIGHT8 = 64;
// for backward pass
const int BLOCKWIDTH_BACK  = 32;

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

// ================================================================ //
//                            forward                               //
// ================================================================ //

/// bf16
__global__ void quant_mm_kernel_asym(
    const __nv_bfloat16* __restrict__ vec,
    const int* __restrict__ mat,
    __nv_bfloat16* __restrict__ mul,
    const __nv_bfloat16* __restrict__ scales,
    const int* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int w_bit,
	int BLOCKHEIGHT
) {
    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    __shared__ __nv_bfloat16 blockvec[BLOCKWIDTH];
    int nb = (32 / w_bit);
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    __nv_bfloat16 w_tmp;

    int z_w = w / nb;
    int z_mod = (w % nb) * w_bit;

    __nv_bfloat16 weight[BLOCKWIDTH];

    for (k = 0; k < BLOCKWIDTH; ++k) {
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        g = as_int(g_idx[g_h + k]);
        __nv_bfloat16 scale = scales[g * width + w];
		__nv_bfloat16 zero;
		if(w_bit == 8){
	        zero = __float2bfloat16(float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1));
	        w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF));
		}else if(w_bit == 4){
	        zero = __float2bfloat16(float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1));
	        w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF));
		}else if(w_bit == 2){
	        zero = __float2bfloat16(float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x3) + 1));
	        w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3));
		}else{ // w_bit == 1
	        zero = __float2bfloat16(float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x1) + 1));
	        w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1));
		}
		weight[k] = __hmul(scale, __hsub(w_tmp, zero));
    }

    __nv_bfloat16 res;
    for (int b = 0; b < batch; ++b) {
        res = __float2bfloat16(0.0f);

        blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
        __syncthreads();
        for (k = 0; k < BLOCKWIDTH; ++k) {
            res = __hfma(weight[k], blockvec[k], res); // __hfma(a, b, c): a * b + c
        }
        atomicAdd(&mul[b * width + w], res);
        __syncthreads();
    }
}

/// half
__global__ void quant_mm_kernel_asym(
    const __half * __restrict__ vec,
    const int* __restrict__ mat,
	__half * __restrict__ mul,
    const __half * __restrict__ scales,
    const int* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int w_bit,
	int BLOCKHEIGHT
) {
    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    __shared__ __half  blockvec[BLOCKWIDTH];
    int nb = (32 / w_bit);
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    __half  w_tmp;

    int z_w = w / nb;
    int z_mod = (w % nb) * w_bit;

    __half  weight[BLOCKWIDTH];

    for (k = 0; k < BLOCKWIDTH; ++k) {
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        g = as_int(g_idx[g_h + k]);
        __half  scale = scales[g * width + w];
		__half  zero;
		if(w_bit == 8){
	        zero = __float2half(float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1));
	        w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)])>> k_bit)& 0xFF));
		}else if(w_bit == 4){
	        zero = __float2half(float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1));
            w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF));
		}else if(w_bit == 2){
	        zero = __float2half(float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x3) + 1));
            w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3));
		}else{ // w_bit == 1
	        zero = __float2half(float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x1) + 1));
            w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1));
		}
        weight[k] = __hmul(scale, __hsub(w_tmp, zero)); // Use __hmul and __hsub for multiplication and subtraction
    }

    __half  res;
    for (int b = 0; b < batch; ++b) {
        res = __float2half(0.0f);

        blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
        __syncthreads();
        for (k = 0; k < BLOCKWIDTH; ++k) {
            res = __hfma(weight[k], blockvec[k], res); // __hfma(a, b, c): a * b + c
        }
        atomicAdd(&mul[b * width + w], res);
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void quant_mm_kernel_asym(
    const scalar_t* __restrict__ vec,
    const int* __restrict__ mat,
	scalar_t* __restrict__ mul,
    const scalar_t* __restrict__ scales,
    const int* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width,
	int w_bit,
	int BLOCKHEIGHT
) {
	int h = BLOCKHEIGHT * blockIdx.x;
	int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

	__shared__ scalar_t blockvec[BLOCKWIDTH];
	int nb = (32 / w_bit);
	int i = width * h + w;
	int g_h = h * nb;
	int k;
	unsigned int g;
	scalar_t w_tmp;

	int z_w = w / nb;
	int z_mod = (w % nb) * w_bit;

	scalar_t weight[BLOCKWIDTH];

	for (k = 0; k < BLOCKWIDTH; ++k){
		int k_w = (k / nb);
		int k_bit = (k % nb) * w_bit;

		g = as_int(g_idx[g_h + k]);
		scalar_t scale = scales[g * width + w];
		scalar_t zero;
		if(w_bit == 8){
			zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1);
			w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF);
		}else if(w_bit == 4){
			zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1);
			w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);
		}else if(w_bit == 2){
			zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x3) + 1);
			w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3);
		}else{ // w_bit=1
			zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x1) + 1);
			w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1);
		}
		weight[k] = scale * (w_tmp - zero);
	}

	scalar_t res;
	for (int b = 0; b < batch; ++b){
		res = 0;

		blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
		__syncthreads();
		for (k = 0; k < BLOCKWIDTH; ++k){
			res += weight[k] * blockvec[k];
		}
		atomicAdd(&mul[b * width + w], res);
		__syncthreads();
	}
}

template <typename scalar_t>
__global__ void quant_mm_kernel(
    const scalar_t* __restrict__ vec,
    const int* __restrict__ mat,
	scalar_t* __restrict__ mul,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int w_bit,
	int BLOCKHEIGHT
) {
	int h = BLOCKHEIGHT * blockIdx.x;
	int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

	__shared__ scalar_t blockvec[BLOCKWIDTH];
	int nb = (32 / w_bit);
	int i = width * h + w;
	int g_h = h * nb;
	int k;
	unsigned int g;
	scalar_t w_tmp;
	scalar_t weight[BLOCKWIDTH];

	for (k = 0; k < BLOCKWIDTH; ++k){
		int k_w = (k / nb);
		int k_bit = (k % nb) * w_bit;

		g = as_int(g_idx[g_h + k]);
		scalar_t scale = scales[g * width + w];
		scalar_t zero = zeros[g * width + w];
		if(w_bit == 8){
			w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF);
		}else if(w_bit == 4){
			w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);
		}else if(w_bit == 2){
			w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3);
		}else{ // w_bit=1
			w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1);
		}
		weight[k] = scale * w_tmp - zero;
	}

	scalar_t res;
	for (int b = 0; b < batch; ++b){
		res = 0;

		blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
		__syncthreads();
		for (k = 0; k < BLOCKWIDTH; ++k){
			res += weight[k] * blockvec[k];
		}
		atomicAdd(&mul[b * width + w], res);
		__syncthreads();
	}
}

/// bf16
__global__ void quant_mm_kernel(
    const __nv_bfloat16* __restrict__ vec,
    const int* __restrict__ mat,
	__nv_bfloat16* __restrict__ mul,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int w_bit,
	int BLOCKHEIGHT
) {
    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    __shared__ __nv_bfloat16 blockvec[BLOCKWIDTH];
    int nb = (32 / w_bit);
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    __nv_bfloat16 w_tmp;
    __nv_bfloat16 weight[BLOCKWIDTH];

    for (k = 0; k < BLOCKWIDTH; ++k) {
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        g = as_int(g_idx[g_h + k]);
        __nv_bfloat16 scale = scales[g * width + w];
		__nv_bfloat16 zero = zeros[g * width + w];
		if(w_bit == 8){
	        w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF));
		}else if(w_bit == 4){
	        w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF));
		}else if(w_bit == 2){
	        w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3));
		}else{ // w_bit == 1
	        w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1));
		}
		weight[k] = __hfma(scale, w_tmp, __hneg(zero)); // __hfma(a, b, c): a * b + c
    }

    __nv_bfloat16 res;
    for (int b = 0; b < batch; ++b) {
        res = __float2bfloat16(0.0f);

        blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
        __syncthreads();
        for (k = 0; k < BLOCKWIDTH; ++k) {
            res = __hfma(weight[k], blockvec[k], res); // __hfma(a, b, c): a * b + c
        }
        atomicAdd(&mul[b * width + w], res);
        __syncthreads();
    }
}

/// half
__global__ void quant_mm_kernel(
    const __half * __restrict__ vec,
    const int* __restrict__ mat,
	__half * __restrict__ mul,
    const __half * __restrict__ scales,
    const __half * __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int w_bit,
	int BLOCKHEIGHT
) {
    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    __shared__ __half  blockvec[BLOCKWIDTH];
    int nb = (32 / w_bit);
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    __half  w_tmp;
    __half  weight[BLOCKWIDTH];

    for (k = 0; k < BLOCKWIDTH; ++k) {
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        g = as_int(g_idx[g_h + k]);
        __half  scale = scales[g * width + w];
		__half  zero = zeros[g * width + w];
		if(w_bit == 8){
	        w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)])>> k_bit)& 0xFF));
		}else if(w_bit == 4){
            w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF));
		}else if(w_bit == 2){
            w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3));
		}else{ // w_bit == 1
            w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1));
		}
        weight[k] = __hfma(scale, w_tmp, __hneg(zero)); // __hfma(a, b, c): a * b + c
    }

    __half  res;
    for (int b = 0; b < batch; ++b) {
        res = __float2half(0.0f);

        blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
        __syncthreads();
        for (k = 0; k < BLOCKWIDTH; ++k) {
            res = __hfma(weight[k], blockvec[k], res); // __hfma(a, b, c): a * b + c
        }
        atomicAdd(&mul[b * width + w], res);
        __syncthreads();
    }
}

// C++-CUDA methods


/**
 * Performs quantized matrix multiplication on CUDA.
 * This function multiplies an input vector `vec` by a matrix `mat`, applying quantization
 * based on the provided scales, zeros, and bit width (`w_bit`). It supports asymmetric
 * quantization if `asym` is true. The function is templated to work with different data
 * types (float32, bfloat16, half precision) for the input vector and supports different
 * quantization bit widths.
 *
 * @param vec A 2D tensor representing the batch of input vectors.
 * @param mat A 2D tensor representing the quantized weight matrix.
 * @param mul A 2D tensor for element-wise multiplication in the quantization formula.
 * @param scales A 2D tensor containing the scale factors for quantization.
 * @param zeros A 2D tensor containing the zero points for quantization.
 * @param g_idx A tensor containing group indices for grouped convolution (not used in plain matmul).
 * @param w_bit The bit width for quantization (e.g., 1, 2, 4, 8 bits).
 * @param asym A boolean flag indicating whether asymmetric quantization is used.
 *
 * The function dynamically selects the CUDA kernel based on the bit width and data type,
 * and it organizes the CUDA grid and block dimensions based on the size of the input
 * and weight tensors. It supports error handling for unsupported bit widths and tensor
 * data types.
 *
 * Note: This function requires CUDA support and is intended to be compiled and run on
 * NVIDIA GPUs. It directly interacts with CUDA kernels (`quant_mm_kernel`, `quant_mm_kernel_asym`)
 * which are assumed to be defined elsewhere in the codebase.
 */
void quantmatmul_cuda(
	torch::Tensor vec,
	torch::Tensor mat,
	torch::Tensor mul,
	torch::Tensor scales,
	torch::Tensor zeros,
	torch::Tensor g_idx,
	int w_bit,
	bool asym){
	int batch = vec.size(0);        // batch
	int vec_height = vec.size(1);   // in_channels
	int height = mat.size(0);       // in_channels/32*bits
	int width = mat.size(1);        // out_channels
	int zero_width = zeros.size(1); // out_channels/32 * bits

	int BLOCKHEIGHT = 0;
	if(w_bit == 8){
		BLOCKHEIGHT = BLOCKHEIGHT8;
	}else if(w_bit == 4){
		BLOCKHEIGHT = BLOCKHEIGHT4;
	}else if(w_bit == 2){
		BLOCKHEIGHT = BLOCKHEIGHT2;
	}else if(w_bit == 1){
		BLOCKHEIGHT = BLOCKHEIGHT1;
	}else{
	    std::cerr << "w_bit:"<< w_bit<<" has not been supported yet!" << std::endl;
	    exit(EXIT_FAILURE);
	}

	dim3 blocks(
		(height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
		(width + BLOCKWIDTH - 1) / BLOCKWIDTH
	);
	dim3 threads(BLOCKWIDTH);

    if(vec.dtype() == torch::kFloat32){
        if(asym){
	        quant_mm_kernel_asym<<<blocks, threads>>>(
				vec.data_ptr<float>(), mat.data_ptr<int>(), mul.data_ptr<float>(),
				scales.data_ptr<float>(), zeros.data_ptr<int>(), g_idx.data_ptr<int>(),
				batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
			);
		}else{
            quant_mm_kernel<<<blocks, threads>>>(
				vec.data_ptr<float>(), mat.data_ptr<int>(), mul.data_ptr<float>(),
				scales.data_ptr<float>(), zeros.data_ptr<float>(), g_idx.data_ptr<int>(),
				batch, vec_height, height, width, w_bit, BLOCKHEIGHT
			);
		}
    }else if(vec.dtype() == torch::kBFloat16){
        if(asym){
	        quant_mm_kernel_asym<<<blocks, threads>>>(
				reinterpret_cast<__nv_bfloat16 *>(vec.data_ptr()),
				mat.data_ptr<int>(),
				reinterpret_cast<__nv_bfloat16 *>(mul.data_ptr()),
				reinterpret_cast<__nv_bfloat16 *>(scales.data_ptr()),
				zeros.data_ptr<int>(), g_idx.data_ptr<int>(),
				batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
			);
		}else{
	        quant_mm_kernel<<<blocks, threads>>>(
				reinterpret_cast<__nv_bfloat16 *>(vec.data_ptr()),
				mat.data_ptr<int>(),
				reinterpret_cast<__nv_bfloat16 *>(mul.data_ptr()),
				reinterpret_cast<__nv_bfloat16 *>(scales.data_ptr()),
				reinterpret_cast<__nv_bfloat16 *>(zeros.data_ptr()),
				g_idx.data_ptr<int>(),
				batch, vec_height, height, width, w_bit, BLOCKHEIGHT
			);
		}
    }else if(vec.dtype() == torch::kHalf){
        if(asym){
	        quant_mm_kernel_asym<<<blocks, threads>>>(
				reinterpret_cast<__half *>(vec.data_ptr()),
				mat.data_ptr<int>(),
				reinterpret_cast<__half *>(mul.data_ptr()),
				reinterpret_cast<__half *>(scales.data_ptr()),
				zeros.data_ptr<int>(), g_idx.data_ptr<int>(),
				batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
			);
		}else{
	        quant_mm_kernel<<<blocks, threads>>>(
				reinterpret_cast<__half *>(vec.data_ptr()),
				mat.data_ptr<int>(),
				reinterpret_cast<__half *>(mul.data_ptr()),
				reinterpret_cast<__half *>(scales.data_ptr()),
				reinterpret_cast<__half *>(zeros.data_ptr()),
				g_idx.data_ptr<int>(),
				batch, vec_height, height, width, w_bit, BLOCKHEIGHT
			);
		}
    }else{
        std::cerr << "tensor type not supported: " << vec.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }
}

// nn specific methods

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
    bool asym){
	const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto option_output = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    // get m, k, n
    int m = x.size(0);
    int k = x.size(1);
    int n = qweight.size(1); // shape(k/32*w_bit, n)
    auto output = torch::zeros({m, n}, option_output);
	if(a_bit == 16){
		quantmatmul_cuda(x, qweight, output, scales, qzeros, g_idx, w_bit, asym);
	}else{
        std::cerr << "a_bit:"<< a_bit<<" has not been supported yet!" << std::endl;
        exit(EXIT_FAILURE);
	}
	return output;
}


// ================================================================ //
//                            backward                              //
// ================================================================ //
//backward: add transpose function

///float
__global__ void back_quant_mm_kernel_asym(
    const float* __restrict__ vec,
    const int* __restrict__ mat,
    float* __restrict__ mul,
    const float* __restrict__ scales,
    const int* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int w_bit,
    int BLOCKHEIGHT
) {

    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH_BACK * blockIdx.y + threadIdx.x;

    float blockvec[BLOCKWIDTH_BACK];

    int nb = (32 / w_bit);//w_bit=2, 4, 8; nb=16, 8, 4;
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    float w_tmp;
    __shared__ float weight[256][BLOCKWIDTH_BACK];

    int z_w = w / nb;
    int z_mod = (w % nb) * w_bit;

    for (k = 0; k < 256 ; ++k){
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        if (threadIdx.x < 32){
            g = as_int(g_idx[g_h + k]);
            float scale = scales[g * width + w];
            float zero;
            if(w_bit == 8){
                zero = float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0xFF) + 1);
                w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF);
            }else if(w_bit == 4){
                zero = float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0xF) + 1);
                w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);
            }else if(w_bit == 2){
                zero = float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1);
                w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3);
            }else{
                zero = float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x1) + 1);
                w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1);
            }
            weight[k][threadIdx.x] = scale * (w_tmp - zero);
        }
    }

    float res;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < BLOCKWIDTH_BACK; ++i) {
            blockvec[i] = vec[b * vec_height + i + blockIdx.y * BLOCKWIDTH_BACK];
        }
        __syncthreads();
        res = 0.0;
        for (int i = 0; i < BLOCKWIDTH_BACK; i++) {
            res += weight[threadIdx.x][i] * blockvec[i];
        }
        atomicAdd(&mul[b * height * nb + threadIdx.x + blockIdx.x * 256], res);
        __syncthreads();
    }
}

///half
__global__ void back_quant_mm_kernel_asym(
    const __half* __restrict__ vec,
    const int* __restrict__ mat,
    __half* __restrict__ mul,
    const __half* __restrict__ scales,
    const int* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int w_bit,
    int BLOCKHEIGHT
) {

    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH_BACK * blockIdx.y + threadIdx.x;

    __half blockvec[BLOCKWIDTH_BACK];

    int nb = (32 / w_bit);//w_bit=2, 4, 8; nb=16, 8, 4;
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    __half w_tmp;
    __shared__ __half weight[256][BLOCKWIDTH_BACK];

    int z_w = w / nb;
    int z_mod = (w % nb) * w_bit;

    for (k = 0; k < 256 ; ++k){
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        if (threadIdx.x < 32){
            g = as_int(g_idx[g_h + k]);
            __half scale = scales[g * width + w];
            __half zero;
            if(w_bit == 8){
                zero = __float2half( float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0xFF) + 1));
                w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF));
            }else if(w_bit == 4){
                zero = __float2half( float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0xF) + 1));
                w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF));
            }else if(w_bit == 2){
                zero = __float2half( float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1));
                w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3));
            }else{
                zero = __float2half( float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x1) + 1));
                w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1));
            }
            weight[k][threadIdx.x] = __hmul(scale, __hsub(w_tmp, zero));
        }
    }

    __half res;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < BLOCKWIDTH_BACK; ++i) {
            blockvec[i] = vec[b * vec_height + i + blockIdx.y * BLOCKWIDTH_BACK];
        }
        __syncthreads();
        res = __float2half(0.0f);
        for (int i = 0; i < BLOCKWIDTH_BACK; i++) {
            res = __hfma(weight[threadIdx.x][i], blockvec[i], res); // __hfma(a, b, c): a * b + c
        }
        atomicAdd(&mul[b * height * nb + threadIdx.x + blockIdx.x * 256], res);
        __syncthreads();
    }
}

///bf16
__global__ void back_quant_mm_kernel_asym(
    const __nv_bfloat16* __restrict__ vec,
    const int* __restrict__ mat,
    __nv_bfloat16* __restrict__ mul,
    const __nv_bfloat16* __restrict__ scales,
    const int* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int w_bit,
    int BLOCKHEIGHT
) {

    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH_BACK * blockIdx.y + threadIdx.x;

    __nv_bfloat16 blockvec[BLOCKWIDTH_BACK];

    int nb = (32 / w_bit);//w_bit=2, 4, 8; nb=16, 8, 4;
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    __nv_bfloat16 w_tmp;
    __shared__ __nv_bfloat16 weight[256][BLOCKWIDTH_BACK];

    int z_w = w / nb;
    int z_mod = (w % nb) * w_bit;

    for (k = 0; k < 256 ; ++k){
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        if (threadIdx.x < 32){
            g = as_int(g_idx[g_h + k]);
            __nv_bfloat16 scale = scales[g * width + w];
            __nv_bfloat16 zero;
            if(w_bit == 8){
                zero = __float2bfloat16( float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0xFF) + 1));
                w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF));
            }else if(w_bit == 4){
                zero = __float2bfloat16( float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0xF) + 1));
                w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF));
            }else if(w_bit == 2){
                zero = __float2bfloat16( float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1));
                w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3));
            }else{
                zero = __float2bfloat16( float((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x1) + 1));
                w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1));
            }
            weight[k][threadIdx.x] = __hmul(scale, __hsub(w_tmp, zero));
        }
    }

    __nv_bfloat16 res;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < BLOCKWIDTH_BACK; ++i) {
            blockvec[i] = vec[b * vec_height + i + blockIdx.y * BLOCKWIDTH_BACK];
        }
        __syncthreads();
        res = __float2bfloat16(0.0f);
        for (int i = 0; i < BLOCKWIDTH_BACK; i++) {
            res = __hfma(weight[threadIdx.x][i], blockvec[i], res); // __hfma(a, b, c): a * b + c
        }
        atomicAdd(&mul[b * height * nb + threadIdx.x + blockIdx.x * 256], res);
        __syncthreads();
    }
}

///float
__global__ void back_quant_mm_kernel(
    const float* __restrict__ vec,
    const int* __restrict__ mat,
    float* __restrict__ mul,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int w_bit,
    int BLOCKHEIGHT
) {

    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH_BACK * blockIdx.y + threadIdx.x;

    float blockvec[BLOCKWIDTH_BACK];

    int nb = (32 / w_bit);//w_bit=2, 4, 8; nb=16, 8, 4;
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    float w_tmp;
    __shared__ float weight[256][BLOCKWIDTH_BACK];

    for (k = 0; k < 256 ; ++k){
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        if (threadIdx.x < 32){
            g = as_int(g_idx[g_h + k]);
            float scale = scales[g * width + w];
            float zero = zeros[g * width + w];
            if(w_bit == 8){
                w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF);
            }else if(w_bit == 4){
                w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);
            }else if(w_bit == 2){
                w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3);
            }else{
                w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1);
            }
            weight[k][threadIdx.x] = scale * w_tmp - zero;
        }
    }

    float res;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < BLOCKWIDTH_BACK; ++i) {
            blockvec[i] = vec[b * vec_height + i + blockIdx.y * BLOCKWIDTH_BACK];
        }
        __syncthreads();
        res = 0.0;
        for (int i = 0; i < BLOCKWIDTH_BACK; i++) {
            res += weight[threadIdx.x][i] * blockvec[i];
        }
        atomicAdd(&mul[b * height * nb + threadIdx.x + blockIdx.x * 256], res);
        __syncthreads();
    }
}

///half
__global__ void back_quant_mm_kernel(
    const __half* __restrict__ vec,
    const int* __restrict__ mat,
    __half* __restrict__ mul,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int w_bit,
    int BLOCKHEIGHT
) {

    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH_BACK * blockIdx.y + threadIdx.x;

    __half blockvec[BLOCKWIDTH_BACK];

    int nb = (32 / w_bit);//w_bit=2, 4, 8; nb=16, 8, 4;
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    __half w_tmp;
    __shared__ __half weight[256][BLOCKWIDTH_BACK];

    for (k = 0; k < 256 ; ++k){
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        if (threadIdx.x < 32){
            g = as_int(g_idx[g_h + k]);
            __half scale = scales[g * width + w];
            __half zero = zeros[g * width + w];
            if(w_bit == 8){
                w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF));
            }else if(w_bit == 4){
                w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF));
            }else if(w_bit == 2){
                w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3));
            }else{
                w_tmp = __float2half(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1));
            }
            weight[k][threadIdx.x] = __hfma(scale, w_tmp, __hneg(zero));
        }
    }

    __half res;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < BLOCKWIDTH_BACK; ++i) {
            blockvec[i] = vec[b * vec_height + i + blockIdx.y * BLOCKWIDTH_BACK];
        }
        __syncthreads();
        res = __float2half(0.0);
        for (int i = 0; i < BLOCKWIDTH_BACK; i++) {
            res = __hfma(weight[threadIdx.x][i], blockvec[i], res);
        }
        atomicAdd(&mul[b * height * nb + threadIdx.x + blockIdx.x * 256], res);
        __syncthreads();
    }
}

///bf16
__global__ void back_quant_mm_kernel(
    const __nv_bfloat16* __restrict__ vec,
    const int* __restrict__ mat,
    __nv_bfloat16* __restrict__ mul,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int w_bit,
    int BLOCKHEIGHT
){

    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH_BACK * blockIdx.y + threadIdx.x;

    __nv_bfloat16 blockvec[BLOCKWIDTH_BACK];

    int nb = (32 / w_bit);//w_bit=2, 4, 8; nb=16, 8, 4;
    int i = width * h + w;
    int g_h = h * nb;
    int k;
    unsigned int g;
    __nv_bfloat16 w_tmp;
    __shared__ __nv_bfloat16 weight[256][BLOCKWIDTH_BACK];

    for (k = 0; k < 256 ; ++k){
        int k_w = (k / nb);
        int k_bit = (k % nb) * w_bit;

        if (threadIdx.x < 32){
            g = as_int(g_idx[g_h + k]);
            __nv_bfloat16 scale = scales[g * width + w];
            __nv_bfloat16 zero = zeros[g * width + w];
            if(w_bit == 8){
                w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF));
            }else if(w_bit == 4){
                w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF));
            }else if(w_bit == 2){
                w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3));
            }else{
                w_tmp = __float2bfloat16(float((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x1));
            }
            weight[k][threadIdx.x] = __hfma(scale, w_tmp, __hneg(zero));
        }
    }

    __nv_bfloat16 res;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < BLOCKWIDTH_BACK; ++i) {
            blockvec[i] = vec[b * vec_height + i + blockIdx.y * BLOCKWIDTH_BACK];
        }
        __syncthreads();
        res = __float2bfloat16(0.0);
        for (int i = 0; i < BLOCKWIDTH_BACK; i++) {
            res = __hfma(weight[threadIdx.x][i], blockvec[i], res);
        }
        atomicAdd(&mul[b * height * nb + threadIdx.x + blockIdx.x * 256], res);
        __syncthreads();
    }
}


/**
 * Performs the backward pass of a quantized matrix multiplication operation on CUDA.
 *
 * This function computes the gradients for the quantized matrix multiplication operation
 * given the input vector `vec`, the quantized matrix `mat`, the multiplier tensor `mul`,
 * scale factors `scales`, zero points `zeros`, and gradient indices `g_idx`. The operation
 * supports different bit-widths for weights, specified by `w_bit`, and can operate in
 * asymmetric mode if `asym` is true.
 *
 * @param vec The input tensor, typically representing activations in a neural network layer.
 *            Shape: [batch, in_channels].
 * @param mat The quantized weight matrix. Shape: [in_channels/32*bits, out_channels].
 * @param mul The multiplier tensor for quantization, used in scaling the input.
 * @param scales The scale factors used for quantization.
 * @param zeros The zero points used for quantization, defining the value that represents 0.
 * @param g_idx Gradient indices, used for mapping gradients in sparse update scenarios.
 * @param w_bit The bit-width of the quantized weights (1, 2, 4, or 8 bits).
 * @param asym A boolean flag indicating whether to perform asymmetric quantization.
 *
 * The function dynamically selects the appropriate CUDA kernel based on the input tensor
 * data type (float32, bfloat16, or half) and the quantization mode (asymmetric or symmetric).
 * It calculates the number of blocks and threads for the CUDA kernel launch based on the
 * dimensions of the input and weight tensors, as well as the specified bit-width.
 *
 * Note: This function exits the program with an error message if `w_bit` is not supported
 * or if the input tensor type is not supported.
 */
void backward_quantmatmul_cuda(
	torch::Tensor vec,
	torch::Tensor mat,
	torch::Tensor mul,
	torch::Tensor scales,
	torch::Tensor zeros,
	torch::Tensor g_idx,
	int w_bit,
	bool asym
){
	int batch = vec.size(0);        // batch
	int vec_height = vec.size(1);   // in_channels
	int height = mat.size(0);       // in_channels/32*bits
	int width = mat.size(1);        // out_channels
	int zero_width = zeros.size(1); // out_channels/32 * bits

	int BLOCKHEIGHT = 0;
	if(w_bit == 8){
		BLOCKHEIGHT = BLOCKHEIGHT8;
	}else if(w_bit == 4){
		BLOCKHEIGHT = BLOCKHEIGHT4;
	}else if(w_bit == 2){
		BLOCKHEIGHT = BLOCKHEIGHT2;
	}else if(w_bit == 1){
		BLOCKHEIGHT = BLOCKHEIGHT1;
	}else{
	    std::cerr << "w_bit:"<< w_bit<<" has not been supported yet!" << std::endl;
	    exit(EXIT_FAILURE);
	}

	dim3 blocks(
		(height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
		(width + BLOCKWIDTH_BACK - 1) / BLOCKWIDTH_BACK
	);
	dim3 threads(256);

    if(vec.dtype() == torch::kFloat32){
        if(asym){
		    back_quant_mm_kernel_asym<<<blocks, threads>>>(
		        vec.data_ptr<float>(), mat.data_ptr<int>(), mul.data_ptr<float>(),
		        scales.data_ptr<float>(), zeros.data_ptr<int>(), g_idx.data_ptr<int>(),
		        batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
		    );
		}else{
            back_quant_mm_kernel<<<blocks, threads>>>(
				vec.data_ptr<float>(), mat.data_ptr<int>(), mul.data_ptr<float>(),
				scales.data_ptr<float>(), zeros.data_ptr<float>(), g_idx.data_ptr<int>(),
				batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
			);
		}
    }else if(vec.dtype() == torch::kBFloat16){
        if(asym){
	        back_quant_mm_kernel_asym<<<blocks, threads>>>(
				reinterpret_cast<__nv_bfloat16 *>(vec.data_ptr()),
				mat.data_ptr<int>(),
				reinterpret_cast<__nv_bfloat16 *>(mul.data_ptr()),
				reinterpret_cast<__nv_bfloat16 *>(scales.data_ptr()),
				zeros.data_ptr<int>(), g_idx.data_ptr<int>(),
				batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
			);
		}else{
	        back_quant_mm_kernel<<<blocks, threads>>>(
				reinterpret_cast<__nv_bfloat16 *>(vec.data_ptr()),
				mat.data_ptr<int>(),
				reinterpret_cast<__nv_bfloat16 *>(mul.data_ptr()),
				reinterpret_cast<__nv_bfloat16 *>(scales.data_ptr()),
				reinterpret_cast<__nv_bfloat16 *>(zeros.data_ptr()),
				g_idx.data_ptr<int>(),
				batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
			);
		}
    }else if(vec.dtype() == torch::kHalf){
        if(asym){
	        back_quant_mm_kernel_asym<<<blocks, threads>>>(
				reinterpret_cast<__half *>(vec.data_ptr()),
				mat.data_ptr<int>(),
				reinterpret_cast<__half *>(mul.data_ptr()),
				reinterpret_cast<__half *>(scales.data_ptr()),
				zeros.data_ptr<int>(), g_idx.data_ptr<int>(),
				batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
			);
		}else{
	        back_quant_mm_kernel<<<blocks, threads>>>(
				reinterpret_cast<__half *>(vec.data_ptr()),
				mat.data_ptr<int>(),
				reinterpret_cast<__half *>(mul.data_ptr()),
				reinterpret_cast<__half *>(scales.data_ptr()),
				reinterpret_cast<__half *>(zeros.data_ptr()),
				g_idx.data_ptr<int>(),
				batch, vec_height, height, width, zero_width, w_bit, BLOCKHEIGHT
			);
		}
    }else{
        std::cerr << "tensor type not supported: " << vec.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }
}


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
    bool asym
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(output_gradient));
    auto option_output = torch::TensorOptions().dtype(output_gradient.dtype()).device(output_gradient.device());
    // get m, k, n
    int m = output_gradient.size(0);
    int n = output_gradient.size(1);
    int k = qweight.size(0); // shape(k/32*w_bit, n)
    int k1= k*32/w_bit;
    auto output = torch::zeros({m, k1}, option_output);
    if(a_bit == 16){
        backward_quantmatmul_cuda(output_gradient, qweight, output, scales, qzeros, g_idx, w_bit, asym);
    }else{
        std::cerr << "a_bit:"<< a_bit<<" has not been supported yet!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return output;
}