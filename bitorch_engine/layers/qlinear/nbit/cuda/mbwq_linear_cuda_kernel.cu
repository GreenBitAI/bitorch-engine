#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <utility>
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

#include "exl2/matrix_view.cuh"
#include "exl2/util.cuh"
#include "exl2/quant/qdq_2.cuh"
#include "exl2/quant/qdq_3.cuh"
#include "exl2/quant/qdq_4.cuh"
#include "exl2/quant/qdq_5.cuh"
#include "exl2/quant/qdq_6.cuh"
#include "exl2/quant/qdq_8.cuh"
#include "exl2/q_gemm_kernel_gptq.cuh"
#include "exl2/kernel_select.cuh"

#define THREADS_X 32
#define THREADS_Y 32
#define BLOCK_KN_SIZE 128


// CUDA kernel function

/**
 * This CUDA kernel is designed to shuffle quantized weights during data loading.
 * It supports different quantization bit-widths by segmenting the quantized data
 * into blocks of varying bit lengths and shuffling each block accordingly.
 *
 * Parameters:
 *    b_q_weight: Pointer to the buffer containing quantized weights that need to be shuffled.
 *    size_k: The total number of elements along the 'k' dimension (not used directly in this kernel).
 *    size_n: The total number of elements along the 'n' dimension, defining the range of threads.
 *    rows_8: The boundary index up to which 8-bit quantization blocks are shuffled.
 *    rows_6: The boundary index up to which 6-bit quantization blocks are shuffled, after 8-bit blocks.
 *    rows_5: The boundary index up to which 5-bit quantization blocks are shuffled, following 6-bit blocks.
 *    rows_4: The boundary index up to which 4-bit quantization blocks are shuffled, following 5-bit blocks.
 *    rows_3: The boundary index up to which 3-bit quantization blocks are shuffled, following 4-bit blocks.
 *    rows_2: The boundary index up to which 2-bit quantization blocks are shuffled, following 3-bit blocks.
 *
 * The kernel iterates over the 'n' dimension within the range specified by `size_n`.
 * For each thread, it sequentially processes blocks of quantized weights, starting from the highest
 * bit-width (8-bit) to the lowest (2-bit), calling different shuffling functions for each bit-width.
 * The shuffling functions are assumed to be defined elsewhere (`shuffle_8bit_4`, `shuffle_6bit_16`, etc.)
 * and are specialized for handling specific quantization formats.
 *
 * This kernel efficiently reorganizes quantized weight data to optimize memory access patterns
 * or prepare the data for further processing, such as during the initialization phase of neural network inference.
 */
__global__ void shuffle_kernel
(
    uint32_t* __restrict__ b_q_weight,
    const int size_k,
    const int size_n,
    const int rows_8,
    const int rows_6,
    const int rows_5,
    const int rows_4,
    const int rows_3,
    const int rows_2
)
{
    int n = blockIdx.x * THREADS_X + threadIdx.x;
    if (n >= size_n) return;
    int k = 0;
    uint32_t* b_ptr = b_q_weight + n;
    while (k < rows_8) { shuffle_8bit_4 (b_ptr, size_n); b_ptr += 1 * size_n; k +=  4; }
    while (k < rows_6) { shuffle_6bit_16(b_ptr, size_n); b_ptr += 3 * size_n; k += 16; }
    while (k < rows_5) { shuffle_5bit_32(b_ptr, size_n); b_ptr += 5 * size_n; k += 32; }
    while (k < rows_4) { shuffle_4bit_8 (b_ptr, size_n); b_ptr += 1 * size_n; k +=  8; }
    while (k < rows_3) { shuffle_3bit_32(b_ptr, size_n); b_ptr += 3 * size_n; k += 32; }
    while (k < rows_2) { shuffle_2bit_16(b_ptr, size_n); b_ptr += 1 * size_n; k += 16; }
}


/*
 * Reconstruct fp16 weight tensor [k,n] exl2 format
 */
__global__ void reconstruct_exl2_kernel
(
    const uint32_t* __restrict__ b_q_weight,
    half* __restrict__ b,
    const half* __restrict__ b_scale,
    const half* __restrict__ b_q_zero,
    const uint16_t* __restrict__ b_q_perm,
    const uint16_t* __restrict__ b_q_group_map,
    const int size_k,
    const int size_n,
    const int groups,
    const int rows_8,
    const int rows_6,
    const int rows_5,
    const int rows_4,
    const int rows_3,
    const int rows_2
){
	MatrixView_half_rw b_(b, size_k, size_n);
    MatrixView_half b_scale_(b_scale, groups, size_n);
    MatrixView_half b_zero_(b_q_zero, groups, size_n);

    int offset_k = BLOCK_KN_SIZE * blockIdx.y;
    int offset_n = BLOCK_KN_SIZE * blockIdx.x;

    // Preload remapping table

    int t = threadIdx.x;
    __shared__ uint16_t perm[BLOCK_KN_SIZE];
    if (offset_k + t < size_k)
        perm[t] = b_q_perm[offset_k + t];

    // Column

    int n = offset_n + t;
    if (n >= size_n) return;

    // Find initial group

    int group = b_q_group_map[offset_k * 2];

    int pre_rows_8 = min(rows_8, offset_k);
    int pre_rows_6 = offset_k > rows_8 ? min(rows_6, offset_k) - rows_8 : 0;
    int pre_rows_5 = offset_k > rows_6 ? min(rows_5, offset_k) - rows_6 : 0;
    int pre_rows_4 = offset_k > rows_5 ? min(rows_4, offset_k) - rows_5 : 0;
    int pre_rows_3 = offset_k > rows_4 ? min(rows_3, offset_k) - rows_4 : 0;
    int pre_rows_2 = offset_k > rows_3 ? min(rows_2, offset_k) - rows_3 : 0;
    int qk = 0;
    qk += pre_rows_8 / 32 * 8;
    qk += pre_rows_6 / 32 * 6;
    qk += pre_rows_5 / 32 * 5;
    qk += pre_rows_4 / 32 * 4;
    qk += pre_rows_3 / 32 * 3;
    qk += pre_rows_2 / 32 * 2;

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    half2 qs_h2 = b_scale_.item_half2half2(group, n);
    half2 qz_h2 = b_zero_.item_half2half2(group, n);

    int nextgroup = offset_k + b_q_group_map[offset_k * 2 + 1];

    int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
    int k = offset_k;
    int lk = 0;

    __syncthreads();

    while (k < rows_8 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            qs_h2 = b_scale_.item_half2half2(group, n);
            qz_h2 = b_zero_.item_half2half2(group, n);

            nextgroup += b_q_group_map[k * 2 + 1];
        }
        for (int p = 0; p < 4; p++)
        {
            half2 dq[4];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            uint32_t q_1 = *b_ptr; b_ptr += size_n;
            dequant_8bit_8(q_0, q_1, dq, size_n);
            for (int j = 0; j < 4; j++)
                dq[j] = __hfma2(dq[j], qs_h2, __hneg2(qz_h2));
            half* dqh = (half*) dq;
            for (int j = 0; j < 8; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_6 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            qs_h2 = b_scale_.item_half2half2(group, n);
            qz_h2 = b_zero_.item_half2half2(group, n);

            nextgroup += b_q_group_map[k * 2 + 1];
        }
        for (int p = 0; p < 2; p++)
        {
            half2 dq[8];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            uint32_t q_1 = *b_ptr; b_ptr += size_n;
            uint32_t q_2 = *b_ptr; b_ptr += size_n;
            dequant_6bit_16(q_0, q_1, q_2, dq, size_n);
            for (int j = 0; j < 8; j++)
                dq[j] = __hfma2(dq[j], qs_h2, __hneg2(qz_h2));
            half* dqh = (half*) dq;
            for (int j = 0; j < 16; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_5 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            qs_h2 = b_scale_.item_half2half2(group, n);
            qz_h2 = b_zero_.item_half2half2(group, n);

            nextgroup += b_q_group_map[k * 2 + 1];
        }
        for (int p = 0; p < 1; p++)
        {
            half2 dq[16];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            uint32_t q_1 = *b_ptr; b_ptr += size_n;
            uint32_t q_2 = *b_ptr; b_ptr += size_n;
            uint32_t q_3 = *b_ptr; b_ptr += size_n;
            uint32_t q_4 = *b_ptr; b_ptr += size_n;
            dequant_5bit_32(q_0, q_1, q_2, q_3, q_4, dq, size_n);
            for (int j = 0; j < 16; j++)
                dq[j] = __hfma2(dq[j], qs_h2, __hneg2(qz_h2));
            half* dqh = (half*) dq;
            for (int j = 0; j < 32; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_4 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            qs_h2 = b_scale_.item_half2half2(group, n);
            qz_h2 = b_zero_.item_half2half2(group, n);

            nextgroup += b_q_group_map[k * 2 + 1];
        }
        for (int p = 0; p < 4; p++)
        {
            half2 dq[4];
            uint32_t q_0 = *b_ptr;
            b_ptr += size_n;
            dequant_4bit_8(q_0, dq, size_n);
            for (int j = 0; j < 4; j++)
                dq[j] = __hfma2(dq[j], qs_h2, __hneg2(qz_h2));
            half* dqh = (half*) dq;
            for (int j = 0; j < 8; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_3 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            qs_h2 = b_scale_.item_half2half2(group, n);
            qz_h2 = b_zero_.item_half2half2(group, n);

            nextgroup += b_q_group_map[k * 2 + 1];
        }
        for (int p = 0; p < 1; p++)
        {
            half2 dq[16];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            uint32_t q_1 = *b_ptr; b_ptr += size_n;
            uint32_t q_2 = *b_ptr; b_ptr += size_n;
            dequant_3bit_32(q_0, q_1, q_2, dq, size_n);
            for (int j = 0; j < 16; j++)
                dq[j] = __hfma2(dq[j], qs_h2, __hneg2(qz_h2));
            half* dqh = (half*) dq;
            for (int j = 0; j < 32; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_2 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            qs_h2 = b_scale_.item_half2half2(group, n);
            qz_h2 = b_zero_.item_half2half2(group, n);

            nextgroup += b_q_group_map[k * 2 + 1];
        }
        for (int p = 0; p < 1; p++)
        {
            half2 dq[8];
            uint32_t q_0 = *b_ptr;
            b_ptr += size_n;
            dequant_2bit_16(q_0, dq, size_n);
            for (int j = 0; j < 8; j++)
                dq[j] = __hfma2(dq[j], qs_h2, __hneg2(qz_h2));
            half* dqh = (half*) dq;
            for (int j = 0; j < 16; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 16;
    }
}


/*
 * Reconstruct fp16 weight tensor [k,n] GPTQ style
 */
__global__ void reconstruct_q4_gptq_kernel(
    const uint32_t* __restrict__ b_q_weight,
    const half* __restrict__ b_gptq_zeros,
    const half* __restrict__ b_gptq_scales,
    const int size_k,
    const int size_n,
    const int groupsize,
    const int groups,
    half* __restrict__ b,
    const uint16_t* __restrict__ b_q_perm
){
    MatrixView_half_rw b_(b, size_k, size_n);
	MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);
    MatrixView_half b_gptq_zeros_(b_gptq_zeros, groups, size_n);

    int offset_k = BLOCK_KN_SIZE * blockIdx.y;
    int offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

    int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

    // Preload remapping table

    __shared__ uint16_t perm[BLOCK_KN_SIZE];
    int t = threadIdx.x;

    if (b_q_perm){
        if (offset_k + t < size_k)
            perm[t] = b_q_perm[offset_k + t];
    }

    // Column

    int n = offset_n + t * 4;
    if (n >= size_n) return;

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // b offset

    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    // Initial zeros/scale

    half2 zeros[4];
    half2 scales[4];

    b_gptq_zeros_.item4_h2(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);

    __syncthreads();

    int k = offset_k;
    int lk = 0;

    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_zeros_.item4_h2(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
        }

        for (int p = 0; p < 4; p++)
        {
            half2 dq[4][4];
            const int4* b_ptr4 = (int4*) b_ptr;
            int4 load_int4 = *b_ptr4;

            dequant_4bit_8(load_int4.x, dq[0]);
            dequant_4bit_8(load_int4.y, dq[1]);
            dequant_4bit_8(load_int4.z, dq[2]);
            dequant_4bit_8(load_int4.w, dq[3]);

            b_ptr += size_n;

            for (int j = 0; j < 4; j++)
            {
                for (int v = 0; v < 4; v++) dq[v][j] = __hfma2(scales[v], dq[v][j], __hneg2(zeros[v]));

                if (b_q_perm){
                    b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
                    b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                } else {
	                b_.set4(offset_k + lk++, n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
	                b_.set4(offset_k + lk++, n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                }
            }
        }
        k += 32;
    }
}


__global__ void reconstruct_q2_gptq_kernel(
    const uint32_t* __restrict__ b_q_weight,
    const half* __restrict__ b_gptq_zeros,
    const half* __restrict__ b_gptq_scales,
    const int size_k,
    const int size_n,
    const int groupsize,
    const int groups,
    half* __restrict__ b,
    const uint16_t* __restrict__ b_q_perm
) {
    MatrixView_half_rw b_(b, size_k, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);
    MatrixView_half b_gptq_zeros_(b_gptq_zeros, groups, size_n);

    // Adjust offsets for 2-bit quantization
    int offset_k = BLOCK_KN_SIZE * blockIdx.y;
	// Keep the * 4 to maintain compatibility with the host code's grid configuration
	int offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

    int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

    __shared__ uint16_t perm[BLOCK_KN_SIZE];
    int t = threadIdx.x;

    if (b_q_perm){
        if (offset_k + t < size_k)
            perm[t] = b_q_perm[offset_k + t];
    }

    // Keep the * 4 for the calculation of n to match the thread distribution strategy
	int n = offset_n + t * 4;
    if (n >= size_n) return;

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    const uint32_t* b_ptr = b_q_weight + (offset_k / 16) * size_n + n; // Adjusted index for 2-bit

    // Fetch initial zeros and scales
    half2 zeros[4];
    half2 scales[4];
    b_gptq_zeros_.item4_h2(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);

    __syncthreads();

    int k = offset_k;
    int lk = 0;

    while (k < end_k) {

        if (k == nextgroup) {
            group++;
            nextgroup += groupsize;
            b_gptq_zeros_.item4_h2(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
        }

        for (int p = 0; p < 1; p++){
	        int4 load_int4[1];
            load_int4[0] = *((int4*) b_ptr);
            b_ptr += size_n;

            half2 dq[4][8];
            dequant_2bit_16(load_int4[0].x, dq[0], size_n);
            dequant_2bit_16(load_int4[0].y, dq[1], size_n);
            dequant_2bit_16(load_int4[0].z, dq[2], size_n);
            dequant_2bit_16(load_int4[0].w, dq[3], size_n);

	        // Apply scales and zeros, then write back the dequantized values
	        for (int j = 0; j < 8; j++){
		        for (int i = 0; i < 4; i++){
		            dq[i][j] = __hfma2(scales[i], dq[i][j], __hneg2(zeros[i]));
                }

                if (b_q_perm){
                    b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
                    b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                } else {
	                b_.set4(offset_k + lk++, n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
	                b_.set4(offset_k + lk++, n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                }
			}
        }
        k += 16; // Increment k considering the processing of 16 values per uint32_t
    }
}



// C++-CUDA methods


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
    int bits
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
    int width = qweight.size(1);      // n

	std::vector<int> vec;
	// start row number of each bit width, it starts from 8-bits, ends by 2-bits
    int rows_8 = 0;
    int rows_6 = 0;
    int rows_5 = 0;
    int rows_4 = 0;
    int rows_3 = 0;
    int rows_2 = 0;
    // ========= bit width permutation ========= //
    // Example: 0b00000010: 2bit, 0b00001110: 2,3,4bit, 0b10111010: 2,4,5,6,8bit
    int kernel_p = 0;

    if(use_mbw){
        // mixed bitwidth configuration:
        uint16_t* cpu_q_groups = (uint16_t*)calloc(groups * 2, sizeof(uint16_t));
        cudaMemcpy(cpu_q_groups, reinterpret_cast<uint16_t *>(cuda_q_groups.data_ptr()), groups * 2 * sizeof(uint16_t), cudaMemcpyDeviceToHost);

        int row = 0;
        for (int i = 0; i < groups; i++){
            int bits = cpu_q_groups[i * 2];
            kernel_p |= (1 << (bits - 1));

            int rows;
            if (i < groups - 1){
                int qrows = cpu_q_groups[i * 2 + 3] - cpu_q_groups[i * 2 + 1];
                rows = qrows * 32 / bits;
            } else {
                rows = height - row;
			}

            if (bits == 8) rows_8 += rows;
            if (bits == 6) rows_6 += rows;
            if (bits == 5) rows_5 += rows;
            if (bits == 4) rows_4 += rows;
            if (bits == 3) rows_3 += rows;
            if (bits == 2) rows_2 += rows;
            row += rows;
        }

        free(cpu_q_groups);

        rows_6 += rows_8;
        rows_5 += rows_6;
        rows_4 += rows_5;
        rows_3 += rows_4;
        rows_2 += rows_3;

        vec.push_back(rows_8);
        vec.push_back(rows_6);
        vec.push_back(rows_5);
        vec.push_back(rows_4);
        vec.push_back(rows_3);
        vec.push_back(rows_2);
        vec.push_back(kernel_p);

    } else {
	    if (bits == 4){
            rows_4 = height;
		} else if (bits == 2){
            rows_2 = height;
		} else {
	        std::cerr << "Error: weight bit width:"<< bits <<" has not been supported yet!" << std::endl;
	        exit(EXIT_FAILURE);
		}
    }

    // Shuffle quantized data
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = 1;

    shuffle_kernel<<<gridDim, blockDim>>>(
        reinterpret_cast<uint32_t *>(qweight.data_ptr()),
        height, width, rows_8, rows_6, rows_5, rows_4,
        rows_3, rows_2);

	return std::make_pair(qweight, vec);
}


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
    torch::Tensor q_perm
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
    int height = qweight.size(0) * (32 / bits); // k
    int width = qweight.size(1);      // n
    int groups = scales.size(0);      // in_channles/group_size

	auto option_output = torch::TensorOptions().dtype(scales.dtype()).device(scales.device());
	auto out = torch::zeros({height, width}, option_output);
	bool is_q_perm_all_zeros = torch::all(q_perm == 0).item<bool>();
	auto perm_value = is_q_perm_all_zeros ? nullptr : reinterpret_cast<uint16_t *>(q_perm.data_ptr());

    dim3 blockDim, gridDim;
    blockDim.x = BLOCK_KN_SIZE;
    blockDim.y = 1;
    gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);

    gridDim.x = DIVIDE(width, BLOCK_KN_SIZE * 4);

    if (bits == 4){
		reconstruct_q4_gptq_kernel<<<gridDim, blockDim>>>(
		    reinterpret_cast<uint32_t *>(qweight.data_ptr()),
		    reinterpret_cast<half *>(zeros.data_ptr()),
		    reinterpret_cast<half *>(scales.data_ptr()),
		    height,
		    width,
		    group_size,
		    groups,
		    reinterpret_cast<half *>(out.data_ptr()),
		    perm_value
		);
	} else if (bits == 2){
		reconstruct_q2_gptq_kernel<<<gridDim, blockDim>>>(
		    reinterpret_cast<uint32_t *>(qweight.data_ptr()),
		    reinterpret_cast<half *>(zeros.data_ptr()),
		    reinterpret_cast<half *>(scales.data_ptr()),
		    height,
		    width,
		    group_size,
		    groups,
		    reinterpret_cast<half *>(out.data_ptr()),
		    perm_value
		);
	} else {
        std::cerr << "Error: weight bit width:"<< bits <<" has not been supported yet!" << std::endl;
        exit(EXIT_FAILURE);
	}
	return out;
}


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
    int bits
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    TORCH_CHECK(x.dtype() == torch::kHalf);
    TORCH_CHECK(x.size(1) == qweight.size(0) * (32 / bits));

	int size_m = x.size(0);       // m
    int size_n = qweight.size(1); // n
	int size_k = x.size(1);       // k
    int groups = scales.size(0);  // in_channles/group_size

	auto option_output = torch::TensorOptions().dtype(x.dtype()).device(x.device());
	auto out = torch::zeros({size_m, size_n}, option_output);

	if (size_m > MAX_Q_GEMM_ROWS){
        // Reconstruct FP16 matrix and using cuBLAS for gemm
        auto fp_w = mbwq_linear_q42fp_weight_cuda(qweight,
									               scales,
									               zeros,
									               group_size,
									               bits,
									               q_perm);
		// indirectly use cublas through torch matmul api
        out = torch::matmul(x, fp_w.to(option_output));

	}else{

		bool is_q_perm_all_zeros = torch::all(q_perm == 0).item<bool>();
		auto perm_value = is_q_perm_all_zeros ? nullptr : reinterpret_cast<uint16_t *>(q_perm.data_ptr());

	    dim3 blockDim, gridDim;
	    blockDim.x = GPTQ_BLOCK_KN_SIZE;
	    blockDim.y = 1;
	    blockDim.z = 1;
	    gridDim.x = DIVIDE(size_n, GPTQ_BLOCK_KN_SIZE * 4);
	    gridDim.y = DIVIDE(size_m, GPTQ_BLOCK_M_SIZE_MAX);
	    gridDim.z = DIVIDE(size_k, GPTQ_BLOCK_KN_SIZE);

	    if (bits == 4){
		    gemm_half_q4_half_gptq_kernel<GPTQ_BLOCK_M_SIZE_MAX><<<gridDim, blockDim>>>(
		        reinterpret_cast<half *>(x.data_ptr()),
		        reinterpret_cast<uint32_t *>(qweight.data_ptr()),
		        reinterpret_cast<half *>(zeros.data_ptr()),
		        reinterpret_cast<half *>(scales.data_ptr()),
		        reinterpret_cast<half *>(out.data_ptr()),
		        size_m,
		        size_n,
		        size_k,
		        groups,
		        group_size,
		        true,
		        perm_value
		    );
		} else if (bits == 2){
		    gemm_half_q2_half_gptq_kernel<GPTQ_BLOCK_M_SIZE_MAX><<<gridDim, blockDim>>>(
		        reinterpret_cast<half *>(x.data_ptr()),
		        reinterpret_cast<uint32_t *>(qweight.data_ptr()),
		        reinterpret_cast<half *>(zeros.data_ptr()),
		        reinterpret_cast<half *>(scales.data_ptr()),
		        reinterpret_cast<half *>(out.data_ptr()),
		        size_m,
		        size_n,
		        size_k,
		        groups,
		        group_size,
		        true,
		        perm_value
		    );
		} else {
	        std::cerr << "Error: weight bit width:"<< bits <<" has not been supported yet!" << std::endl;
	        exit(EXIT_FAILURE);
		}
	}

    return out;
}


/**
 * Converts exl2 format quantized weights into half-precision floating point format using a custom linear reconstruction kernel.
 * This function is designed for CUDA execution, utilizing specific memory layouts and quantization schemes.
 *
 * @param qweight Tensor representing quantized weights, typically stored in an integer format.
 * @param scales Tensor containing scales for quantization levels.
 * @param zeros Tensor representing quantized zero points.
 * @param qperm Tensor representing permutation indices for quantized weight groups.
 * @param qgroup_map Tensor mapping each weight to its corresponding quantization group.
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
    torch::Tensor qperm,
    torch::Tensor qgroup_map,
    std::vector<int> rows
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
    int groups = scales.size(0);
    int height = qperm.size(0);  //k
    int width = qweight.size(1); //n
    auto option_output = torch::TensorOptions().dtype(torch::kHalf).device(qweight.device());
	auto out = torch::zeros({height, width}, option_output);

    int rows_8 = rows[0];
    int rows_6 = rows[1];
    int rows_5 = rows[2];
    int rows_4 = rows[3];
    int rows_3 = rows[4];
    int rows_2 = rows[5];

    dim3 blockDim, gridDim;
    blockDim.x = BLOCK_KN_SIZE;
    blockDim.y = 1;
    gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);
    gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

	reconstruct_exl2_kernel<<<gridDim, blockDim>>>
    (
        reinterpret_cast<uint32_t *>(qweight.data_ptr()),
        reinterpret_cast<half *>(out.data_ptr()),
        reinterpret_cast<half *>(scales.data_ptr()),
        reinterpret_cast<half *>(zeros.data_ptr()),
        reinterpret_cast<uint16_t*>(qperm.data_ptr()),
        reinterpret_cast<uint16_t*>(qgroup_map.data_ptr()),
        height,
        width,
        groups,
        rows_8,
        rows_6,
        rows_5,
        rows_4,
        rows_3,
        rows_2
    );

    return out;
}


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
 * @param qperm Permutation indices for quantized weight matrix, supporting optimized memory access patterns.
 * @param qgroup_map Mapping tensor that associates groups with quantization parameters.
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
    torch::Tensor qperm,
    torch::Tensor qgroup_map,
    std::vector<int> rows,
    bool use_cublas
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    TORCH_CHECK(x.dtype() == torch::kHalf);

	int size_m = x.size(0);       // m
    int size_n = qweight.size(1); // n
	int size_k = qperm.size(0);   // k
    int groups = scales.size(0);

	auto option_output = torch::TensorOptions().dtype(torch::kHalf).device(x.device());
	auto out = torch::zeros({size_m, size_n}, option_output);

	if (use_cublas || size_m > MAX_Q_GEMM_ROWS){
        // Reconstruct FP16 matrix and using cuBLAS for gemm
        auto fp_w = mbwq_linear_exl2fp_weight_cuda(qweight,
									               scales,
									               zeros,
									               qperm,
									               qgroup_map,
									               rows);

        // indirectly use cublas through torch matmul api
        out = torch::matmul(x, fp_w.to(option_output));

	}else{
	    int rows_8 = rows[0];
	    int rows_6 = rows[1];
	    int rows_5 = rows[2];
	    int rows_4 = rows[3];
	    int rows_3 = rows[4];
	    int rows_2 = rows[5];
	    int kernel_p = rows[6];

	    int m_count = min(size_m, EXL2_BLOCK_M_SIZE_MAX);

	    dim3 blockDim, gridDim;
	    blockDim.x = EXL2_BLOCK_KN_SIZE;
	    blockDim.y = 1;
	    blockDim.z = 1;
	    gridDim.x = DIVIDE(size_n, EXL2_BLOCK_KN_SIZE * 4);
	    gridDim.y = DIVIDE(size_m, m_count);
	    gridDim.z = DIVIDE(size_k, EXL2_BLOCK_KN_SIZE);

	    fp_gemm_half_q_half_kernel kernel = pick_gemm_half_q_half_kernel(kernel_p, m_count);
		TORCH_CHECK(kernel != NULL);

	    kernel<<<gridDim, blockDim>>>
	    (
	        reinterpret_cast<half *>(x.data_ptr()),
	        reinterpret_cast<uint32_t *>(qweight.data_ptr()),
	        reinterpret_cast<half *>(out.data_ptr()),
	        reinterpret_cast<half *>(scales.data_ptr()),
	        reinterpret_cast<half *>(zeros.data_ptr()),
	        reinterpret_cast<uint16_t*>(qperm.data_ptr()),
	        reinterpret_cast<uint16_t*>(qgroup_map.data_ptr()),
	        size_m,
	        size_n,
	        size_k,
	        groups,
	        rows_8,
	        rows_6,
	        rows_5,
	        rows_4,
	        rows_3,
	        rows_2,
	        true,
	        NULL,
	        1
	    );
    }

	return out;
}