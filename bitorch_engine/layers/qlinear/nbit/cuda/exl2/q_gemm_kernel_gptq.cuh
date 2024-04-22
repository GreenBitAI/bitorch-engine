#include "compat.cuh"
#include "config.h"
#include "matrix_view.cuh"
#include "quant/qdq_4.cuh"

#define GPTQ_BLOCK_KN_SIZE 128
#define GPTQ_BLOCK_M_SIZE_MAX MAX_Q_GEMM_ROWS_KERNEL
#define GPTQ_MAX_GROUPS_IN_BLOCK (GPTQ_BLOCK_KN_SIZE / 32)


__forceinline__ __device__ half2 dot22_8_h2(half2(&dq)[4], const half* a_ptr, const half2 scale, const half2 zero)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 4; i++){
        result = __hfma2(__hfma2(scale, dq[i], __hneg2(zero)), *a2_ptr++, result);
    }
    return result;
}

__forceinline__ __device__ half2 dot22_16_h2(half2(&dq)[8], const half* a_ptr, const half2 scale, const half2 zero)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 8; i++){
        half2 dq_2 = __hfma2(scale, dq[i], __hneg2(zero));
        result = __hfma2(dq_2, *a2_ptr++, result);
    }
    return result;
}


template <int m_count>
__global__ void gemm_half_q4_half_gptq_kernel(
    const half* __restrict__ a,
    const uint32_t* __restrict__ b_q_weight,
    const half* __restrict__ b_gptq_zeros,
    const half* __restrict__ b_gptq_scales,
    half* __restrict__ c,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groups,
    const int groupsize,
    const bool clear,
    const uint16_t* __restrict__ b_q_perm
){
    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half_rw c_(c, size_m, size_n);
    MatrixView_half b_gptq_zeros_(b_gptq_zeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

    int t = threadIdx.x;

    // Block

    int offset_n = blockIdx.x * GPTQ_BLOCK_KN_SIZE * 4;
    int offset_m = blockIdx.y * m_count;
    int offset_k = blockIdx.z * GPTQ_BLOCK_KN_SIZE;

    int m_count_min = min(size_m - offset_m, m_count);

    int end_n = min(offset_n + GPTQ_BLOCK_KN_SIZE * 4, size_n);
    int end_m = min(offset_m + m_count_min, size_m);
    int end_k = min(offset_k + GPTQ_BLOCK_KN_SIZE, size_k);

    int n = offset_n + t * 4;

    // Preload block_a

    __shared__ half block_a[m_count][GPTQ_BLOCK_KN_SIZE];

    if (offset_k + t < end_k)
    {
        for (int m = 0; m < m_count_min; ++m)
        {
            const half* a_ptr = a_.item_ptr(offset_m + m, 0);
            half* block_a_ptr = block_a[m];

            half a0;
            if (b_q_perm)
                a0 = a_ptr[b_q_perm[offset_k + t]];
            else
                a0 = a_ptr[offset_k + t];
            block_a_ptr[t] = a0;
        }
    }

    // Zero output

    if (n >= size_n) return;

    if (clear && blockIdx.z == 0) // && (threadIdx.x & 1) == 0)
    {
        for (int m = 0; m < m_count_min; m++)
            *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
    }

    __syncthreads();

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // a, b offset

    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
    const half* a_ptr = &block_a[0][0];
    int a_stride = GPTQ_BLOCK_KN_SIZE;

    // Initial group

    half2 zeros[4];
    half2 scales[4];
    b_gptq_zeros_.item4_h2(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);


//    __syncthreads();

    // Column result

    half2 block_c[m_count][4] = {};

    // Dequantize and multiply

    int k = offset_k;
    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_zeros_.item4_h2(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            half2 dq[4][4];
            const int4* b_ptr4 = (int4*) b_ptr;
            int4 load_int4 = *b_ptr4;

            dequant_4bit_8(load_int4.x, dq[0]);
            dequant_4bit_8(load_int4.y, dq[1]);
            dequant_4bit_8(load_int4.z, dq[2]);
            dequant_4bit_8(load_int4.w, dq[3]);

            #pragma unroll
            for (int m = 0; m < m_count_min; m++)
            {
                block_c[m][0] = __hadd2(dot22_8_h2(dq[0], a_ptr + m * a_stride, scales[0], zeros[0]), block_c[m][0]);
                block_c[m][1] = __hadd2(dot22_8_h2(dq[1], a_ptr + m * a_stride, scales[1], zeros[1]), block_c[m][1]);
                block_c[m][2] = __hadd2(dot22_8_h2(dq[2], a_ptr + m * a_stride, scales[2], zeros[2]), block_c[m][2]);
                block_c[m][3] = __hadd2(dot22_8_h2(dq[3], a_ptr + m * a_stride, scales[3], zeros[3]), block_c[m][3]);
            }

            b_ptr += size_n;
            a_ptr += 8;
        }

        k += 32;
    }

    for (int m = 0; m < m_count_min; m++)
    {
        half2 *out = (half2*) c_.item_ptr(offset_m + m, n);
        half result0 = __hadd(__low2half(block_c[m][0]), __high2half(block_c[m][0]));
        half result1 = __hadd(__low2half(block_c[m][1]), __high2half(block_c[m][1]));
        half result2 = __hadd(__low2half(block_c[m][2]), __high2half(block_c[m][2]));
        half result3 = __hadd(__low2half(block_c[m][3]), __high2half(block_c[m][3]));
        half2 result01 = __halves2half2(result0, result1);
        half2 result23 = __halves2half2(result2, result3);

        atomicAdd(out    , result01);
        atomicAdd(out + 1, result23);
    }
}


template <int m_count>
__global__ void gemm_half_q2_half_gptq_kernel(
    const half* __restrict__ a,
    const uint32_t* __restrict__ b_q_weight,
    const half* __restrict__ b_gptq_zeros,
    const half* __restrict__ b_gptq_scales,
    half* __restrict__ c,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groups,
    const int groupsize,
    const bool clear,
    const uint16_t* __restrict__ b_q_perm
){
    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half_rw c_(c, size_m, size_n);
    MatrixView_half b_gptq_zeros_(b_gptq_zeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

    int t = threadIdx.x;

    // Block

    int offset_n = blockIdx.x * GPTQ_BLOCK_KN_SIZE * 4;
    int offset_m = blockIdx.y * m_count;
    int offset_k = blockIdx.z * GPTQ_BLOCK_KN_SIZE;

    int m_count_min = min(size_m - offset_m, m_count);

    int end_n = min(offset_n + GPTQ_BLOCK_KN_SIZE * 4, size_n);
    int end_m = min(offset_m + m_count_min, size_m);
    int end_k = min(offset_k + GPTQ_BLOCK_KN_SIZE, size_k);

    int n = offset_n + t * 4;

    // Preload block_a

    __shared__ half block_a[m_count][GPTQ_BLOCK_KN_SIZE];

    if (offset_k + t < end_k)
    {
        for (int m = 0; m < m_count_min; ++m)
        {
            const half* a_ptr = a_.item_ptr(offset_m + m, 0);
            half* block_a_ptr = block_a[m];

            half a0;
            if (b_q_perm)
                a0 = a_ptr[b_q_perm[offset_k + t]];
            else
                a0 = a_ptr[offset_k + t];
            block_a_ptr[t] = a0;
        }
    }

    // Zero output

    if (n >= size_n) return;

    if (clear && blockIdx.z == 0) // && (threadIdx.x & 1) == 0)
    {
        for (int m = 0; m < m_count_min; m++)
            *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
    }

    __syncthreads();

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // a, b offset

    const uint32_t* b_ptr = b_q_weight + (offset_k / (32 / 2)) * size_n + n;
    const half* a_ptr = &block_a[0][0];
    int a_stride = GPTQ_BLOCK_KN_SIZE;

    // Initial group

    half2 zeros[4];
    half2 scales[4];
    b_gptq_zeros_.item4_h2(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);

    // Column result

    half2 block_c[m_count][4] = {};

    // Dequantize and multiply

    int k = offset_k;

    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_zeros_.item4_h2(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
        }

        for (int p = 0; p < 1; p++)
        {
            int4 load_int4[1];
            load_int4[0] = *((int4*) b_ptr);
            b_ptr += size_n;

            half2 dq[4][8];
            dequant_2bit_16(load_int4[0].x, dq[0], size_n);
            dequant_2bit_16(load_int4[0].y, dq[1], size_n);
            dequant_2bit_16(load_int4[0].z, dq[2], size_n);
            dequant_2bit_16(load_int4[0].w, dq[3], size_n);

            for (int m = 0; m < m_count_min; m++)
            {
                block_c[m][0] = __hadd2(dot22_16_h2(dq[0], a_ptr + m * a_stride, scales[0], zeros[0]), block_c[m][0]);
                block_c[m][1] = __hadd2(dot22_16_h2(dq[1], a_ptr + m * a_stride, scales[1], zeros[1]), block_c[m][1]);
                block_c[m][2] = __hadd2(dot22_16_h2(dq[2], a_ptr + m * a_stride, scales[2], zeros[2]), block_c[m][2]);
                block_c[m][3] = __hadd2(dot22_16_h2(dq[3], a_ptr + m * a_stride, scales[3], zeros[3]), block_c[m][3]);
            }
            a_ptr += 16;
        }
        k += 16;
    }

    for (int m = 0; m < m_count_min; m++)
    {
        half2 *out = (half2*) c_.item_ptr(offset_m + m, n);
        half result0 = __hadd(__low2half(block_c[m][0]), __high2half(block_c[m][0]));
        half result1 = __hadd(__low2half(block_c[m][1]), __high2half(block_c[m][1]));
        half result2 = __hadd(__low2half(block_c[m][2]), __high2half(block_c[m][2]));
        half result3 = __hadd(__low2half(block_c[m][3]), __high2half(block_c[m][3]));
        half2 result01 = __halves2half2(result0, result1);
        half2 result23 = __halves2half2(result2, result3);

        atomicAdd(out    , result01);
        atomicAdd(out + 1, result23);
    }
}

