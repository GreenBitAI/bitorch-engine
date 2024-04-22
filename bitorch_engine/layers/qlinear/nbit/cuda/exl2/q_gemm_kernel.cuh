#include "compat.cuh"
#include "config.h"
#include "matrix_view.cuh"
#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"
#include "quant/qdq_8.cuh"

#define EXL2_BLOCK_KN_SIZE 64
#define EXL2_BLOCK_M_SIZE_MAX MAX_Q_GEMM_ROWS_KERNEL
#define EXL2_MAX_GROUPS_IN_BLOCK (EXL2_BLOCK_KN_SIZE / 32)


__forceinline__ __device__ half dot22_8_h(half2(&dq)[4], const half* a_ptr, const half g_result, const half qs_h, const half qz_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    half2 qs_h2 = __halves2half2(qs_h, qs_h);
    half2 qz_h2 = __halves2half2(qz_h, qz_h);
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        half2 dq_2 = __hfma2(qs_h2, dq[i], __hneg2(qz_h2));
        result = __hfma2(dq_2, *a2_ptr++, result);
    }
    float result_f = __half2float(__hadd(__low2half(result), __high2half(result)));
    result_f += __half2float(g_result);
    return __float2half_rn(result_f);
}

__forceinline__ __device__ half dot22_16_h(half2(&dq)[8], const half* a_ptr, const half g_result, const half qs_h, const half qz_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    half2 qs_h2 = __halves2half2(qs_h, qs_h);
    half2 qz_h2 = __halves2half2(qz_h, qz_h);
    #pragma unroll
    for (int i = 0; i < 8; i++){
        half2 dq_2 = __hfma2(qs_h2, dq[i], __hneg2(qz_h2));
        result = __hfma2(dq_2, *a2_ptr++, result);
	}
    float result_f = __half2float(__hadd(__low2half(result), __high2half(result)));
    result_f += __half2float(g_result);
    return __float2half_rn(result_f);
}

__forceinline__ __device__ half dot22_32_h(half2(&dq)[16], const half* a_ptr, const half g_result, const half qs_h, const half qz_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    half2 qs_h2 = __halves2half2(qs_h, qs_h);
    half2 qz_h2 = __halves2half2(qz_h, qz_h);
    #pragma unroll
    for (int i = 0; i < 16; i += 1)
        result = __hfma2(__hfma2(dq[i], qs_h2, __hneg2(qz_h2)), *a2_ptr++, result);

    float result_f = __half2float(__hadd(__low2half(result), __high2half(result)));
    result_f += __half2float(g_result);
    return __float2half_rn(result_f);
}


typedef void (*fp_gemm_half_q_half_kernel)
(
    const half*,
    const uint32_t*,
    half*,
    const half*,
    const half*,
    const uint16_t*,
    const uint16_t*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const bool,
    const half*,
    const int
);


template <int m_count, int kernel_p, bool use_r_weights, bool mul_r_weights>
__global__ void gemm_half_q_half_kernel
(
    const half*      __restrict__ a,
    const uint32_t*  __restrict__ b_q_weight,
    half*            __restrict__ c,
    const half* __restrict__ b_scale,
    const half* __restrict__ b_zero,
    const uint16_t* __restrict__ b_q_perm,
    const uint16_t* __restrict__ b_q_group_map,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groups,
    const int rows_8,
    const int rows_6,
    const int rows_5,
    const int rows_4,
    const int rows_3,
    const int rows_2,
    const bool clear,
    const half* r_weights,
    const int r_weights_stride
){
    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half_rw c_(c, size_m, size_n);
    MatrixView_half b_scale_(b_scale, groups, size_n);
    MatrixView_half b_zero_(b_zero, groups, size_n);

    int t = threadIdx.x;

    // Block

    int offset_n = blockIdx.x * EXL2_BLOCK_KN_SIZE * 4;
    int offset_m = blockIdx.y * m_count;
    int offset_k = blockIdx.z * EXL2_BLOCK_KN_SIZE;

    int m_count_min = min(size_m - offset_m, m_count);

    int end_n = min(offset_n + EXL2_BLOCK_KN_SIZE * 4, size_n);
    int end_m = min(offset_m + m_count_min, size_m);
    int end_k = min(offset_k + EXL2_BLOCK_KN_SIZE, size_k);
    int n = offset_n + t * 4;

    // Read weights

    half_uint16 weights[MAX_Q_GEMM_WEIGHTS];
    if constexpr (use_r_weights)
    {
        uint16_t any_w = 0;
        const half* w_ptr = r_weights;
        for (int m = 0; m < m_count_min; ++m)
        {
            weights[m].as_half = *w_ptr;
            w_ptr += r_weights_stride;
            any_w |= weights[m].as_uint16;
        }
        if (!any_w) return;  // Early exit if all weights are zero -- does not zero output (!!!)
    }

    // Preload block_a

    __shared__ half block_a[m_count][EXL2_BLOCK_KN_SIZE];

    if (offset_k + t < end_k)
    {
        for (int m = 0; m < m_count_min; ++m)
        {
            const half* a_ptr = a_.item_ptr(offset_m + m, 0);
            half* block_a_ptr = block_a[m];
            half a0 = a_ptr[b_q_perm[offset_k + t]];
            block_a_ptr[t] = a0;
        }
    }

    // Clear

    if (n >= size_n) return;

    if (clear && blockIdx.z == 0) // && (threadIdx.x & 1) == 0)
    {
        for (int m = 0; m < m_count_min; m++)
            *((uint64_t*) c_.item_ptr(offset_m + m, n)) = 0;
    }

    __syncthreads();

    // Find initial group

    int group = b_q_group_map[offset_k * 2];

    // Preload scales

    half scales[EXL2_MAX_GROUPS_IN_BLOCK][4];
    half zeros[EXL2_MAX_GROUPS_IN_BLOCK][4];

    int temp_k = offset_k;
    for (int g = 0; temp_k < end_k; g++)
    {
        half qscales[4];
        half qzeros[4];

        b_scale_.item4(qscales, group + g, n);
        b_zero_.item4(qzeros, group + g, n);

        scales[g][0] = qscales[0];
        scales[g][1] = qscales[1];
        scales[g][2] = qscales[2];
        scales[g][3] = qscales[3];

        zeros[g][0] = qzeros[0];
        zeros[g][1] = qzeros[1];
        zeros[g][2] = qzeros[2];
        zeros[g][3] = qzeros[3];

        temp_k += b_q_group_map[temp_k * 2 + 1];
    }

    // a, b offset

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
    const half* a_ptr = &block_a[0][0];
    int a_stride = EXL2_BLOCK_KN_SIZE;

    // Initial group

    int scales_idx = 0;
    half qs_h0 = scales[scales_idx][0];
    half qs_h1 = scales[scales_idx][1];
    half qs_h2 = scales[scales_idx][2];
    half qs_h3 = scales[scales_idx][3];

    half qz_h0 = zeros[scales_idx][0];
    half qz_h1 = zeros[scales_idx][1];
    half qz_h2 = zeros[scales_idx][2];
    half qz_h3 = zeros[scales_idx][3];

    int nextgroup = offset_k + b_q_group_map[offset_k * 2 + 1];

    // Column result

    half block_c[m_count][4] = {};

    // Dequantize groups

    int k = offset_k;

    if constexpr (kernel_p & 0b10000000) {
    while (k < rows_8 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h0 = scales[scales_idx][0];
            qs_h1 = scales[scales_idx][1];
            qs_h2 = scales[scales_idx][2];
            qs_h3 = scales[scales_idx][3];

			qz_h0 = zeros[scales_idx][0];
			qz_h1 = zeros[scales_idx][1];
			qz_h2 = zeros[scales_idx][2];
			qz_h3 = zeros[scales_idx][3];
            nextgroup += b_q_group_map[k * 2 + 1];
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            uint4 load_int4[2];
            load_int4[0] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[1] = *((uint4*) b_ptr); b_ptr += size_n;

            half2 dq[4][4];
            dequant_8bit_8(load_int4[0].x, load_int4[1].x, dq[0], size_n);
            dequant_8bit_8(load_int4[0].y, load_int4[1].y, dq[1], size_n);
            dequant_8bit_8(load_int4[0].z, load_int4[1].z, dq[2], size_n);
            dequant_8bit_8(load_int4[0].w, load_int4[1].w, dq[3], size_n);

            for (int m = 0; m < m_count_min; m++)
            {
                if constexpr (use_r_weights) { if (!weights[m].as_uint16) continue; }

                block_c[m][0] = dot22_8_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0, qz_h0);
                block_c[m][1] = dot22_8_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1, qz_h1);
                block_c[m][2] = dot22_8_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2, qz_h2);
                block_c[m][3] = dot22_8_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3, qz_h3);
            }
            a_ptr += 8;
        }
        k += 32;
    }}

    if constexpr (kernel_p & 0b00100000) {
    while (k < rows_6 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h0 = scales[scales_idx][0];
            qs_h1 = scales[scales_idx][1];
            qs_h2 = scales[scales_idx][2];
            qs_h3 = scales[scales_idx][3];

			qz_h0 = zeros[scales_idx][0];
			qz_h1 = zeros[scales_idx][1];
			qz_h2 = zeros[scales_idx][2];
			qz_h3 = zeros[scales_idx][3];
            nextgroup += b_q_group_map[k * 2 + 1];
        }

        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            uint4 load_int4[3];
            load_int4[0] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[1] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[2] = *((uint4*) b_ptr); b_ptr += size_n;

            half2 dq[4][8];
            dequant_6bit_16(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0], size_n);
            dequant_6bit_16(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1], size_n);
            dequant_6bit_16(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2], size_n);
            dequant_6bit_16(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3], size_n);

            for (int m = 0; m < m_count_min; m++)
            {
                if constexpr (use_r_weights) { if (!weights[m].as_uint16) continue; }

                block_c[m][0] = dot22_16_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0, qz_h0);
                block_c[m][1] = dot22_16_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1, qz_h1);
                block_c[m][2] = dot22_16_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2, qz_h2);
                block_c[m][3] = dot22_16_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3, qz_h3);
            }
            a_ptr += 16;
        }
        k += 32;
    }}

    if constexpr (kernel_p & 0b00010000) {
    while (k < rows_5 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h0 = scales[scales_idx][0];
            qs_h1 = scales[scales_idx][1];
            qs_h2 = scales[scales_idx][2];
            qs_h3 = scales[scales_idx][3];

			qz_h0 = zeros[scales_idx][0];
			qz_h1 = zeros[scales_idx][1];
			qz_h2 = zeros[scales_idx][2];
			qz_h3 = zeros[scales_idx][3];
            nextgroup += b_q_group_map[k * 2 + 1];
        }

        #pragma unroll
        for (int j = 0; j < 1; j++)
        {
            uint4 load_int4[5];
            load_int4[0] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[1] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[2] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[3] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[4] = *((uint4*) b_ptr); b_ptr += size_n;

            half2 dq[4][16];
            dequant_5bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, load_int4[3].x, load_int4[4].x, dq[0], size_n);
            dequant_5bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, load_int4[3].y, load_int4[4].y, dq[1], size_n);
            dequant_5bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, load_int4[3].z, load_int4[4].z, dq[2], size_n);
            dequant_5bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, load_int4[3].w, load_int4[4].w, dq[3], size_n);

            for (int m = 0; m < m_count_min; m++)
            {
                if constexpr (use_r_weights) { if (!weights[m].as_uint16) continue; }

                block_c[m][0] = dot22_32_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0, qz_h0);
                block_c[m][1] = dot22_32_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1, qz_h1);
                block_c[m][2] = dot22_32_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2, qz_h2);
                block_c[m][3] = dot22_32_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3, qz_h3);
            }
            a_ptr += 32;
        }

        k += 32;
    }}

    if constexpr (kernel_p & 0b00001000) {
    while (k < rows_4 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h0 = scales[scales_idx][0];
            qs_h1 = scales[scales_idx][1];
            qs_h2 = scales[scales_idx][2];
            qs_h3 = scales[scales_idx][3];

			qz_h0 = zeros[scales_idx][0];
			qz_h1 = zeros[scales_idx][1];
			qz_h2 = zeros[scales_idx][2];
			qz_h3 = zeros[scales_idx][3];
            nextgroup += b_q_group_map[k * 2 + 1];
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            uint4 load_int4[1];
            load_int4[0] = *((uint4*) b_ptr); b_ptr += size_n;

            half2 dq[4][4];
            dequant_4bit_8(load_int4[0].x, dq[0], size_n);
            dequant_4bit_8(load_int4[0].y, dq[1], size_n);
            dequant_4bit_8(load_int4[0].z, dq[2], size_n);
            dequant_4bit_8(load_int4[0].w, dq[3], size_n);

            for (int m = 0; m < m_count_min; m++)
            {
                if constexpr (use_r_weights) { if (!weights[m].as_uint16) continue; }

                block_c[m][0] = dot22_8_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0, qz_h0);
                block_c[m][1] = dot22_8_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1, qz_h1);
                block_c[m][2] = dot22_8_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2, qz_h2);
                block_c[m][3] = dot22_8_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3, qz_h3);
            }
            a_ptr += 8;
        }
        k += 32;
    }}

    if constexpr (kernel_p & 0b00000100) {
    while (k < rows_3 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h0 = scales[scales_idx][0];
            qs_h1 = scales[scales_idx][1];
            qs_h2 = scales[scales_idx][2];
            qs_h3 = scales[scales_idx][3];

			qz_h0 = zeros[scales_idx][0];
			qz_h1 = zeros[scales_idx][1];
			qz_h2 = zeros[scales_idx][2];
			qz_h3 = zeros[scales_idx][3];
            nextgroup += b_q_group_map[k * 2 + 1];
        }

        #pragma unroll
        for (int j = 0; j < 1; j++)
        {
            uint4 load_int4[3];
            load_int4[0] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[1] = *((uint4*) b_ptr); b_ptr += size_n;
            load_int4[2] = *((uint4*) b_ptr); b_ptr += size_n;

            half2 dq[4][16];
            dequant_3bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0], size_n);
            dequant_3bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1], size_n);
            dequant_3bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2], size_n);
            dequant_3bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3], size_n);

            for (int m = 0; m < m_count_min; m++)
            {
                if constexpr (use_r_weights) { if (!weights[m].as_uint16) continue; }

                block_c[m][0] = dot22_32_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0, qz_h0);
                block_c[m][1] = dot22_32_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1, qz_h1);
                block_c[m][2] = dot22_32_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2, qz_h2);
                block_c[m][3] = dot22_32_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3, qz_h3);
            }
            a_ptr += 32;
        }
        k += 32;
    }}

    if constexpr (kernel_p & 0b00000010) {
    while (k < rows_2 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h0 = scales[scales_idx][0];
            qs_h1 = scales[scales_idx][1];
            qs_h2 = scales[scales_idx][2];
            qs_h3 = scales[scales_idx][3];

			qz_h0 = zeros[scales_idx][0];
			qz_h1 = zeros[scales_idx][1];
			qz_h2 = zeros[scales_idx][2];
			qz_h3 = zeros[scales_idx][3];
            nextgroup += b_q_group_map[k * 2 + 1];
        }

        #pragma unroll
        for (int j = 0; j < 1; j++)
        {
            uint4 load_int4[1];
            load_int4[0] = *((uint4*) b_ptr); b_ptr += size_n;

            half2 dq[4][8];
            dequant_2bit_16(load_int4[0].x, dq[0], size_n);
            dequant_2bit_16(load_int4[0].y, dq[1], size_n);
            dequant_2bit_16(load_int4[0].z, dq[2], size_n);
            dequant_2bit_16(load_int4[0].w, dq[3], size_n);

            for (int m = 0; m < m_count_min; m++)
            {
                if constexpr (use_r_weights) { if (!weights[m].as_uint16) continue; }
                block_c[m][0] = dot22_16_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0, qz_h0);
                block_c[m][1] = dot22_16_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1, qz_h1);
                block_c[m][2] = dot22_16_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2, qz_h2);
                block_c[m][3] = dot22_16_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3, qz_h3);
            }

            a_ptr += 16;
        }
        k += 16;
    }}

    // Accumulate column sums in c

    for (int m = 0; m < m_count_min; m++)
    {
        half2* out = (half2*)c_.item_ptr(offset_m + m, n);
        half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
        half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);

        if constexpr (mul_r_weights)
        {
            half2 w_mul2 = __half2half2(weights[m].as_half);
            result01 = __hmul2(result01, w_mul2);
            result23 = __hmul2(result23, w_mul2);
        }

        atomicAdd(out    , result01);
        atomicAdd(out + 1, result23);
    }
}

