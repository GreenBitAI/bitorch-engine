#ifndef _kernel_select_cuh
#define _kernel_select_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q_gemm_kernel.cuh"

inline fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int perm, const int max_m)
{
    switch (perm)
    {
        //exl2_a
        case 0b00000010:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00000010, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00000010, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00000010, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00000010, false, false>;
        case 0b00000110:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00000110, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00000110, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00000110, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00000110, false, false>;
        case 0b00000100:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00000100, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00000100, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00000100, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00000100, false, false>;
        case 0b00001110:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00001110, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00001110, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00001110, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00001110, false, false>;
        case 0b00001100:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00001100, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00001100, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00001100, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00001100, false, false>;
        case 0b00001000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00001000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00001000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00001000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00001000, false, false>;
        case 0b00011000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00011000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00011000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00011000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00011000, false, false>;
        case 0b00010000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00010000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00010000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00010000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00010000, false, false>;
        case 0b00110000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00110000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00110000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00110000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00110000, false, false>;
        case 0b00100000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00100000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00100000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00100000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00100000, false, false>;
        //exl2_b
        case 0b10000000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b10000000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b10000000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b10000000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b10000000, false, false>;
        case 0b00100110:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00100110, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00100110, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00100110, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00100110, false, false>;
        case 0b00010100:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00010100, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00010100, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00010100, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00010100, false, false>;
        case 0b10001100:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b10001100, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b10001100, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b10001100, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b10001100, false, false>;
        case 0b10011000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b10011000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b10011000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b10011000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b10011000, false, false>;
        case 0b10110000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b10110000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b10110000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b10110000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b10110000, false, false>;
        case 0b10101100:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b10101100, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b10101100, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b10101100, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b10101100, false, false>;
        case 0b00001010:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00001010, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00001010, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00001010, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00001010, false, false>;
        case 0b00101000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b00101000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b00101000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b00101000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b00101000, false, false>;
        case 0b10100000:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b10100000, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b10100000, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b10100000, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b10100000, false, false>;
        case 0b10111110:
            if (max_m == 1) return gemm_half_q_half_kernel<1, 0b10111110, false, false>;
            if (max_m == 2) return gemm_half_q_half_kernel<2, 0b10111110, false, false>;
            if (max_m == 3) return gemm_half_q_half_kernel<3, 0b10111110, false, false>;
            if (max_m == 4) return gemm_half_q_half_kernel<4, 0b10111110, false, false>;
        default:
            printf("ERROR: No exl2-kernel found for permutation %x\n", perm);
            return NULL;
    }
}

#endif