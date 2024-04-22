#ifndef _config_h
#define _config_h

#define MAX_Q_GEMM_ROWS 32
#define MAX_Q_GEMM_ROWS_KERNEL 4
#define MAX_Q_GEMM_WEIGHTS 4  // must be <= MAX_Q_GEMM_ROWS

/*
 The macros defined by QMODE_*BIT determine whether qweight should be rearranged. This rearrangement should,
 to some extent, enhance computational efficiency. Please note that in `quant/qdq_*.cuh`,
 two methods of dequantization are implemented for each bit level. When QMODE_*BIT=1,
 the rearrangement method `shuffle_*bit_*()` and the corresponding `dequant_*bit_*()` method are implemented.
 Therefore, it is important to note that if QMODE_*BIT=1, the qweight tensor needs to be rearranged by calling
 the `shuffle_*bit_*()` method.
*/
#define QMODE_2BIT 0
#define QMODE_3BIT 0
#define QMODE_4BIT 0
#define QMODE_5BIT 0
#define QMODE_6BIT 0
#define QMODE_8BIT 0

#endif
