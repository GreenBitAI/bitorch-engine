#include <torch/extension.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>

// Macro to set a specific bit in a variable.
// It shifts the 'val' left by 'pos' positions and performs an OR operation with 'var' to set the bit.
// 'var': The variable whose bit is to be set.
// 'pos': The position of the bit to be set (0-based).
// 'val': The value to set at the specified position (typically 1 to set or 0 to clear the bit, but using bitwise OR here implies setting only).
#define BIT_SET(var, pos, val) var |= (val << pos)

// Type definition for a binary word using an 8-bit unsigned integer to represent binary data efficiently.
typedef uint8_t BINARY_WORD;

// It calculates the size of BINARY_WORD in bytes, multiplies by CHAR_BIT (the number of bits per byte, typically 8),
// to determine the total number of bits available in a BINARY_WORD.
const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);

// A constant defined for loop unrolling in algorithms.
// 'UNROLLN' specifies the factor of loop unrolling used to optimize loops in binary operations or processing,
// aiming to enhance performance by reducing loop overhead and increasing instruction-level parallelism.
#define UNROLLN 6


/**
 * @brief Converts a floating-point array to a binary representation.
 *
 * This function takes an array of floating-point numbers (row) and converts it
 * into a binary format stored in b_row. Each bit in the binary representation
 * corresponds to the sign of the floating-point number in the original array:
 * a bit is set to 1 if the number is non-negative, and 0 otherwise. The size
 * parameter specifies the number of elements in the input array. The conversion
 * processes BITS_PER_BINARY_WORD floating-point numbers at a time, packing their
 * binary sign representation into a single BINARY_WORD. This process is
 * parallelized using OpenMP to improve performance.
 *
 * @param row Pointer to the input array of floating-point numbers.
 * @param b_row Pointer to the output array where the binary representation is stored.
 * @param size The number of elements in the input array.
 */
void _get_binary_row(float* row, BINARY_WORD * b_row, int size){
    #pragma omp parallel for
    for (int i = 0; i < size; i+=BITS_PER_BINARY_WORD) {
		BINARY_WORD rvalue=0;
		BINARY_WORD sign;
		for (int j = 0;j < BITS_PER_BINARY_WORD; ++j) {
			sign = (row[i+j]>=0);
			BIT_SET(rvalue, j, sign);
		}
		b_row[i/BITS_PER_BINARY_WORD] = rvalue;
    }
}


/**
 * Binarizes a matrix column-wise with loop unrolling and register variables optimization.
 * Achieves approximately 30% performance improvement over the get_binary_col() method without using OpenMP.
 * This function optimizes the process of converting floating-point matrix columns into binary columns,
 * significantly speeding up operations that rely on binary data representations, especially in machine learning contexts.
 *
 * @param col Pointer to the input matrix stored in row-major order. This matrix is of size n x k, where
 *            n is the number of rows (height) and k is the number of columns (width).
 * @param b_col Pointer to the output binary matrix. This matrix is represented as an array of BINARY_WORD,
 *              where each word stores BITS_PER_BINARY_WORD bits, effectively compressing the binary representation.
 * @param n The number of rows in the input matrix. It is assumed that n is divisible by BITS_PER_BINARY_WORD
 *          for proper packing into the binary format.
 * @param k The number of columns in the input matrix. The function processes k columns in groups of 4
 *          for efficiency, leveraging SIMD-like optimizations manually.
 *
 * Note: This function uses OpenMP for parallel processing of columns, further enhancing performance. It's designed
 *       to work efficiently with compilers and environments that support OpenMP. For environments without OpenMP,
 *       the parallel directive will be ignored, and the function will still operate correctly (albeit with potential
 *       performance differences).
 *
 * Warning: The usage of register keyword is deprecated in C++17 and has no effect. The comments mentioning
 *          register variables are kept for historical context and understanding the original optimization intent.
 */
void _get_binary_col_unrolled(float* col, BINARY_WORD * b_col, int n, int k){
    for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
		BINARY_WORD * y_col_pt = &b_col[y*k];
		#pragma omp parallel for
		for(int x=0; x < k; x+=4){
			// todo: register removed for c++17 there should be no difference?
			BINARY_WORD rvalue0=0, rvalue1=0, rvalue2=0, rvalue3=0;

			for(int b=0; b<BITS_PER_BINARY_WORD; b+=4){
				// todo: register removed for c++17 there should be no difference?
				BINARY_WORD sign0, sign1, sign2, sign3, sign4, sign5, sign6, sign7,
				sign8, sign9, sign10, sign11, sign12, sign13, sign14, sign15;

				float* col_0 = &col[(y*BITS_PER_BINARY_WORD+b)*k + x];
				float* col_1 = &col[(y*BITS_PER_BINARY_WORD+b+1)*k + x];
				float* col_2 = &col[(y*BITS_PER_BINARY_WORD+b+2)*k + x];
				float* col_3 = &col[(y*BITS_PER_BINARY_WORD+b+3)*k + x];

				sign0 = (*col_0>=0);
				sign1 = (*col_1>=0);
				sign2 = (*col_2>=0);
				sign3 = (*col_3>=0);

				BIT_SET(rvalue0, b, sign0);
				BIT_SET(rvalue0, (b+1), sign1);
				BIT_SET(rvalue0, (b+2), sign2);
				BIT_SET(rvalue0, (b+3), sign3);

				sign4 = (*(col_0+1)>=0);
				sign5 = (*(col_1+1)>=0);
				sign6 = (*(col_2+1)>=0);
				sign7 = (*(col_3+1)>=0);

				BIT_SET(rvalue1, b, sign4);
				BIT_SET(rvalue1, (b+1), sign5);
				BIT_SET(rvalue1, (b+2), sign6);
				BIT_SET(rvalue1, (b+3), sign7);

				sign8 = (*(col_0+2)>=0);
				sign9 = (*(col_1+2)>=0);
				sign10 = (*(col_2+2)>=0);
				sign11 = (*(col_3+2)>=0);

				BIT_SET(rvalue2, b, sign8);
				BIT_SET(rvalue2, (b+1), sign9);
				BIT_SET(rvalue2, (b+2), sign10);
				BIT_SET(rvalue2, (b+3), sign11);

				sign12 = (*(col_0+3)>=0);
				sign13 = (*(col_1+3)>=0);
				sign14 = (*(col_2+3)>=0);
				sign15 = (*(col_3+3)>=0);

				BIT_SET(rvalue3, b, sign12);
				BIT_SET(rvalue3, (b+1), sign13);
				BIT_SET(rvalue3, (b+2), sign14);
				BIT_SET(rvalue3, (b+3), sign15);
			}
			BINARY_WORD * pnter = &y_col_pt[x];
			*pnter = rvalue0;
			*(pnter+1) = rvalue1;
			*(pnter+2) = rvalue2;
			*(pnter+3) = rvalue3;
		}
    }
}


/**
 * Performs a matrix multiplication using the XNOR and popcount operations, specifically optimized for binary neural network computations.
 * This function is deprecated and was designed for experimentation with binary neural networks.
 * It unrolls the inner loop for improved performance with a fixed unroll factor (UNROLLN).
 *
 * Parameters:
 * - M: The number of rows in matrix A and the resulting matrix C.
 * - N: The number of columns in matrix B and the resulting matrix C.
 * - K: The number of columns in matrix A and rows in matrix B.
 * - A: Pointer to the first matrix (binary) of size MxK.
 * - lda: Leading dimension of matrix A, typically set to K.
 * - B: Pointer to the second matrix (binary) of size KxN.
 * - ldb: Leading dimension of matrix B, typically set to N.
 * - C: Pointer to the output matrix of size MxN. This matrix is in floating-point format as it accumulates popcount results.
 * - ldc: Leading dimension of matrix C, typically set to N.
 *
 * The function computes the matrix product of A and B, treating A and B as binary matrices.
 * It XORs corresponding bits of A and B, then counts the number of set bits (popcount).
 * The result is scaled and added to elements of C. The computation is done in chunks using loop unrolling for efficiency.
 * Finally, it converts the accumulation in C to a (+1, -1) representation by scaling the popcount results.
 *
 * Note: This function is marked as deprecated and is intended for experimental use only.
 * It assumes that binary matrices are represented using BINARY_WORD data type,
 * and the computation is parallelized using OpenMP for improved performance on multi-core processors.
 */
void _xnor_gemm_unrolled_deprecated(int M, int N, int K,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc){
	int m,k,n;
	#pragma omp parallel for
	for (m = 0; m < M; ++m) {
		#pragma omp parallel for
		for (k = 0; k < ((K / UNROLLN) * UNROLLN); k+=UNROLLN) {
			BINARY_WORD A_PART[UNROLLN];
			A_PART[0] = A[m*lda+k];
			A_PART[1] = A[m*lda+k+1];
			A_PART[2] = A[m*lda+k+2];
			A_PART[3] = A[m*lda+k+3];
			A_PART[4] = A[m*lda+k+4];
			A_PART[5] = A[m*lda+k+5];
			#pragma omp parallel for
			for (n = 0; n < N; ++n) {
				int popc[UNROLLN];
				popc[0] = __builtin_popcountl(A_PART[0] ^ B[(k+0)*ldb+n]);
				popc[1] = __builtin_popcountl(A_PART[1] ^ B[(k+1)*ldb+n]);
				popc[2] = __builtin_popcountl(A_PART[2] ^ B[(k+2)*ldb+n]);
				popc[3] = __builtin_popcountl(A_PART[3] ^ B[(k+3)*ldb+n]);
				popc[4] = __builtin_popcountl(A_PART[4] ^ B[(k+4)*ldb+n]);
				popc[5] = __builtin_popcountl(A_PART[5] ^ B[(k+5)*ldb+n]);
				C[m*ldc+n] += popc[0] + popc[1] + popc[2] + popc[3] + popc[4] + popc[5];
			}
		}

		#pragma omp parallel for
		for (k=(K / UNROLLN) * UNROLLN; k < K; ++k) {
			BINARY_WORD A_PART = A[m*lda+k];
			#pragma omp parallel for
			for (n = 0; n < N; ++n) {
				C[m * ldc + n] += __builtin_popcountl(A_PART ^ B[k * ldb + n]);
			}
		}
	}

	// convert to (+1,-1) based presentation form
	#pragma omp parallel for
	for (int i=0; i < M*N; i++) {
		C[i] = -(2*C[i] - BITS_PER_BINARY_WORD*K);
	}
}


/**
 * Performs matrix multiplication using the XNOR and popcount operations, optimized with loop unrolling.
 * This function specifically targets binary neural network operations, where the inputs are binary matrices,
 * and the multiplication is performed using XNOR followed by popcount to count the number of set bits.
 * The result is a matrix of floating-point numbers representing the accumulation of set bits.
 * This approach is particularly useful for binary neural networks, offering significant speedup
 * compared to traditional floating-point matrix multiplication.
 *
 * @param M The number of rows in matrix A and the resulting matrix C.
 * @param N The number of columns in matrix B and the resulting matrix C.
 * @param K The number of columns in matrix A and rows in matrix B.
 * @param A Pointer to the first matrix (binary) of size MxK.
 * @param lda Leading dimension of matrix A, typically set to K.
 * @param B Pointer to the second matrix (binary) of size KxN.
 * @param ldb Leading dimension of matrix B, typically set to N.
 * @param C Pointer to the result matrix (float) of size MxN.
 * @param ldc Leading dimension of matrix C, typically set to N.
 *
 * The function utilizes loop unrolling for efficiency, processing multiple elements in a single iteration
 * to reduce loop overhead and improve instruction-level parallelism. The outermost loop over M is parallelized
 * using OpenMP to exploit multi-core parallelism, enhancing performance on multi-threaded systems.
 * The inner loops over K and N are manually unrolled (as indicated by UNROLLN), allowing the compiler to
 * optimize memory access patterns and register usage.
 *
 * After computing the XNOR and popcount, the result is adjusted to represent the count in terms of (+1, -1)
 * instead of the binary (0, 1), as is common in binary neural network operations. This final adjustment
 * is performed in parallel for each element of the output matrix C, reflecting the aggregated bit similarity
 * across the input matrices.
 */
void _xnor_gemm_unrolled(int M, int N, int K,
                         BINARY_WORD *A, int lda,
                         BINARY_WORD *B, int ldb,
                         float *C, int ldc) {
    int m, k, n;

    // The loop for m should be parallelized, not each inner loop.
#pragma omp parallel for private(m, k, n)
    for (m = 0; m < M; ++m) {
        int m_ldc = m * ldc;
        int m_lda = m * lda;

        // Loop unrolling for k.
        for (k = 0; k < ((K / UNROLLN) * UNROLLN); k += UNROLLN) {
            BINARY_WORD A_PART[UNROLLN];
            for (int u = 0; u < UNROLLN; ++u) {
                A_PART[u] = A[m_lda + k + u];
            }

            // Loop unrolling for n.
            for (n = 0; n < N; ++n) {
                int popc[UNROLLN];
                for (int u = 0; u < UNROLLN; ++u) {
                    popc[u] = __builtin_popcountll(A_PART[u] ^ B[(k + u) * ldb + n]);
                }

                // Calculate the sum of popcounts.
                int popc_sum = popc[0] + popc[1] + popc[2] + popc[3] + popc[4] + popc[5];
                C[m_ldc + n] += popc_sum;
            }
        }

        // Handle the remaining k.
        for (k = (K / UNROLLN) * UNROLLN; k < K; ++k) {
            BINARY_WORD A_PART = A[m_lda + k];
            for (n = 0; n < N; ++n) {
                C[m_ldc + n] += __builtin_popcountll(A_PART ^ B[k * ldb + n]);
            }
        }
    }

    // Convert to (+1,-1) based presentation form
    #pragma omp parallel for
    for (int i = 0; i < M * N; i++) {
        C[i] = -(2 * C[i] - BITS_PER_BINARY_WORD * K);
    }
}


/**
 * Performs a binary linear forward pass using unpacked weights.
 *
 * This function executes a binary version of a linear (fully connected) layer's forward pass
 * with binary inputs represented by BINARY_WORD pointers and produces a standard floating-point output tensor.
 * The weights are initially in a standard floating-point format and are packed into a binary format
 * internally to perform the binary operations.
 *
 * @param weights The floating-point weight matrix of the linear layer, with shape [k, n].
 * @param binary_row Pointer to the binary representation of the input rows, assumed to be packed.
 * @param m Number of rows in the input binary matrix (batch size).
 * @param n Number of columns in the output matrix (number of output features).
 * @param k Number of columns in the input binary matrix (number of input features).
 * @return A torch::Tensor containing the floating-point output of the binary linear layer.
 *
 * The function first transposes and ensures the weight tensor is contiguous for binary packing.
 * It then performs the binary packing of the weights into a format suitable for binary matrix multiplication.
 * Finally, it executes the binary multiplication using an XNOR-based General Matrix Multiply (GEMM) algorithm
 * optimized for binary inputs, producing the floating-point output tensor.
 */
torch::Tensor binary_linear_forward_unpacked_weight(
    torch::Tensor weights,
    BINARY_WORD* binary_row,
    int m,
    int n,
    int k) {

	auto option_quantize = torch::TensorOptions().dtype(torch::kUInt8).device(weights.device());
    auto output = torch::zeros({m, n});
    auto w_packed = torch::zeros({k*n/BITS_PER_BINARY_WORD}, option_quantize);
    BINARY_WORD* binary_col = (BINARY_WORD*)(w_packed.data_ptr());
    // NOTE: .contiguous() will physically reorder the tensor layout according to the C style,
    // Ensure weights tensor is contiguous in memory after transposing, for correct binary packing.
    weights = weights.transpose(0,1).contiguous();
    _get_binary_col_unrolled(weights.data_ptr<float>(), binary_col, k, n);

    _xnor_gemm_unrolled(m, n, k/BITS_PER_BINARY_WORD,
          binary_row, k/BITS_PER_BINARY_WORD,
          binary_col, n,
          output.data_ptr<float>(), n);
    return output;
}


/**
 * Performs a forward pass using a binary linear layer with binarized weights.
 * This function leverages the efficiency of XNOR GEMM operations for binary neural networks.
 *
 * @param weights A torch::Tensor containing the pre-packed binary weights of the layer.
 *                The weights tensor should be in a compatible format for binary operations.
 * @param binary_row A pointer to a BINARY_WORD array representing the input data after
 *                   being processed into binary form. This array should have the data arranged
 *                   in a format suitable for the binary GEMM operation.
 * @param m The number of rows in the output matrix, typically representing the batch size.
 * @param n The number of columns in the output matrix, usually corresponding to the number
 *          of output features or neurons in the layer.
 * @param k The size of the second dimension of the input (binary_row) and the first
 *          dimension of the weights, representing the number of input features.
 *
 * This function computes the matrix multiplication of the binarized input data (binary_row)
 * with the binarized weights (weights), producing a dense output tensor. The binary matrix
 * multiplication is performed using an optimized XNOR GEMM operation, which is unrolled
 * for efficiency. The result is a dense tensor representing the layer's output before any
 * activation function is applied.
 *
 * The function assumes that the weights are already binarized and packed into BINARY_WORD
 * format, with each bit representing a binary weight. The input data must also be pre-processed
 * into a similar binary packed format. The dimensions of the input and weights should match
 * the requirements of the binary GEMM operation, with adjustments for the bits per binary word.
 *
 * @return A torch::Tensor representing the output of the binary linear layer. The output tensor
 *         is dense and contains floating point values, which can be further processed by subsequent
 *         layers or activation functions.
 */
torch::Tensor binary_linear_forward_binarized_weight(
    torch::Tensor weights,
    BINARY_WORD* binary_row,
    int m,
    int n,
    int k) {

    auto output = torch::zeros({m, n});
    BINARY_WORD* binary_col = (BINARY_WORD*) weights.data_ptr();

    _xnor_gemm_unrolled(m, n, k/BITS_PER_BINARY_WORD,
          binary_row, k/BITS_PER_BINARY_WORD,
          binary_col, n,
          output.data_ptr<float>(), n);

    return output;
}


/**
 * Retrieves a binary row from a packed tensor.
 *
 * This function converts a row of a given input tensor into its binary representation
 * using a specified packed tensor. It leverages direct memory access to efficiently
 * retrieve and convert the tensor data into a binary format suitable for further processing
 * or computation in binary neural networks or other binary operations.
 *
 * @param input A torch::Tensor representing the input data. The tensor should hold
 *              floating-point values that are to be accessed and processed.
 * @param a_packed A torch::Tensor that is already packed in a binary format (BINARY_WORD).
 *                 This tensor acts as the source or basis for the binary conversion process.
 * @param m The number of rows in the input tensor.
 * @param n The number of columns in the output binary row. This typically corresponds
 *          to the dimensionality of the binary representation.
 * @param k The scaling factor or a parameter that influences the conversion process,
 *          potentially representing the kernel size or other dimensional factors in the
 *          binary conversion process.
 *
 * @return BINARY_WORD* A pointer to the binary row representation. This pointer points
 *         to a location within the a_packed tensor's data, representing the binary form
 *         of the specified row in the input tensor.
 *
 * Note: The function assumes that the input tensor and the a_packed tensor are properly
 *       aligned and formatted for the binary conversion process. It performs a direct
 *       memory operation to map a floating-point tensor to its binary representation.
 */
BINARY_WORD* get_binary_row(
    torch::Tensor input,
    torch::Tensor a_packed,
    int m,
    int n,
    int k){
    BINARY_WORD* binary_row = (BINARY_WORD*)(a_packed.data_ptr());
    _get_binary_row(input.data_ptr<float>(), binary_row, m*k);
    return binary_row;
}


/**
 * Packs a weight tensor into a binary format.
 *
 * This function transforms a given floating-point weight tensor into a compact binary representation,
 * aiming to reduce memory footprint and possibly accelerate computation in neural networks.
 * It operates by first transposing the input tensor, ensuring it is contiguous, and then
 * converting it into a binary format where each bit represents a weight. The binary conversion
 * is performed by the `_get_binary_col_unrolled` function, which presumably unrolls the input
 * tensor and packs multiple floating-point values into a single binary word.
 *
 * @param weights The input weight tensor to be packed. Expected to be a floating-point tensor
 *                where the first dimension corresponds to the number of output channels
 *                and the second dimension to the number of input channels.
 * @param n The number of input channels, corresponds to the second dimension of the input tensor.
 * @param k The number of output channels, corresponds to the first dimension of the input tensor.
 * @return A tensor of type torch::kUInt8 representing the packed binary weights. The size of this
 *         tensor is calculated based on the number of input/output channels and the constant
 *         BITS_PER_BINARY_WORD, which defines how many bits are used for each binary word.
 *
 * Note: The `BITS_PER_BINARY_WORD` constant determines the packing density and must be defined
 *       elsewhere in the code. The `BINARY_WORD` type should match the size defined by this constant.
 *       This function assumes `weights` is a 2D tensor and the packing process is optimized
 *       for neural network weights specifically.
 */
torch::Tensor pack_weight(
    torch::Tensor weights,
    int n,
    int k) {
    weights = weights.transpose(0,1).contiguous();
    torch::Tensor binary_weight = torch::empty({k*n/BITS_PER_BINARY_WORD},
                                torch::TensorOptions().dtype(torch::kUInt8));
    BINARY_WORD* binary_col = (BINARY_WORD*)(binary_weight.data_ptr());
    _get_binary_col_unrolled(weights.data_ptr<float>(), binary_col, k, n);

    return binary_weight;
}


/**
 * Performs the forward pass of a binary linear layer.
 *
 * This function computes the forward pass for a binary linear layer using either binarized or unpacked weights,
 * depending on the state of the weights tensor. It supports efficient computation by packing the input tensor into
 * binary format before performing the binary linear operation.
 *
 * @param input The input tensor with shape [m, k], where m is the batch size, and k is the input feature dimension.
 * @param weights The weights tensor. It can be either in a binarized format with shape [k*n/BITS_PER_BINARY_WORD] if
 *                already binarized, or in an unpacked format with shape [k, n] for regular float weights.
 * @param m The batch size, indicating the number of input vectors.
 * @param n The output feature dimension.
 * @param k The input feature dimension.
 * @return torch::Tensor The output tensor of the binary linear operation with shape [m, n].
 *
 * The function first checks if the weights are binarized by comparing the total number of elements in the weights tensor
 * to what is expected if the weights were binarized (k*n/BITS_PER_BINARY_WORD). If the weights are already binarized,
 * it calls `binary_linear_forward_binarized_weight` to compute the forward pass using the binarized weights. Otherwise,
 * it calls `binary_linear_forward_unpacked_weight` to handle the computation with regular, unpacked weights.
 *
 * The input tensor is packed into binary format (a_packed) to facilitate the binary operations. This packing is done
 * based on the BITS_PER_BINARY_WORD, which defines how many bits are used to represent a binary word in the packed format.
 *
 * Note: This function assumes that the input tensor and weights have compatible shapes and that the input tensor is
 *       already prepared (e.g., quantized if necessary) for binary operations.
 */
torch::Tensor binary_linear_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int m,
    int n,
    int k) {

	auto option_quantize = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    auto a_packed = torch::zeros({m*k/BITS_PER_BINARY_WORD}, option_quantize);
    auto output = torch::zeros({m, n});

    BINARY_WORD* binary_row = get_binary_row(input, a_packed, m, n, k);
    // uses the total number of weight elements to identify if it is already binarized
    if(weights.numel() == k*n/BITS_PER_BINARY_WORD) {
        return binary_linear_forward_binarized_weight(weights, binary_row, m, n, k);
    }else{
        return binary_linear_forward_unpacked_weight(weights, binary_row, m, n, k);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &binary_linear_forward, "binary linear forward (CPU)");
    m.def("w_pack", &pack_weight, "binary weight packing (CPU)");
}
