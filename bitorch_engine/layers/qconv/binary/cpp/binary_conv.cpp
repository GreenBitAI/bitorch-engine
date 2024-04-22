#include <torch/extension.h>
#include <torch/nn/options/padding.h>
#include <torch/nn/options/fold.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <omp.h>

// variable, position, value
#define BIT_SET(var, pos, val) var |= (val << pos)
typedef uint8_t BINARY_WORD;
const int BITS_PER_BINARY_WORD = sizeof(BINARY_WORD) * CHAR_BIT;
#define UNROLLN 6

/**
 * @brief Binarizes a matrix row into a compact binary format.
 *
 * This function converts a single row of a floating-point matrix into a binary representation,
 * where each bit in the output represents the sign (positive or negative) of the corresponding
 * floating-point element. The binary representation is stored in a format optimized for
 * performance and space efficiency, using a predefined number of bits per binary word.
 *
 * The binarization process involves setting each bit in the binary word based on the sign of
 * each floating-point number: a bit is set to 1 if the number is non-negative (>=0) and set to 0
 * if the number is negative. This compact binary format is particularly useful for operations
 * that can exploit bitwise operations for computational efficiency.
 *
 * @param row Pointer to the floating-point array representing the matrix row to be binarized.
 * @param b_row Pointer to the output array where the binarized row will be stored. Each element
 *              in this array is a BINARY_WORD containing the binary representation of a chunk of
 *              floating-point numbers from the input row.
 * @param size The total number of floating-point elements in the input row. This determines the
 *             length of the loop and the number of chunks to be processed.
 *
 * @note This function uses OpenMP for parallel processing to enhance performance. It parallelizes
 *       both the computation of signs for elements within a chunk and the combination of these
 *       signs into the binary representation.
 */
void _get_binary_row(float* row, BINARY_WORD * b_row, int size){
    #pragma omp parallel for
    for (int i = 0; i < size; i += BITS_PER_BINARY_WORD) {
        BINARY_WORD rvalue = 0;
        BINARY_WORD sign[BITS_PER_BINARY_WORD];

        // Compute the signs for all elements in the chunk in parallel
        #pragma omp simd
        for (int j = 0; j < BITS_PER_BINARY_WORD; ++j) {
            sign[j] = (row[i + j] >= 0);
        }

        // Combine the signs into the binary representation
        for (int j = 0; j < BITS_PER_BINARY_WORD; ++j) {
            BIT_SET(rvalue, j, sign[j]);
        }

        b_row[i / BITS_PER_BINARY_WORD] = rvalue;
    }
}

/**
  * @brief binarize matrix column wise.
  * Loop unroll and using register vars.
  * ~30% performance improvement without openmp
  * compared with get_binary_col() method.
  */
void _get_binary_col_unrolled(float* col, BINARY_WORD * b_col, int n, int k)
{
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

void _xnor_gemm_unrolled_deprecated(int M, int N, int K,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc
){
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
 * Implements an optimized XNOR GEMM (General Matrix Multiply) operation using loop unrolling and parallelization.
 * This function is specifically designed for binary neural network computations, where both input matrices are
 * binary, and the multiplication operation is replaced by XNOR and popcount operations.
 *
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 * @param A Pointer to the first binary matrix (input).
 * @param lda Leading dimension of matrix A, typically equal to K.
 * @param B Pointer to the second binary matrix (input).
 * @param ldb Leading dimension of matrix B, typically equal to N.
 * @param C Pointer to the output matrix. Must be preallocated and initialized to zeros.
 * @param ldc Leading dimension of matrix C, typically equal to N.
 *
 * The function computes the matrix product of A and B, storing the result in C. Instead of traditional multiplication,
 * it uses bitwise XNOR followed by popcount operations to efficiently perform binary matrix multiplication.
 * The computation is optimized through loop unrolling and OpenMP parallelization to enhance performance.
 */
void _xnor_gemm_unrolled(int M, int N, int K,
                         BINARY_WORD *A, int lda,
                         BINARY_WORD *B, int ldb,
                         float *C, int ldc
 ) {
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

template <typename T>
void im2col(
    const T* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_col
) {
    const int64_t height_col = output_height;
    const int64_t width_col = output_width;
    const int64_t channels_col = channels * kernel_h * kernel_w;

    for (const auto c_col : c10::irange(channels_col)) {
        int64_t w_offset = c_col % kernel_w;
        int64_t h_offset = (c_col / kernel_w) % kernel_h;
        int64_t c_im = c_col / kernel_h / kernel_w;

        for (const auto h_col : c10::irange(height_col)) {
            int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

            for (const auto w_col : c10::irange(width_col)) {
                int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                data_col[(c_col * height_col + h_col) * width_col + w_col] =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                    ? data_im[(c_im * height + h_im) * width + w_im]
                    : static_cast<T>(-1);
            }
        }
    }
}


/**
 * Converts an image to its binary column representation for efficient binary convolution operations.
 *
 * This function transforms input image pixels into binary format based on their sign, facilitating
 * the use of binary operations in convolutional neural networks. It applies necessary padding,
 * stride, and dilation as per convolutional layer parameters.
 *
 * @param data_im Pointer to the input image data in floating-point format.
 * @param channels Number of channels in the input image.
 * @param height Height of the input image.
 * @param width Width of the input image.
 * @param output_height Height of the output feature map.
 * @param output_width Width of the output feature map.
 * @param kernel_h Height of the convolution kernel.
 * @param kernel_w Width of the convolution kernel.
 * @param pad_h Padding applied along the height of the image.
 * @param pad_w Padding applied along the width of the image.
 * @param stride_h Stride applied along the height of the image.
 * @param stride_w Stride applied along the width of the image.
 * @param dilation_h Dilation applied to the height of the kernel.
 * @param dilation_w Dilation applied to the width of the kernel.
 * @param data_col Pointer to the output buffer for the binary column data.
 *
 * The function iterates over each channel, height, and width of the output feature map,
 * converting the relevant input image pixels to binary form and storing them in `data_col`.
 * Pixels falling outside the input image due to padding are considered as zero.
 */
void im2binary_col(
    const float* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    BINARY_WORD* data_col
) {

    const int64_t height_col = output_height;
    const int64_t width_col = output_width;
    const int64_t channels_col = channels * kernel_h * kernel_w;

    for (const auto c_col : c10::irange(channels_col / BITS_PER_BINARY_WORD)) {
        for (const auto h_col : c10::irange(height_col)) {
            for (const auto w_col : c10::irange(width_col)) {
                BINARY_WORD rvalue=0;
                BINARY_WORD sign;
                for(int b=0; b<BITS_PER_BINARY_WORD; ++b){
                    auto bc_col = (c_col*BITS_PER_BINARY_WORD + b);
                    int64_t w_offset = bc_col % kernel_w;
                    int64_t h_offset = (bc_col / kernel_w) % kernel_h;
                    int64_t c_im = bc_col / kernel_h / kernel_w;
                    int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
                    int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                    if(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                        sign = (data_im[(c_im * height + h_im) * width + w_im]>=0);
                    else
                        sign = 0; // padding elements
                    BIT_SET(rvalue, b, sign);
                }

                data_col[(c_col * height_col + h_col) * width_col + w_col] = rvalue;
            }
        }
    }
}


torch::Tensor binary_conv_gemm(
    BINARY_WORD* binary_row,
    BINARY_WORD* binary_col,
    torch::Tensor& output,
    int m,
    int n,
    int k
){
    _xnor_gemm_unrolled(m, n, k/BITS_PER_BINARY_WORD,
          binary_row, k/BITS_PER_BINARY_WORD,
          binary_col, n,
          output.data_ptr<float>(), n);

    return output;
}


// already bit-packed weights
BINARY_WORD* get_binarized_input_col(
    torch::Tensor input_col,
    BINARY_WORD* binary_col,
    int m,
    int n,
    int k
){

    _get_binary_col_unrolled(input_col.data_ptr<float>(), binary_col, k, n);

    return binary_col;
}


BINARY_WORD* get_binarized_weight(
    torch::Tensor weights,
    int m,
    int n,
    int k
){
    BINARY_WORD * binary_row;

    if(weights.numel() == m*k/BITS_PER_BINARY_WORD) {
        binary_row = (BINARY_WORD *)weights.data_ptr();
    }else{
        binary_row = new BINARY_WORD[m*k/BITS_PER_BINARY_WORD];
        _get_binary_row(weights.data_ptr<float>(), binary_row, m*k);
    }

    return binary_row;
}


BINARY_WORD** get_batched_binary_arr(
    int batch_size,
    int arr_len
){
    BINARY_WORD **binary_arrs = new BINARY_WORD*[batch_size];
    for (int i=0; i < batch_size; i++){
        binary_arrs[i] = new BINARY_WORD[arr_len];
    }
    return binary_arrs;
}


void release_batched_binary_arr(
    BINARY_WORD** batched_arr,
    int batch_size
){
    for (int i=0; i < batch_size; i++){
        delete [] batched_arr[i];
    }
    delete[] batched_arr;
}


/**
 * Performs the forward pass of binary convolution.
 *
 * Args:
 *   input (torch::Tensor): The input tensor with shape (batch_size, channels, height, width).
 *   weights (torch::Tensor): The weights tensor for the convolution.
 *   m (int): The number of output channels.
 *   n (int): The spatial size of the output tensor (height*width).
 *   k (int): The total number of input elements contributing to each output (i.e., channels*kernel_height*kernel_width).
 *   kernel_size (int): The size of the convolution kernel.
 *   stride (int): The stride of the convolution.
 *   padding (int): The padding added to both sides of the input tensor.
 *   dilation (int): The spacing between kernel elements.
 *   output_edge (int): The size of one edge (height or width) of the output tensor.
 *
 * Returns:
 *   torch::Tensor: The output tensor of the convolution with shape (batch_size, m, output_edge, output_edge).
 *
 * This function performs a binary convolution operation using the provided input and weights tensors.
 * It supports different kernel sizes, strides, paddings, and dilations. The computation is optimized
 * for performance using OpenMP for parallel processing across multiple threads.
 */
torch::Tensor binary_conv_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int m,
    int n,
    int k,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_edge
) {
    namespace F = torch::nn::functional;

    int batch_size = input.size(0);
    // debug m, n, k
//    std::cout << "m:" << m << " n:"<< n << " k:" << k << ", batch_size:" << batch_size << std::endl;

    /// output tensor
    auto output = torch::zeros({batch_size, m, output_edge, output_edge});
    /// container for binary cols and rows
    BINARY_WORD **binary_cols = get_batched_binary_arr(batch_size, k*n/BITS_PER_BINARY_WORD);
    // uses the total number of weight elements to identify if it is already binarized
    BINARY_WORD * binary_row = get_binarized_weight(weights, m, n, k);
    const bool requires_columns = (kernel_size != 1 || stride != 1 || padding != 0 || dilation != 1);
    int num_threads = omp_get_max_threads();
    omp_set_dynamic(0); // Explicitly disable dynamic teams
    omp_set_num_threads(num_threads);
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
        for (const auto n_index : c10::irange(start, end)) {
            auto output_2d = output[n_index];
            auto input_3d = input[n_index];
            BINARY_WORD* binary_col = binary_cols[n_index];
            auto input_col = torch::zeros({input.size(1) * kernel_size * kernel_size, output_edge * output_edge},
                                    input.options());
            if(requires_columns){
                im2binary_col(
                    input_3d.data_ptr<float>(), // input tensor
                    input.size(1), // input channels
                    input.size(2), // input height
                    input.size(3), // input width
                    output_edge,
                    output_edge,
                    kernel_size,
                    kernel_size,
                    padding,
                    padding,
                    stride,
                    stride,
                    dilation,
                    dilation,
                    binary_col
                );
            }else{
                get_binarized_input_col(input_3d, binary_col, m, n, k);
            }
            /// XOR gemm
            binary_conv_gemm(binary_row, binary_col, output_2d, m, n, k);
        }

    }); /// end at::parallel_for

    /// free memory
    delete [] binary_row;
    release_batched_binary_arr(binary_cols, batch_size);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &binary_conv_forward, "binary conv forward (CPU)");
}
