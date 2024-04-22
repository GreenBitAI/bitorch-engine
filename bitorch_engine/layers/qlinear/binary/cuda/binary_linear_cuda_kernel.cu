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

namespace xnor_cuda {

typedef unsigned int BINARY_WORD; // 32-bit binary word
const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);
const int kMaxThreadsPerBlock (1024);

//get lane id
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));

__global__ void uint8_to_uint32(const uint8_t* in, uint32_t* out, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 4) {
        uint32_t val = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            val |= (uint32_t)in[(i*4) + j] << ((3-j)*8);
        }
        out[i] = val;
    }
}

__global__ void uint32_to_uint8(const uint32_t* in, uint8_t* out, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * 4) {
        int j = i / 4;
        int k = 3 - (i % 4);
        out[i] = (in[j] >> (k*8)) & 0xFF;
    }
}

// convert int64 pointer to uint32 pointer
__global__ void long2uint(long *a, BINARY_WORD *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) b[i] = (BINARY_WORD)(a[i] & 0xffffffff);
}


// convert int64 pointer to uint32 pointer
__global__ void uint2long(BINARY_WORD *a, long *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) b[i] = (long)(a[i]);
}


template <typename T>
__global__ void BMMA_toBit32Row_new(const T* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;

    T f0 = A[(bx*8+wx)*A_width+by*128+wy*32+laneid];
    BINARY_WORD r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
    //if (laneid == 0) B[(bx*8+wx)*A_width/32+wy] = r0;
    if (laneid == 0) B[(bx*gridDim.y+by)*8*128/(4*8)+wx*128/(4*8)+wy] = r0;
}

__global__ void BMMA_toBit32Row_new(const __nv_bfloat16* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;

    __nv_bfloat16 f0 = A[(bx*8+wx)*A_width+by*128+wy*32+laneid];
    BINARY_WORD r0 = __brev(__ballot_sync(0xffffffff, f0>=__float2bfloat16(0.0f)?1:0));
    //if (laneid == 0) B[(bx*8+wx)*A_width/32+wy] = r0;
    if (laneid == 0) B[(bx*gridDim.y+by)*8*128/(4*8)+wx*128/(4*8)+wy] = r0;
}

__global__ void BMMA_toBit32Row_new(const __half* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;

    __half f0 = A[(bx*8+wx)*A_width+by*128+wy*32+laneid];
    BINARY_WORD r0 = __brev(__ballot_sync(0xffffffff, __hge(f0, __float2half(0.0f))?1:0));
    //if (laneid == 0) B[(bx*8+wx)*A_width/32+wy] = r0;
    if (laneid == 0) B[(bx*gridDim.y+by)*8*128/(4*8)+wx*128/(4*8)+wy] = r0;
}

//================== BMMA_toBit32Col_new ==================//
__global__ void BMMA_toBit32Col_new(const __nv_bfloat16* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;

    __nv_bfloat16 f0 = A[(bx*128+wx*32+laneid)*A_width+(by*8)+wy];
    BINARY_WORD r0 = __brev(__ballot_sync(0xffffffff, f0>=__float2bfloat16(0.0f)?1:0));
    //if (laneid == 0) B[((by*8+wy)*A_height/32)+wx] = r0;
    if (laneid == 0) B[(by*gridDim.x+bx)*8*(128/(4*8))+wy*128/(4*8)+wx] = r0;
}

__global__ void BMMA_toBit32Col_new(const __half* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;

    __half f0 = A[(bx*128+wx*32+laneid)*A_width+(by*8)+wy];
    BINARY_WORD r0 = __brev(__ballot_sync(0xffffffff, __hge(f0, __float2half(0.0f))?1:0));
    //if (laneid == 0) B[((by*8+wy)*A_height/32)+wx] = r0;
    if (laneid == 0) B[(by*gridDim.x+bx)*8*(128/(4*8))+wy*128/(4*8)+wx] = r0;
}

template <typename T>
__global__ void BMMA_toBit32Col_new(const T* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;

    T f0 = A[(bx*128+wx*32+laneid)*A_width+(by*8)+wy];
    BINARY_WORD r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
    //if (laneid == 0) B[((by*8+wy)*A_height/32)+wx] = r0;
    if (laneid == 0) B[(by*gridDim.x+bx)*8*(128/(4*8))+wy*128/(4*8)+wx] = r0;
}
//====================================//


///k: A_width, n: B_width, m: A_height
__global__ void BMMAS_new(const BINARY_WORD *A, const BINARY_WORD *B,
		int *C, const unsigned m, const unsigned n, const unsigned k)
{
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    int bx = blockIdx.x * blockDim.y + threadIdx.y;
    int by = blockIdx.y;
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (int j=0; j<(k/128); j++)
    {
        load_matrix_sync(a_frag, A + bx*8*k/32 + j*128*8/32, 128);
        load_matrix_sync(b_frag, B + by*8*k/32 + j*128*8/32, 128);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    // convert to +1 and -1
    #pragma unroll
    for (int i=0; i<c_frag.num_elements; i++)
        c_frag.x[i] = k - 2*c_frag.x[i];

    store_matrix_sync(C+(bx*8*n+by*8), c_frag, n, wmma::mem_row_major);
}


//======================================================================================
// From row-major normal array to row-major 32-bit-array. This func is general which
// allows padding when A_width cannot divide 32.
//======================================================================================
template <typename T>
__global__ void ToBit32RowUd(const T* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval=0;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = ( (by*32+laneid<A_width) && (bx*32+i<A_height) ) ? A[(bx*32+i)*A_width+by*32+laneid]:(T)-1;
        Bval = (Bval<<1) + (f0>=0?1:0);
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}

__global__ void ToBit32RowUd(const __nv_bfloat16* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval=0;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        __nv_bfloat16 f0 = ( (by*32+laneid<A_width) && (bx*32+i<A_height) ) ?
                                A[(bx*32+i)*A_width+by*32+laneid]:
                                __float2bfloat16(-1.0f);
        Bval = (Bval<<1) + (f0>=__float2bfloat16(0.0f)?1:0);
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}

__global__ void ToBit32RowUd(const __half* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval=0;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        __half f0 = ( (by*32+laneid<A_width) && (bx*32+i<A_height) ) ?
                        A[(bx*32+i)*A_width+by*32+laneid]:
                        __float2half(-1.0f);
        Bval = (Bval<<1) + (__hge(f0, __float2half(0.0f))?1:0);
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}
//======================================================================================
// From row-major normal array to column-major 32-bit-array. This func is general which
// allows padding when A_width cannot divide 32.
//======================================================================================
template <typename T>
__global__ void ToBit32ColUd(const T* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = ( (by*32+laneid<A_width) && (bx*32+i<A_height) )?
            A[(bx*32+i)*A_width+by*32 +laneid]:(T)-1;
        unsigned r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
        if (laneid == i) Bval = r0;
    }
    if (laneid < A_height*A_width)
        B[by*gridDim.x*32+bx*32+laneid] = Bval;
}

__global__ void ToBit32ColUd(const __nv_bfloat16* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        __nv_bfloat16 f0 = ( (by*32+laneid<A_width) && (bx*32+i<A_height) )?
            A[(bx*32+i)*A_width+by*32 +laneid]:__float2bfloat16(-1.0f);
        unsigned r0 = __brev(__ballot_sync(0xffffffff, f0>=__float2bfloat16(0.0f)?1:0));
        if (laneid == i) Bval = r0;
    }
    if (laneid < A_height*A_width)
        B[by*gridDim.x*32+bx*32+laneid] = Bval;
}

__global__ void ToBit32ColUd(const __half* __restrict__ A, BINARY_WORD* B,
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        __half f0 = ( (by*32+laneid<A_width) && (bx*32+i<A_height) )?
            A[(bx*32+i)*A_width+by*32 +laneid]:__float2half(-1.0f);
        unsigned r0 = __brev(__ballot_sync(0xffffffff, __hge(f0, __float2half(0.0f))?1:0));
        if (laneid == i) Bval = r0;
    }
    if (laneid < A_height*A_width)
        B[by*gridDim.x*32+bx*32+laneid] = Bval;
}

//======================================================================================
// This function performs 32-bit Matmul with padding. A and B are 32-bit-array,
// C is normal array in row-major. The dot product is among A-row and B-row.
// A(A_width, A_height) * B(B_height, B_width) = C(A_height, B_width), A_width = B_height
//======================================================================================
template <typename T>
__global__ void BMM32_Arow_Brow_UD(const BINARY_WORD* __restrict__ A, const BINARY_WORD* __restrict__ B,
        T* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    T* Csub = &C[blockIdx.x*B_width*32+blockIdx.y*32];
    register unsigned Cm[32] = {0};
    const int steps = (A_width+31)/32*32;

    for (int i = 0; (i*32) < steps; i++)
    {
        unsigned r0 = Asub[i*32*gridDim.x+laneid];
        unsigned r1 = Bsub[i*32*gridDim.y+laneid];
#pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __shfl_sync(0xffffffff, r1, j); //from lane-j, r1 of matrix B
            Cm[j] += __popc(r0 ^ r2); //can remove C to exploit register reuse
        }
    }
    if ( (blockIdx.x*32+laneid)<A_height )
    {
        for (int i=0; i<32; i++)
            if (blockIdx.y*32+i<B_width)
                Csub[laneid*B_width+i] = A_width - (T)(Cm[i])*2;
    }
}

__global__ void BMM32_Arow_Brow_UD(const BINARY_WORD* __restrict__ A, const BINARY_WORD* __restrict__ B,
        __nv_bfloat16* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    __nv_bfloat16* Csub = &C[blockIdx.x*B_width*32+blockIdx.y*32];
    register unsigned Cm[32] = {0};
    const int steps = (A_width+31)/32*32;

    for (int i = 0; (i*32) < steps; i++)
    {
        unsigned r0 = Asub[i*32*gridDim.x+laneid];
        unsigned r1 = Bsub[i*32*gridDim.y+laneid];
#pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __shfl_sync(0xffffffff, r1, j); //from lane-j, r1 of matrix B
            Cm[j] += __popc(r0 ^ r2); //can remove C to exploit register reuse
        }
    }
    if ( (blockIdx.x*32+laneid)<A_height )
    {
        for (int i=0; i<32; i++)
            if (blockIdx.y*32+i<B_width)
                Csub[laneid*B_width+i] = __float2bfloat16(static_cast<float>(A_width - (int)(Cm[i])*2));
    }
}

__global__ void BMM32_Arow_Brow_UD(const BINARY_WORD* __restrict__ A, const BINARY_WORD* __restrict__ B,
        __half* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    __half* Csub = &C[blockIdx.x*B_width*32+blockIdx.y*32];
    register unsigned Cm[32] = {0};
    const int steps = (A_width+31)/32*32;

    for (int i = 0; (i*32) < steps; i++)
    {
        unsigned r0 = Asub[i*32*gridDim.x+laneid];
        unsigned r1 = Bsub[i*32*gridDim.y+laneid];
#pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __shfl_sync(0xffffffff, r1, j); //from lane-j, r1 of matrix B
            Cm[j] += __popc(r0 ^ r2); //can remove C to exploit register reuse
        }
    }
    if ( (blockIdx.x*32+laneid)<A_height )
    {
        for (int i=0; i<32; i++)
            if (blockIdx.y*32+i<B_width)
                Csub[laneid*B_width+i] =__float2half(static_cast<float>(A_width - (int)(Cm[i])*2));
    }
}
} // namespace cuda



inline std::vector<int> get_divisors(int num){
    std::vector<int> dv;
    int square_root = (int) sqrt(num) + 1;
    for (int i = 1; i < square_root; i++) {
        if (num % i == 0 && i*i==num){
            dv.push_back(i);
        }else if (num % i == 0 && i*i!=num){
            dv.push_back(i);
            dv.push_back(num/i);
        }
    }
    return dv;
}

inline int get_next_block_dim(int c){
    std::vector<int> divs = get_divisors(c);
    if (!divs.empty()){
        int dim =
            divs.at(divs.size()-1)
                < xnor_cuda::kMaxThreadsPerBlock
                ? divs.at(divs.size()-1) : xnor_cuda::kMaxThreadsPerBlock;
        return dim;
    }
    return 1;
}

/*
 * m: number of output channels (num_filter) per group
 * n: number of input channels per group * kernel size(e.g., 3x3=9) / BITS_PER_BINARY_WORD
 * k: number of pixels of output images per channel (output dimension)
 */
inline int get_next_block_dim(int m, int n, int k){
    std::vector<int> divs = get_divisors(n);
    int square_root_max_threads = (int) sqrt(xnor_cuda::kMaxThreadsPerBlock) + 1;
    if (!divs.empty()){
        std::sort(divs.begin(), divs.end());
        for (int i = divs.size()-1; i > -1 ; --i){
            int sel_mid = divs[i];
            if (sel_mid < m/2 && sel_mid < k/2 && sel_mid <= square_root_max_threads){
                return sel_mid;
            }
        }
    }
    return 1;
}


// C++-CUDA methods

// already bit-packed weights
xnor_cuda::BINARY_WORD* from_binarized_weights(
    torch::Tensor weights,
    int n,
    int k
){

    //set data pointers
    uint8_t *fB = weights.data_ptr<uint8_t>();

	// Get binary col from weights via bit-packing (k x n) -> (k/32 x n)
	xnor_cuda::BINARY_WORD* binary_col;
    cudaMalloc(
        &binary_col,
        (k*n/xnor_cuda::BITS_PER_BINARY_WORD * sizeof(xnor_cuda::BINARY_WORD)));

    // 256 is tested to be the fastest option for 1080Ti
	int threads_per_block = 256; //get_next_block_dim(n*k/xnor_cuda::BITS_PER_BINARY_WORD);
	dim3 block_w(threads_per_block, 1, 1);
	dim3 grid_w(n*k*4/(threads_per_block*xnor_cuda::BITS_PER_BINARY_WORD)+1, 1);
	xnor_cuda::uint8_to_uint32<<<
		grid_w,
		block_w,
		0,
		c10::cuda::getCurrentCUDAStream()>>>(
			fB,
			binary_col,
			n*k*4/xnor_cuda::BITS_PER_BINARY_WORD);

    return binary_col;
}


template <typename T_i, typename T_w>
torch::Tensor binary_linear_forward_BTC(
    torch::Tensor input,
    torch::Tensor weights,
    int m,
    int n,
    int k
){
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    auto output = torch::empty(
        {m, n},
        torch::TensorOptions().dtype(torch::kInt32).device(input.device() ));
    T_i *fA = reinterpret_cast<T_i *>(input.data_ptr());
    int *tC = output.data_ptr<int>();
    xnor_cuda::BINARY_WORD *tA, *tB;

	cudaMalloc(
		&tA,
		m*k/xnor_cuda::BITS_PER_BINARY_WORD*sizeof(xnor_cuda::BINARY_WORD));

    // uses the total number of weight elements to identify if it is already binarized
    // *4 since we use uint8 storing binary weights in pytorch
    if(weights.numel() == k*n*4/xnor_cuda::BITS_PER_BINARY_WORD
            && weights.dtype() == torch::kUInt8){ /// binarized weight
        tB = from_binarized_weights(weights, n, k);
    }else{ /// weights
        T_w *fB = reinterpret_cast<T_w *>(weights.data_ptr());
        cudaMalloc(
            &tB,
            k*n/xnor_cuda::BITS_PER_BINARY_WORD*sizeof(xnor_cuda::BINARY_WORD));

        xnor_cuda::BMMA_toBit32Col_new<<<
            dim3(k/128,n/8),
            dim3(32,4,8),
            0,
            c10::cuda::getCurrentCUDAStream()
            >>>(fB, tB, k, n);
    }

    /// bit packing for inputs & tensor core based Bgemm
    xnor_cuda::BMMA_toBit32Row_new<<<
        dim3(m/8,k/128),
        dim3(32,8,4),
        0,
        c10::cuda::getCurrentCUDAStream()
        >>>(fA, tA, m, k);

    xnor_cuda::BMMAS_new<<<
        dim3(m/8, n/8),
        dim3(32, 1),
        0,
        c10::cuda::getCurrentCUDAStream()
        >>>(tA, tB, tC, m, n, k);

    cudaFree(tA);
    cudaFree(tB);

    // check if convert to the same dtype as input
    if(output.dtype() != input.dtype())
        output = output.to(input.dtype());
    return output;
}


template <typename T_i, typename T_w>
torch::Tensor binary_linear_forward_BSTC(
    torch::Tensor input,
    torch::Tensor weight,
    int m,
    int n,
    int k
){
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    auto output = torch::empty(
                    {m, n},
                    torch::TensorOptions()
                        .dtype(input.dtype())
                        .device(input.device())
                  );
    T_i *fA = reinterpret_cast<T_i *>(input.data_ptr());
    T_i *fC = reinterpret_cast<T_i *>(output.data_ptr());
    xnor_cuda::BINARY_WORD *tA, *tB;
	cudaMalloc(
		&tA,
		((m+31)/32*32) * (k+31)/xnor_cuda::BITS_PER_BINARY_WORD * sizeof(xnor_cuda::BINARY_WORD));

    // uses the total number of weight elements to identify if it is already binarized
    if(weight.numel() == k*n*4/xnor_cuda::BITS_PER_BINARY_WORD
            && weight.dtype() == torch::kUInt8){ /// binarized weight
        tB = from_binarized_weights(weight, n, k);
    }else{ /// weight
        T_w *fB = reinterpret_cast<T_w *>(weight.data_ptr());
        cudaMalloc(
            &tB,
            ((k+31)/32*32) * (n+31)/xnor_cuda::BITS_PER_BINARY_WORD * sizeof(xnor_cuda::BINARY_WORD));

        xnor_cuda::ToBit32RowUd<<<
            dim3((k+31)/32,
            (n+31)/32),
            32,
            0,
            c10::cuda::getCurrentCUDAStream()
            >>>(fB, tB, k, n);
    }

    xnor_cuda::ToBit32ColUd<<<
        dim3((m+31)/32,(k+31)/32),
        32,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(fA, tA, m, k);

    xnor_cuda::BMM32_Arow_Brow_UD<<<
        dim3((m+31)/32,(n+31)/32),
        32,
        0,
        c10::cuda::getCurrentCUDAStream()
        >>>(tA,
            tB,
            fC,
            m,
            k,
            n);

    cudaFree(tA);
    cudaFree(tB);
    return output;
}


template <typename T_i, typename T_w>
torch::Tensor binary_linear_forward_combined(
    torch::Tensor input,
    torch::Tensor weight,
    int m,
    int n,
    int k
){
    torch::Tensor output;
    if(m % 8 == 0 && k % 128 == 0 && n % 8 == 0)
        output = binary_linear_forward_BTC<T_i, T_w>(input, weight, m, n, k);
    else
        output = binary_linear_forward_BSTC<T_i, T_w>(input, weight, m, n, k);
    return output;
}


template <typename T_i, typename T_w>
torch::Tensor _binary_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int bmm_type,
    bool transpose
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    torch::Tensor output;
     // get m, k, n
    int m = input.size(0);
    int k = input.size(1);
    int n;

    if(weight.dtype() == torch::kUInt8){
        n = (int)((float)weight.numel()/(input.size(1)/xnor_cuda::BITS_PER_BINARY_WORD*4));
    }else{
        n = weight.size(0); // shape(n, k)
        if (transpose)
            weight = weight.t().contiguous();
    }

    if(bmm_type == 1)
        output = binary_linear_forward_BSTC<T_i, T_w>(input, weight, m, n, k);
    else if(bmm_type == 2)
        output = binary_linear_forward_BTC<T_i, T_w>(input, weight, m, n, k);
    else
        output = binary_linear_forward_combined<T_i, T_w>(input, weight, m, n, k);
    return output;
}


template <typename T>
torch::Tensor _binary_mm_cuda(
    torch::Tensor x,
    torch::Tensor y,
    int bmm_type
){
	const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    torch::Tensor output;
     // get m, k, n
    int m = x.size(0);
    int k = x.size(1);
    int n = y.size(1);

    if(bmm_type == 1)
        output = binary_linear_forward_BSTC<T, T>(x, y, m, n, k);
    else if(bmm_type == 2)
        output = binary_linear_forward_BTC<T, T>(x, y, m, n, k);
    else
        output = binary_linear_forward_combined<T, T>(x, y, m, n, k);
    return output;
}


/**
 * Performs a forward pass of the binary linear layer using CUDA.
 *
 * This function facilitates binary linear operations with support for different data types
 * for input and weight tensors, including floating point and quantized types. It leverages
 * CUDA for efficient computation, especially suited for deep learning models running on GPU.
 *
 * @param input The input tensor, which can be of type torch::kFloat32 (float), torch::kBFloat16 (bfloat16),
 *              or torch::kHalf (half).
 * @param weight The weight tensor, which supports torch::kInt8 (int8), torch::kFloat32 (float),
 *               and torch::kUInt8 (uint8) data types.
 * @param bmm_type An integer specifying the type of binary matrix multiplication to perform.
 *                 This parameter allows for customization of the operation based on the model's requirements.
 * @param transpose A boolean indicating whether the weight matrix should be transposed during the operation.
 *
 * @return A tensor containing the result of the binary linear operation.
 *
 * @note This function dynamically dispatches to specialized template functions based on the data types of
 *       the input and weight tensors. It supports a combination of float, bfloat16, half, int8, and uint8
 *       types, ensuring flexibility in handling various neural network architectures.
 *       If the data type of the input or weight tensor is not supported, the function will terminate the
 *       program and print an error message.
 */
torch::Tensor binary_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int bmm_type,
    bool transpose
) {
	if(weight.dtype() == torch::kInt8){
		if(input.dtype() == torch::kFloat32)
            return _binary_linear_cuda_forward<float, int8_t>(input, weight, bmm_type, transpose);
        else if(input.dtype() == torch::kBFloat16)
            return _binary_linear_cuda_forward<__nv_bfloat16, int8_t>(input, weight, bmm_type, transpose);
        else if(input.dtype() == torch::kHalf)
            return _binary_linear_cuda_forward<__half, int8_t>(input, weight, bmm_type, transpose);
        else{
            std::cerr << "tensor type not supported: " << input.dtype() << std::endl;
            exit(EXIT_FAILURE);
        }
    }else if(weight.dtype() == torch::kFloat32){
        if(input.dtype() == torch::kFloat32)
            return _binary_linear_cuda_forward<float, float>(input, weight, bmm_type, transpose);
        else if(input.dtype() == torch::kBFloat16)
            return _binary_linear_cuda_forward<__nv_bfloat16, float>(input, weight, bmm_type, transpose);
        else if(input.dtype() == torch::kHalf)
            return _binary_linear_cuda_forward<__half, float>(input, weight, bmm_type, transpose);
        else{
            std::cerr << "tensor type not supported: " << input.dtype() << std::endl;
            exit(EXIT_FAILURE);
        }
    }else if(weight.dtype() == torch::kUInt8){
        if(input.dtype() == torch::kFloat32)
            return _binary_linear_cuda_forward<float, uint8_t>(input, weight, bmm_type, transpose);
        else if(input.dtype() == torch::kBFloat16)
            return _binary_linear_cuda_forward<__nv_bfloat16, uint8_t>(input, weight, bmm_type, transpose);
        else if(input.dtype() == torch::kHalf)
            return _binary_linear_cuda_forward<__half, uint8_t>(input, weight, bmm_type, transpose);
        else{
            std::cerr << "tensor type not supported: " << input.dtype() << std::endl;
            exit(EXIT_FAILURE);
        }
    }else{
        std::cerr << "tensor type not supported: " << weight.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }
}


/**
 * Performs binary matrix multiplication on CUDA using specified data types.
 *
 * This function dispatches the binary matrix multiplication operation to specialized
 * CUDA kernels based on the data type of the input tensors. It supports int8, float32,
 * bfloat16, and half (float16) data types. The function checks if the data types of both
 * input tensors match and then calls the appropriate templated CUDA kernel function.
 *
 * @param x A torch::Tensor representing the first matrix in the multiplication.
 * @param y A torch::Tensor representing the second matrix in the multiplication.
 * @param bmm_type An integer indicating the type of binary matrix multiplication to perform.
 *
 * @return A torch::Tensor containing the result of the binary matrix multiplication.
 *
 * @throws std::runtime_error If the input tensors have different data types or if an unsupported
 *         data type is provided.
 */
torch::Tensor binary_mm_cuda(
    torch::Tensor x,
    torch::Tensor y,
    int bmm_type
){
    if(y.dtype() != x.dtype()){
        std::cerr << "The input tensors must have the same dtype. x_dtype: " << x.dtype()
                   << " and y_dtype: " << y.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }

    if(y.dtype() == torch::kInt8){
        return _binary_mm_cuda<int8_t>(x, y, bmm_type);
    }else if(y.dtype() == torch::kFloat32){
        return _binary_mm_cuda<float>(x, y, bmm_type);
    }else if(y.dtype() == torch::kBFloat16){
        return _binary_mm_cuda<__nv_bfloat16>(x, y, bmm_type);
    }else if(y.dtype() == torch::kHalf){
        return _binary_mm_cuda<__half>(x, y, bmm_type);
    }
    else{
        std::cerr << "tensor type not supported: " << y.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }
}


/**
 * Transforms a floating-point weight tensor into a binary packed tensor using CUDA.
 * This function supports different binary matrix multiplication (BMM) types and allows for optional weight transposition.
 *
 * The binary weight tensor is packed into 8-bit unsigned integers (uint8_t) where each bit represents a binary weight.
 * This packing process reduces memory footprint and can accelerate binary neural network operations on CUDA-enabled devices.
 *
 * Parameters:
 *   - weight (torch::Tensor): The floating-point weight tensor to be binarized and packed. It should have a shape of [n, k].
 *   - bmm_type (int): Specifies the binary matrix multiplication (BMM) type. Different types may indicate different
 *                     binarization and packing strategies. For example, '2' and '3' have specific conditions for using BTC32 packing.
 *   - transpose (bool): If true, transposes the weight tensor before binarization. This is useful for aligning the data
 *                       for certain operations that require the weights to be in a specific orientation.
 *
 * Process:
 *   1. Optionally transposes the weight tensor for alignment purposes.
 *   2. Allocates a tensor `packed_w` to hold the binarized and packed weights in uint8 format.
 *   3. Depending on the `bmm_type` and dimensions of `weight`, chooses the appropriate CUDA kernel to perform
 *      the binarization and packing. There are two main paths:
 *        - BTC32: Used when `bmm_type` is 2, or when `bmm_type` is 3 and both dimensions `n` and `k` meet specific divisibility conditions.
 *        - BSTC-32: Used in other cases, adjusting the dimensions to be multiples of 32 if necessary.
 *   4. Invokes a CUDA kernel to pack the 32-bit binary columns into 8-bit unsigned integers.
 *   5. Frees the GPU memory allocated for the binary column representation.
 *
 * Returns:
 *   - torch::Tensor: The binarized and packed weight tensor, now in uint8 format, suitable for efficient binary operations on CUDA devices.
 *
 * Note:
 *   - This function assumes the input tensor is already on a CUDA device and utilizes CUDA streams for kernel execution.
 *   - The use of `xnor_cuda::BITS_PER_BINARY_WORD` implies a specific packing strategy, typically 32 bits per binary word.
 */
template <typename T>
torch::Tensor _get_binary_weight_cuda(
    torch::Tensor weight,
    int bmm_type,
    bool transpose
) {
	const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));

    int n = weight.size(0);
    int k = weight.size(1);
    /// weight transpose
    if (transpose)
        weight = weight.t().contiguous();

    torch::Tensor packed_w =
        torch::empty(
           {k*n/xnor_cuda::BITS_PER_BINARY_WORD*(32/8)},
           torch::TensorOptions().dtype(torch::kUInt8).device(weight.device())
        );
    xnor_cuda::BINARY_WORD* binary_col;
    T *fB = reinterpret_cast<T *>(weight.data_ptr());

    if(bmm_type == 2 || (bmm_type == 3 && k % 128 == 0 && n % 8 == 0)){///BTC32
        cudaMalloc(
            &binary_col,
            k*n/xnor_cuda::BITS_PER_BINARY_WORD*sizeof(xnor_cuda::BINARY_WORD));

        xnor_cuda::BMMA_toBit32Col_new<<<
            dim3(k/128,n/8),
            dim3(32,4,8),
            0,
            c10::cuda::getCurrentCUDAStream()
            >>>(fB, binary_col, k, n);
    }else{ ///BSTC-32
        cudaMalloc(
            &binary_col,
            ((k+31)/32*32) * (n+31)/xnor_cuda::BITS_PER_BINARY_WORD * sizeof(xnor_cuda::BINARY_WORD));
        xnor_cuda::ToBit32RowUd<<<
            dim3((k+31)/32,(n+31)/32),
            32,
            0,
            c10::cuda::getCurrentCUDAStream()
            >>>(fB, binary_col, k, n);
    }

    int threads_per_block = get_next_block_dim(n*k/xnor_cuda::BITS_PER_BINARY_WORD*(32/8));
    dim3 block_w(threads_per_block, 1, 1);
    dim3 grid_w(n*k*4/(threads_per_block*xnor_cuda::BITS_PER_BINARY_WORD)+1, 1);
    xnor_cuda::uint32_to_uint8<<<
        grid_w,
        block_w,
        0,
        c10::cuda::getCurrentCUDAStream()
        >>>(binary_col,
	        packed_w.data_ptr<uint8_t>(),
	        n*k/xnor_cuda::BITS_PER_BINARY_WORD);

    cudaFree(binary_col);
    return packed_w;
}



/**
 * Converts a given weight tensor to its binary representation based on the specified data type.
 *
 * This function supports weight tensors of different data types (int8, float, bfloat16, and half)
 * and converts them to a binary format suitable for certain binary matrix multiplication (BMM) operations.
 * The conversion process is dependent on the data type of the input tensor and whether the tensor
 * should be transposed as part of the conversion.
 *
 * @param weight The input weight tensor to be converted to binary format.
 * @param bmm_type An integer specifying the type of binary matrix multiplication operation.
 *                 This parameter can influence how the binary conversion is performed.
 * @param transpose A boolean indicating whether the weight tensor should be transposed
 *                  as part of the conversion process.
 * @return torch::Tensor A tensor containing the binary representation of the input weight tensor.
 *                       The specific format of the binary representation is determined by the
 *                       data type of the input tensor.
 *
 * @note This function is templated to handle different data types of the input tensor by
 *       calling the appropriate specialized version of the _get_binary_weight_cuda function.
 *       If the data type of the input tensor is not supported, the function prints an error message
 *       and exits the program.
 */
torch::Tensor get_binary_weight_cuda(
    torch::Tensor weight,
    int bmm_type,
    bool transpose
) {
	if(weight.dtype() == torch::kInt8){
        return _get_binary_weight_cuda<int8_t>(weight, bmm_type, transpose);
    }else if(weight.dtype() == torch::kFloat32){
        return _get_binary_weight_cuda<float>(weight, bmm_type, transpose);
    }else if(weight.dtype() == torch::kBFloat16){
        return _get_binary_weight_cuda<__nv_bfloat16>(weight, bmm_type, transpose);
    }else if(weight.dtype() == torch::kHalf){
        return _get_binary_weight_cuda<__half>(weight, bmm_type, transpose);
    }else{
        std::cerr << "tensor type not supported: " << weight.dtype() << std::endl;
        exit(EXIT_FAILURE);
    }
}

