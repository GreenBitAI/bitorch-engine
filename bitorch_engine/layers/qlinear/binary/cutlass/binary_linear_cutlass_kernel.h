#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/tensor_view.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/gemm/device/gemm_batched.h>

namespace xnor_cutlass {

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

using ElementAccumulator = int32_t;
using ElementCompute = int32_t;
using ElementInput1Bit = cutlass::uint1b_t;
using ElementOutput = int32_t;
using ElementComputeEpilogue = ElementOutput;

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, B and C.
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<32>;  //
using SwizzleThreadBlockBatched = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
// Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
//
//    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
//
using EpilogueOp =
	cutlass::epilogue::thread::LinearCombination<
          ElementOutput,                                    // <- data type of output matrix
          128 / cutlass::sizeof_bits<ElementOutput>::value, // <- this is the number of elements per
                                                            // vectorized memory access. For half
                                                            // precision, it's 8 elements. This becomes
                                                            // the vector width of math instructions in
                                                            // epilogue too
          ElementAccumulator,
          ElementCompute>;

// ====================== ARCH_SM_80 ====================== //
#ifdef ARCH_SM_80 // sm_80
// small 1 case 7
inline cudaError_t sm80_gemm_1(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<32, 32, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			2,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}

// small 2 case 8
inline cudaError_t sm80_gemm_2(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<32, 32, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			3,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}

// small 3 case 9
inline cudaError_t sm80_gemm_3(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<32, 32, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			4,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}

// medium 1 case 4
inline cudaError_t sm80_gemm_4(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<128, 128, 512>,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			2,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}

// medium 2 case 5
inline cudaError_t sm80_gemm_5(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<128, 128, 512>,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			3,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}

// medium 3 case 6
inline cudaError_t sm80_gemm_6(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<128, 128, 512>,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			4,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}

// large 1, case 1
inline cudaError_t sm80_gemm_7(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<128, 256, 512>,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			2,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}

// large 2 case 2
inline cudaError_t sm80_gemm_8(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<128, 256, 512>,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			3,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}

// large 3 case 3
inline cudaError_t sm80_gemm_9(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm =
		cutlass::gemm::device::Gemm<
			ElementInput1Bit,
			LayoutInputA,
			ElementInput1Bit,
			LayoutInputB,
			ElementOutput,
			LayoutOutput,
			ElementAccumulator,
			MMAOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<128, 256, 512>,
			cutlass::gemm::GemmShape<64, 64, 512>,
			cutlass::gemm::GemmShape<16, 8, 256>,
			EpilogueOp,
			SwizzleThreadBlock,
			4,
			128,
			128,
			false,
			cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}
#endif // end of ifdef ARCH_SM_80

// ====================== ARCH_SM_75 ====================== //
#ifdef ARCH_SM_75 // sm_75
// large 1
inline cudaError_t sm75_gemm_1(
	cutlass::gemm::GemmCoord problem_size,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputA> tensor_a,
	cutlass::TensorRef<ElementInput1Bit, LayoutInputB> tensor_b,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c,
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_ref_c,
	ElementComputeEpilogue alpha_e,
	ElementComputeEpilogue beta_e,
	int split_k_slices) {

	using CutlassGemm = cutlass::gemm::device::Gemm<
		ElementInput1Bit,
		LayoutInputA,
		ElementInput1Bit,
		LayoutInputB,
		ElementOutput,
		LayoutOutput,
		ElementAccumulator,
		MMAOp,
		cutlass::arch::Sm75,
		cutlass::gemm::GemmShape<128, 256, 512>,
		cutlass::gemm::GemmShape<64, 64, 512>,
		cutlass::gemm::GemmShape<8, 8, 128>,
		EpilogueOp,
		SwizzleThreadBlock,
		4,
		128,
		128,
		false,
		cutlass::arch::OpXorPopc>;
	// Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
	// instantiated CUTLASS kernel
	typename CutlassGemm::Arguments args{
		problem_size,           // <- problem size of matrix multiplication
		tensor_a,               // <- reference to matrix A on device
		tensor_b,               // <- reference to matrix B on device
		tensor_c,               // <- reference to matrix C on device
		tensor_ref_c,           // <- reference to matrix C on device
		{alpha_e, beta_e},      // <- tuple of alpha and beta
		split_k_slices          // <- k-dimension split factor
	};

	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	// Allocate workspace memory
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	// Define a CUTLASS GEMM type
	CutlassGemm gemm_operator;

	// Check the problem size is supported or not
	cutlass::Status status = gemm_operator.can_implement(args);
	CUTLASS_CHECK(status);

	// Initialize CUTLASS kernel with arguments and workspace pointer
	status = gemm_operator.initialize(args, workspace.get());
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_operator();
	CUTLASS_CHECK(status);

	//
	// Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	//
	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
    }

	// Return success, if no errors were encountered.
	return cudaSuccess;
}
#endif // end of ifdef ARCH_SM_75

// ======================================================== //
//                       batched Gemm                       //
// =========================================================//

// ====================== ARCH_SM_80 ====================== //
// 1
#ifdef ARCH_SM_80 // sm_80
using sm80_batched_gemm_1 =
	cutlass::gemm::device::GemmBatched<
		ElementInput1Bit,     // Data-type of A matrix
		LayoutInputA,      // Layout of A matrix
		ElementInput1Bit,     // Data-type of B matrix
		LayoutInputB,      // Layout of B matrix
		ElementOutput,     // Data-type of C matrix
		LayoutOutput,      // Layout of C matrix
		ElementAccumulator,
		MMAOp,
		cutlass::arch::Sm80,
		cutlass::gemm::GemmShape<128, 256, 512>,
		cutlass::gemm::GemmShape<64, 64, 512>,
		cutlass::gemm::GemmShape<16, 8, 256>,
		EpilogueOp,
		SwizzleThreadBlockBatched,
		4,
		128,
		128,
		cutlass::arch::OpXorPopc>;
#endif // end of ifdef ARCH_SM_80

// ====================== ARCH_SM_75 ====================== //
#ifdef ARCH_SM_75 // sm_75
// 1
using sm75_batched_gemm_1 =
	cutlass::gemm::device::GemmBatched<
		ElementInput1Bit,     // Data-type of A matrix
		LayoutInputA,      // Layout of A matrix
		ElementInput1Bit,     // Data-type of B matrix
		LayoutInputB,      // Layout of B matrix
		ElementOutput,     // Data-type of C matrix
		LayoutOutput,      // Layout of C matrix
		ElementAccumulator,
		MMAOp,
		cutlass::arch::Sm75,
		cutlass::gemm::GemmShape<128, 256, 512>,
		cutlass::gemm::GemmShape<64, 64, 512>,
		cutlass::gemm::GemmShape<8, 8, 128>,
		EpilogueOp,
		SwizzleThreadBlockBatched,
		2,
		128,
		128,
		cutlass::arch::OpXorPopc>;
#endif // end of ifdef ARCH_SM_75

} // namespace xnor_cutlass