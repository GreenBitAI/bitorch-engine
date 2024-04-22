#include <mlx/mlx.h>
#include <mlx/device.h>
#include <mlx/dtype.h>
#include <torch/extension.h>
#include <vector>

typedef float16_t half_t;

template <typename T>
mlx::core::array mlx_from_torch_tensor(torch::Tensor tensor, mlx::core::Dtype valueDtype)
{
	return mlx::core::array(
		reinterpret_cast<T *>(tensor.mutable_data_ptr()),   // Pointer to the data.
		std::vector<int32_t>({                              // Shape of the array as a vector of int32 (mlx requires 32 bit int).
			static_cast<int32_t>(tensor.size(0)),
			static_cast<int32_t>(tensor.size(1))
		}),
		valueDtype                                          // Set dtype.
	);
}

torch::Tensor mlx_to_torch_tensor(mlx::core::array arr, torch::Dtype dtype)
{
	return torch::from_blob(
		arr.data<void>(),                           // Pointer to the data.
		{                                           // Shape of the array as a vector of int64 (torch requires longs).
			static_cast<int64_t>(arr.shape()[0]),
			static_cast<int64_t>(arr.shape()[1])
		},
		torch::TensorOptions().dtype(dtype));       // Set dtype.
}


/**
 * Performs forward pass for mixed precision quantized (MPQ) linear multiplication using a custom MLX library.
 * This function processes an input tensor `x` with quantized weights `qweight`, applying scale factors `scales`
 * and zero points `zeros` for quantization, within specified group sizes and weight bit precision. It's designed
 * for CPU execution, enforcing inputs and computations to reside on the CPU.
 *
 * @param x The input tensor, expected to be on CPU, containing the features to be processed.
 * @param qweight The quantized weights tensor, also on CPU, to be used in the matrix multiplication.
 * @param scales The scale factors for the quantized weights, aiding in de-quantization to real values during the computation.
 * @param zeros The zero points for the quantized weights, also used in de-quantization process.
 * @param group_size The size of groups for performing the quantized matrix multiplication, affecting how inputs are partitioned.
 * @param w_bit The precision of the quantized weights in bits, supporting 2, 4, or 8 bits for the computation.
 *
 * @details
 * - The function begins by validating the input tensors to ensure they are CPU tensors, contiguous, and that the weight bits
 *   are within the supported range.
 * - It then converts PyTorch tensors to MLX core arrays, using appropriate data types for the MLX backend computations.
 * - Quantized matrix multiplication is performed with the MLX library, leveraging the given scale factors, zero points,
 *   group size, and weight bit precision.
 * - The MLX library uses lazy evaluation for computations; thus, the function explicitly evaluates the output before
 *   converting it back to a PyTorch tensor.
 * - The result is a tensor of the computed output in float16 format, ready for further processing in PyTorch pipelines.
 *
 * @return A torch::Tensor containing the result of the mixed precision quantized linear multiplication in float16 format.
 *
 * @note
 * - This function is designed to operate exclusively on CPU tensors and will verify the device type of its inputs.
 * - It assumes the MLX backend for computation, which requires inputs to be converted to MLX core arrays.
 * - The use of float16 and uint32 data types for MLX core arrays is based on the precision and requirements of the inputs and computation.
 */
torch::Tensor mpq_linear_mlx_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size,
    int w_bit)
{
	// Check the input parameters
	TORCH_CHECK((w_bit == 2) || (w_bit == 4) || (w_bit == 8), "weights must have {2,4,8} bits.");
	TORCH_CHECK(x.device().type() == torch::kCPU, "x must be a CPU tensor. Cannot read from MPS.");
	TORCH_CHECK(qweight.device().type() == torch::kCPU, "qweight must be a CPU tensor. Cannot read from MPS.");
	TORCH_CHECK(scales.device().type() == torch::kCPU, "scales must be a CPU tensor. Cannot read from MPS.");
	TORCH_CHECK(zeros.device().type() == torch::kCPU, "zeros must be a CPU tensor. Cannot read from MPS.");
	TORCH_CHECK(x.is_contiguous(), "x must be contiguous.");
	TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous.");
	TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous.");
	TORCH_CHECK(zeros.is_contiguous(), "zeros must be contiguous.");

	mlx::core::array x_arr = mlx_from_torch_tensor<half_t>(x, mlx::core::float16);
	mlx::core::array qweight_arr = mlx_from_torch_tensor<uint32_t>(qweight, mlx::core::uint32);
	mlx::core::array scales_arr = mlx_from_torch_tensor<half_t>(scales, mlx::core::float16);
	mlx::core::array zeros_arr = mlx_from_torch_tensor<half_t>(zeros, mlx::core::float16);
	mlx::core::array output = mlx::core::quantized_matmul(
		x_arr,
		qweight_arr,
		scales_arr,
		zeros_arr,
		true,
		group_size,
		w_bit,
		mlx::core::default_stream(mlx::core::default_device())
	);

	mlx::core::eval(output); // Mlx uses lazy evaluation. Run and wait for the computation to finish.
	return mlx_to_torch_tensor(output, torch::kFloat16);
}