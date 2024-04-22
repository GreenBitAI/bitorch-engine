#include <torch/extension.h>

torch::Tensor mpq_linear_mlx_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size,
    int w_bit);