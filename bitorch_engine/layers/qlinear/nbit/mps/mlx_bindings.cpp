#include <torch/extension.h>
#include "mpq_linear_mlx.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mpq_forward", &mpq_linear_mlx_forward, "Forward call for mlx acclelerated MPS quantized linear layer.");
}