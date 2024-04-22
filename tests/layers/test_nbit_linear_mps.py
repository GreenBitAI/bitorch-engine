import time

import pytest
import torch
import numpy as np

from bitorch_engine.layers.qlinear.nbit.mps import MPQLinearMlx
from tests.layers.util import get_mps_test_device

"""
    Test nbit inference layers
"""

# lower print threshold
torch.set_printoptions(threshold=100)


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available, skipping MPS-related test")
@pytest.mark.parametrize("w_bit", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("num_input_features", [1024, 8192])
@pytest.mark.parametrize("num_hidden_fc", [1024, 8192])
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("llama_v", [2])
@pytest.mark.parametrize("model_size", ["1.1b", "7b", "30b", "70b"])
def test_mpq_linear_mps(w_bit, dtype, num_input_features, num_hidden_fc, batch_size, group_size, llama_v, model_size):
    import mlx.core

    if ((w_bit == 1 or w_bit == 8) and group_size == 32) \
            or (model_size in ["30b", "70b"] and w_bit != 2) \
            or (model_size in ["1.1b"] and w_bit != 2):
        pytest.skip()

    # === following configuration from low_bit_llama https://github.com/GreenBitAI/low_bit_llama === #
    double_groupsize = -1
    if group_size == 32 and model_size not in ["1.1b", "1.1B"]:
        asym = True
    else:
        asym = False

    if w_bit == 2:
        if asym:
            double_groupsize = -1
        else:
            if group_size == 32:
                double_groupsize = 32
            else:
                if llama_v == 1 and model_size not in ["30b", "30B"]:
                    double_groupsize = 64
                else:
                    double_groupsize = 32
    else:
        if model_size in ["3b", "3B"]:
            double_groupsize = 64
        elif model_size in ["7b", "7B"]:
            double_groupsize = 256

    v1 = (llama_v == 1) and model_size in ["7b", "7B"]

    if w_bit not in [2, 4, 8]:
        print("w_bit not supported")
        pytest.skip()
    if group_size not in [32, 64, 128]:
        print("group_size not supported")
        pytest.skip()
    if asym:
        print("asym not supported")
        pytest.skip()
    # =============================================================================================== #

    print("\nM:{}, N:{}, K:{}, bits:{}, dtype:{}, groupsize:{}, llama_v:{}, model_size:{}."
          .format(batch_size, num_hidden_fc, num_input_features, w_bit, dtype, group_size, llama_v, model_size))

    num_runs = 10
    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_mps_test_device()
    input_data_mps = to_device(input_data, device)
    grad_input_data_mps = to_device(grad_input_data, device)

    time_engine = 0
    for i in range(num_runs):
        b_linear = torch.nn.Linear(num_input_features, num_hidden_fc, bias=False, dtype=dtype)
        b_linear.to(device)
        start_time = time.time()
        result = b_linear(input_data_mps)
        time_engine += time.time() - start_time
    print(f"pytorch linear forward run time (device {device}): {time_engine / num_runs:.6f} s")

    mpq_linear_layer = MPQLinearMlx(in_channels=num_input_features,
                                    out_channels=num_hidden_fc,
                                    w_bit=w_bit,
                                    dtype=dtype,
                                    group_size=group_size,
                                    dq_group_size=double_groupsize,
                                    dq_mode=1 if v1 else 2,
                                    use_gba_quant=True,
                                    asym=asym,
                                    requires_grad=False)

    mpq_linear_layer.to(device)
    mpq_linear_layer.prepare_params()

    time_engine = 0
    time_mlx = 0
    for i in range(num_runs):
        random_int_tensor = torch.randint(1, 1000, size=mpq_linear_layer.qweight.shape,
                                          dtype=mpq_linear_layer.qweight.dtype, device=mpq_linear_layer.qweight.device)
        mpq_linear_layer.set_qweight_data(random_int_tensor)
        start_time = time.time()
        result1 = mpq_linear_layer(input_data_mps)
        time_engine += time.time() - start_time

        qweight_mlx = mlx.core.array(mpq_linear_layer.qweight.numpy().astype(np.uint32))
        scales_mlx = mlx.core.array(mpq_linear_layer.scales.numpy())
        zeros_mlx = mlx.core.array(mpq_linear_layer.zeros.numpy())

        start_time = time.time()
        input_mlx = mlx.core.array(input_data_mps.cpu().numpy())
        mlx_matmul = mlx.core.quantized_matmul(
            input_mlx,
            qweight_mlx,
            scales=scales_mlx,
            biases=zeros_mlx,
            transpose=True,
            group_size=group_size,
            bits=w_bit)
        mlx_matmul = torch.from_numpy(np.array(mlx_matmul)).to('mps')
        time_mlx += time.time() - start_time
        assert mlx_matmul.equal(result1), "mps matmul failed"
    print(f"bitorch-engine mpq linear forward (MPS) run time: {time_engine / num_runs:.6f} s")
    print(f"Mlx (MPS) run time: {time_mlx / num_runs:.6f} s")