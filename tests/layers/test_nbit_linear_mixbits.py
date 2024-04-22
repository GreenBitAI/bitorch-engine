import time, math
import pytest
import torch
import json

from bitorch_engine.layers.qlinear.nbit.cuda import MBWQLinearCuda
from tests.layers.util import get_cuda_test_device, get_packed_info, get_q_groups


"""
    Test nbit inference layers
"""

# lower print threshold
torch.set_printoptions(threshold=100)


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


# ========= testing data ========== #

test_json_string = '{' \
                   '"q_proj": { "group_size": { "4": 32, "2": 32 }, "bits": [ 4, 2 ], "bits_prop": [ 0.75, 0.25 ], "scale_bits": 4 }, ' \
                   '"k_proj": { "group_size": { "4": 32, "2": 32 }, "bits": [ 4, 2 ], "bits_prop": [ 0.25, 0.75 ], "scale_bits": 4 } ' \
                   '}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("num_input_features", [128, 4096])
@pytest.mark.parametrize("num_hidden_fc", [128, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mbwq_linear_exl2_cuda(num_input_features, num_hidden_fc, batch_size):

    # support 4-bit und bf16 only!
    dtype = torch.half
    num_runs = 1
    device = get_cuda_test_device()
    torch.cuda.set_device(device)

    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    # to gpu
    input_data_cuda = to_device(input_data, device)

    # Parsing the JSON string into a Python dictionary
    gbe_strategy = json.loads(test_json_string)

    for key, value in gbe_strategy.items():
        # read attribute information from predefined json config
        groups, rows = get_packed_info(num_input_features, value["bits"], value["bits_prop"], value["group_size"])

        print("\nM:{}, N:{}, K:{}, bits:{}, group_size:{}, bits_prop:{}, groups:{}, packed_rows:{}."
              .format(batch_size, num_hidden_fc, num_input_features, str(value["bits"]), str(value["group_size"]), str(value["bits_prop"]),
                      groups, rows))

        # creating int weights
        random_int4_tensor = torch.randint(0, 2 ** 16 - 1, size=(rows, num_hidden_fc))
        int_weight_cuda = to_device(random_int4_tensor, device)

        mbwq_linear_layer = MBWQLinearCuda(in_channels=num_input_features, out_channels=num_hidden_fc,
                                           w_bit=4, dtype=dtype, group_size=32,
                                           dq_group_size=1, use_gba_quant=True,
                                           asym=False, dq_mode=2, use_mbw=True,
                                           groups=groups, rows_packed=rows)

        mbwq_linear_layer.set_qweight_data(int_weight_cuda)
        # random scales and zeros
        scales = torch.randn_like(mbwq_linear_layer.scales).half()
        zeros = torch.randn_like(mbwq_linear_layer.zeros).half()
        mbwq_linear_layer.set_scales(scales)
        mbwq_linear_layer.set_zeros(zeros)
        mbwq_linear_layer.q_perm = torch.tensor([i for i in range(num_input_features)], dtype=torch.short)

        # get q_groups
        q_groups = get_q_groups(groups, value["bits"], value["group_size"], num_input_features, value["bits_prop"])

        assert mbwq_linear_layer.q_groups.numel() == len(q_groups)

        mbwq_linear_layer.q_groups = torch.Tensor(q_groups).to(torch.short)

        # to device
        mbwq_linear_layer.to(device)

        # will perform qweight layout transformation
        mbwq_linear_layer.prepare_params()

        # Testing fp weight reconstruction
        reconstructed_fp_weights = MBWQLinearCuda.exl2fp_weight(mbwq_linear_layer.qweight, mbwq_linear_layer.scales,
                                                                 mbwq_linear_layer.zeros, mbwq_linear_layer.q_perm,
                                                                 mbwq_linear_layer.q_group_map, mbwq_linear_layer.rows).to(dtype)

        # pytorch result:
        result_pt = torch.matmul(input_data_cuda.mul(mbwq_linear_layer.channel_scale), reconstructed_fp_weights)

        # Testing inference output
        start_time = time.time()
        for i in range(num_runs):
            result = mbwq_linear_layer(input_data_cuda)
        torch.cuda.synchronize()
        time_engine = time.time() - start_time

        print("bitorch-engine mbwq_linear (CUDA) run time: %.6f s" % (time_engine/num_runs))

        assert torch.all(torch.isclose(result, result_pt, rtol=2, atol=2, equal_nan=False))