import pytest
import torch
import torch.nn.functional as F
import numpy as np
import time
from bitorch.layers import QConv2d
from bitorch_engine.utils.quant_operators import get_binary_row
from bitorch_engine.layers.qconv.binary.cpp import BinaryConv2dCPP
from tests.layers.util import get_cuda_test_device

if torch.cuda.is_available():
    from bitorch_engine.layers.qconv.binary.cutlass import BinaryConv2dCutlass

"""
    Test binary inference layers
"""

# lower print threshold
torch.set_printoptions(threshold=10)


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

# Input shape: (batch size, num of input channels, h, w)
TEST_INPUT_DATA = [
    ((8, 128, 128, 128), [128, 64],
        {"kernel_size": 1, "weight_quantization": "sign", "input_quantization": "sign", "padding": 0,
         "stride": 1, "dilation": 1}),
    ((128, 32, 64, 64), [32, 64],
     {"kernel_size": 1, "weight_quantization": "sign", "input_quantization": "sign", "padding": 3,
      "stride": 1, "dilation": 2}),
    ((8, 64, 16, 16), [64, 128],
     {"kernel_size": 1, "weight_quantization": "sign", "input_quantization": "sign", "padding": 0,
      "stride": 1, "dilation": 1}),
    ((1, 32, 64, 64), [32, 32],
     {"kernel_size": 3, "weight_quantization": "sign", "input_quantization": "sign", "padding": 0,
      "stride": 2, "dilation": 1}),
    ((64, 64, 128, 128), [64, 128],
     {"kernel_size": 3, "weight_quantization": "sign", "input_quantization": "sign", "padding": 0,
      "stride": 2, "dilation": 1}),
    ((256, 32, 128, 128), [32, 32],
     {"kernel_size": 3, "weight_quantization": "sign", "input_quantization": "sign", "padding": 0,
      "stride": 2, "dilation": 1}),
    ((16, 128, 8, 8), [128, 32],
     {"kernel_size": 5, "weight_quantization": "sign", "input_quantization": "sign", "padding": 1,
      "stride": 2, "dilation": 1}),
    ((16, 64, 128, 128), [64, 32],
     {"kernel_size": 7, "weight_quantization": "sign", "input_quantization": "sign", "padding": 1,
      "stride": 2, "dilation": 1})
]

TEST_INPUT_DATA_CUTLASS = [
    ((1, 1024, 64, 64), [1024, 32],
     {"kernel_size": 3, "padding": 2, "stride": 2, "dilation": 1}),
    ((128, 1024, 56, 56), [1024, 64],
    {"kernel_size": 1, "padding": 0, "stride": 2, "dilation": 1}),
    ((8, 1024, 64, 64), [1024, 64],
     {"kernel_size": 3, "padding": 1, "stride": 1, "dilation": 1}),
    ((1, 2048, 64, 64), [2048, 64],
     {"kernel_size": 1, "padding": 0, "stride": 1, "dilation": 1}),
    ((8, 4096, 128, 128), [4096, 128],
     {"kernel_size": 5, "padding": 1, "stride": 1, "dilation": 1})
]


@pytest.mark.parametrize("input_shape, args, kwargs", TEST_INPUT_DATA)
def test_binary_conv_inference_cpu(input_shape, args, kwargs):
    print("\n")
    print("input_shape:{}, intput/output nums:{}, other args:{}".format(input_shape, args, kwargs))
    bits_binary_word = 8
    num_runs = 10
    input = np.random.uniform(-1, 1, input_shape)
    layer = QConv2d(*args, **kwargs)
    input_tensor = torch.tensor(input).float()

    start_time = time.time()
    for i in range(num_runs):
        result_bitorch = layer(input_tensor)
    time_engine = time.time() - start_time
    # print("bitorch binary conv: %.6f s" % (time_engine / num_runs))

    weight = layer.weight.clone()
    padding = kwargs["padding"]
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    dilation = kwargs["dilation"]

    binary_conv_layer = BinaryConv2dCPP(in_channels=args[0],
                                        out_channels=args[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation)
    binary_conv_layer.set_weight_data(weight)
    result_bitorch_engine = binary_conv_layer(input_tensor)

    # NOTE: we don't do this anymore
    # convert [-1,+1] to [0,1] results
    # scale_range = input_shape[1] * kernel_size * kernel_size
    # result_bitorch = (scale_range - result_bitorch) / 2

    # test binarized weights
    size_binary = int(weight.nelement()//bits_binary_word)
    binarized_row = torch.zeros(size_binary, dtype=torch.uint8)
    binarized_row = get_binary_row(weight.reshape(-1, ), binarized_row, weight.nelement(), bits_binary_word)
    binary_conv_layer.qweight = binarized_row

    start_time = time.time()
    for i in range(num_runs):
        result_bitorch_engine_binarized_weight = binary_conv_layer(input_tensor)
    time_engine = time.time() - start_time
    # print("engine binary conv: %.6f s" % (time_engine / num_runs))

    assert torch.equal(result_bitorch, result_bitorch_engine)
    assert torch.equal(result_bitorch_engine, result_bitorch_engine_binarized_weight)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("input_shape, args, kwargs", TEST_INPUT_DATA_CUTLASS)
def test_binary_conv_cutlass(input_shape, args, kwargs):
    print("\n")
    print("input_shape:{}, input/output nums:{}, other args:{}".format(input_shape, args, kwargs))
    num_runs = 10
    input = np.random.uniform(-1, 1, input_shape)
    input_tensor = torch.tensor(input).float()

    # to gpu
    device = get_cuda_test_device()
    input_tensor_cuda = to_device(input_tensor, device)
    layer = QConv2d(*args, **kwargs)
    layer.to(device)
    weight_cuda = layer.weight.clone()

    start_time = time.time()
    for i in range(num_runs):
        result_bitorch = layer(input_tensor_cuda)
    time_engine = time.time() - start_time
    # print("bitorch binary conv: %.6f s" % (time_engine / num_runs))

    padding = kwargs["padding"]
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    dilation = kwargs["dilation"]

    binary_conv_cutlass = BinaryConv2dCutlass(in_channels=int(args[0]),
                                        out_channels=args[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        device=device)
    binary_conv_cutlass.to(device)
    binary_conv_cutlass.set_weight_data(weight_cuda)

    result_unpacked = binary_conv_cutlass(input_tensor_cuda)
    result_unpacked = F.layer_norm(result_unpacked, normalized_shape=result_unpacked.shape)
    # print(result_unpacked)

    binary_conv_cutlass.generate_quantized_weight(qweight_only=True)
    binary_conv_cutlass.eval()
    start_time = time.time()
    for i in range(num_runs):
        result_packed = binary_conv_cutlass(input_tensor_cuda)
    time_engine = time.time() - start_time
    result_packed = F.layer_norm(result_packed, normalized_shape=result_packed.shape)
    # print("engine binary conv: %.6f s" % (time_engine / num_runs))
    # print(result_packed)
    assert torch.equal(result_unpacked, result_packed)

