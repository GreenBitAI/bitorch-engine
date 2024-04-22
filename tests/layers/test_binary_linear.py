import pytest
import torch
import time
import math
from bitorch.layers import QLinear

from torch.nn import Parameter
from bitorch_engine.utils.quant_operators import get_binary_row, get_binary_col
from bitorch_engine.layers.qlinear.binary.cpp import BinaryLinearCPP as BinaryLinearCPP
from tests.layers.util import to_device, get_cuda_test_device, get_cuda_test_device_id
from bitorch_engine.utils.quant_operators import nv_tensor_quant

if torch.cuda.is_available():
    from bitorch_engine.layers.qlinear.binary.cuda import BinaryLinearCuda as BinaryLinearCuda, BMM
    from bitorch_engine.layers.qlinear.binary.cutlass import BinaryLinearCutlass, BinaryMatMul

"""
    Test binary inference layers
"""

# lower print threshold
torch.set_printoptions(threshold=100)


@pytest.mark.parametrize("num_input_features", [64, 512, 1024])
@pytest.mark.parametrize("num_hidden_fc", [20, 128])
@pytest.mark.parametrize("batch_size", [1, 64, 128])
def test_cpu_kernel(num_input_features, num_hidden_fc, batch_size):
    # setup data
    bits_binary_word = 8
    print("num_input_features:{}, num_hidden:{}, batch_size:{}".format(num_input_features, num_hidden_fc, batch_size))
    num_runs = 10

    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features))
    weight = torch.normal(0, 1, size=(num_hidden_fc, num_input_features))

    # get QLinear output
    qlayer = QLinear(num_input_features, num_hidden_fc, bias=False, weight_quantization="sign", input_quantization="sign")
    qlayer.weight = Parameter(weight)

    start_time = time.time()
    for i in range(num_runs):
        y_dot = qlayer(input_data)
    time_engine = time.time() - start_time
    print("bitorch binary linear: %.6f s" % (time_engine / num_runs))
    # print("QLinear out:")
    # print(y_dot)

    ## run c++ xor inference
    binary_layer = BinaryLinearCPP(num_input_features, num_hidden_fc)
    binary_layer.weight = Parameter(weight)
    xor_output = binary_layer(input_data)
    # print("C++ XOR out:")
    # print(xor_output)

    binary_layer.generate_quantized_weight()

    start_time = time.time()
    for i in range(num_runs):
        binarized_w_output = binary_layer(input_data)
    time_engine = time.time() - start_time
    print("engine binary linear: %.6f s" % (time_engine / num_runs))

    assert torch.equal(y_dot, xor_output)
    assert torch.equal(xor_output, binarized_w_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("num_input_features", [128, 640, 896, 2048])
@pytest.mark.parametrize("num_hidden_fc", [8, 16, 24, 32, 40, 384, 2048])
@pytest.mark.parametrize("batch_size", [8, 16, 24, 32, 40, 128, 2048])
def test_combined_cuda_kernel_output(dtype, num_input_features, num_hidden_fc, batch_size):
    print("\nM:{}, N:{}, K:{}, dtype:{}.".format(batch_size, num_hidden_fc, num_input_features, dtype))
    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False, dtype=dtype)
    weight = torch.normal(0, 1, size=(num_hidden_fc, num_input_features), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)

    input_data_cuda = to_device(input_data, device)
    grad_input_data_cuda = to_device(grad_input_data, device)
    weight_cuda = to_device(weight, device)

    # get QLinear output
    qlayer = QLinear(num_input_features, num_hidden_fc, bias=False, weight_quantization="sign", input_quantization="sign",
                     dtype=dtype)
    qlayer.to(device)
    weight_cuda_b = weight_cuda - weight_cuda.mean()
    weight_int8, scale_w = nv_tensor_quant(weight_cuda_b)
    weight_int8 = torch.where(weight_int8 == 0, weight_cuda_b.sign(), weight_int8)
    qlayer.weight = Parameter(weight_int8.to(dtype))
    y_dot = qlayer(input_data_cuda)
    # print("CUDA QLinear out:")
    # print(y_dot)

    binary_layer_cuda = BinaryLinearCuda(num_input_features, num_hidden_fc, device=device, dtype=dtype)
    binary_layer_cuda.set_weight_data(weight_cuda)
    binary_layer_cuda.to(device)
    output_combined = binary_layer_cuda(input_data_cuda, bmm_type=BMM.ADAPTIVE)
    # print("CUDA combined kernel out:")
    # print(output_combined)

    output_combined.backward(grad_input_data_cuda)
    torch.cuda.synchronize()

    assert torch.equal(y_dot*binary_layer_cuda.scale_a*binary_layer_cuda.scale_w, output_combined)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("num_input_features", [128, 640, 896, 1024, 2048])
@pytest.mark.parametrize("num_hidden_fc", [8, 16, 32, 40, 128, 1024, 2048])
@pytest.mark.parametrize("batch_size", [128, 1024, 2048])
def test_cuda_btc_output(dtype, num_input_features, num_hidden_fc, batch_size):
    print("\nM:{}, N:{}, K:{}, dtype:{}.".format(batch_size, num_hidden_fc, num_input_features, dtype))
    # setup data
    bits_binary_word = 8

    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False, dtype=dtype)
    weight = torch.normal(0, 1, size=(num_hidden_fc, num_input_features), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)

    input_data_cuda = to_device(input_data, device)
    grad_input_data_cuda = to_device(grad_input_data, device)
    weight_cuda = to_device(weight, device)

    # get QLinear output
    qlayer = QLinear(num_input_features, num_hidden_fc, bias=False, weight_quantization="sign", input_quantization="sign",
                     dtype=dtype)
    qlayer.to(device)
    weight_cuda_b = weight_cuda - weight_cuda.mean()
    weight_int8, scale_w = nv_tensor_quant(weight_cuda_b)
    weight_int8 = torch.where(weight_int8 == 0, weight_cuda_b.sign(), weight_int8)
    qlayer.weight = Parameter(weight_int8.to(dtype))
    y_dot = qlayer(input_data_cuda)
    y_dot = y_dot.clone()
    # print("CUDA QLinear out:")
    # print(y_dot)

    binary_layer_cuda = BinaryLinearCuda(num_input_features, num_hidden_fc, device=device, dtype=dtype)
    binary_layer_cuda.set_weight_data(weight_cuda)
    binary_layer_cuda.to(device)

    output_tensorcore = binary_layer_cuda(input_data_cuda, bmm_type=BMM.BTC32)
    # print("CUDA BTC out:")
    # print(output_tensorcore)

    output_tensorcore.backward(grad_input_data_cuda)
    torch.cuda.synchronize()
    # print(binary_layer_cuda.weight.grad)

    assert torch.equal((y_dot*binary_layer_cuda.scale_a*binary_layer_cuda.scale_w), output_tensorcore)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("num_input_features", [32, 64, 256, 2048, 4096])
@pytest.mark.parametrize("num_hidden_fc", [8, 32, 40, 512, 2048, 4096])
@pytest.mark.parametrize("batch_size", [32, 64, 128, 256, 512, 2048])
def test_cuda_BSTC_output(dtype, num_input_features, num_hidden_fc, batch_size):
    print("\nM:{}, N:{}, K:{}, dtype:{}.".format(batch_size, num_hidden_fc, num_input_features, dtype))
    # setup data
    bits_binary_word = 8
    num_runs = 1
    # dtype = torch.half

    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False, dtype=dtype)
    weight = torch.normal(0, 1, size=(num_hidden_fc, num_input_features), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)

    input_data_cuda = to_device(input_data, device)
    grad_input_data_cuda = to_device(grad_input_data, device)
    weight_cuda = to_device(weight, device)

    # get QLinear output
    qlayer = QLinear(num_input_features, num_hidden_fc, bias=False, weight_quantization="sign", input_quantization="sign",
                     dtype=dtype)
    qlayer.to(device)

    weight_cuda_b = weight_cuda - weight_cuda.mean()
    weight_int8, scale_w = nv_tensor_quant(weight_cuda_b)
    weight_int8 = torch.where(weight_int8 == 0, weight_cuda_b.sign(), weight_int8).to(torch.int8)
    qlayer.weight = Parameter(weight_int8.to(dtype))

    for i in range(num_runs):
        y_dot = qlayer(input_data_cuda)
    y_dot = y_dot.clone()
    torch.cuda.synchronize()
    # print("CUDA QLinear out:")
    # print(y_dot)

    binary_layer_cuda = BinaryLinearCuda(num_input_features, num_hidden_fc, device=device, dtype=dtype)
    binary_layer_cuda.set_weight_data(weight_cuda)
    binary_layer_cuda.to(device)

    for i in range(num_runs):
        output_bstc32 = binary_layer_cuda(input_data_cuda, bmm_type=BMM.BSTC32)
    torch.cuda.synchronize()
    # print("CUDA BSTC out:")
    # print(output_bstc32)

    output_bstc32.backward(grad_input_data_cuda)
    torch.cuda.synchronize()
    # print(binary_layer_cuda.weight.grad)

    assert torch.equal((y_dot*binary_layer_cuda.scale_a*binary_layer_cuda.scale_w), output_bstc32)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("num_input_features", [512])
@pytest.mark.parametrize("num_hidden_fc", [512])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
def test_cuda_weight_packing(num_input_features, num_hidden_fc, batch_size, dtype):
    print("\nM:{}, N:{}, K:{}, dtype:{}.".format(batch_size, num_hidden_fc, num_input_features, dtype))
    # setup data
    bits_binary_word = 8
    # dtype = torch.float
    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    weight = torch.normal(0, 1, size=(num_hidden_fc, num_input_features), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)

    input_data_cuda = to_device(input_data, device)
    weight_cuda = to_device(weight, device)

    # weight packing test
    binary_layer_cuda = BinaryLinearCuda(num_input_features, num_hidden_fc, device=device, dtype=dtype)
    binary_layer_cuda.set_weight_data(weight_cuda) # it will change weight to int8 weight
    weight_cuda = binary_layer_cuda.weight
    binary_layer_cuda.to(device)

    # BTC
    output_btc_unpacked = binary_layer_cuda(input_data_cuda, bmm_type=BMM.BTC32)
    packed_w_btc = BinaryLinearCuda.w_pack(weight_cuda,
                                              bmm_type=BMM.BTC32)
    binary_layer_cuda.generate_quantized_weight()
    binary_layer_cuda.set_quantized_weight_data(packed_w_btc)
    binary_layer_cuda.training = False
    output_btc_packed = binary_layer_cuda(input_data_cuda, bmm_type=BMM.BTC32)
    assert torch.equal(output_btc_unpacked.data, output_btc_packed.data)

    # BSTC32
    binary_layer_cuda.qweight = None
    binary_layer_cuda.training = True
    output_bstc_unpacked = binary_layer_cuda(input_data_cuda, bmm_type=BMM.BSTC32)
    packed_w_bstc = BinaryLinearCuda.w_pack(weight_cuda,
                                              bmm_type=BMM.BSTC32)
    binary_layer_cuda.set_quantized_weight_data(packed_w_bstc)
    binary_layer_cuda.training = False
    output_bstc_packed = binary_layer_cuda(input_data_cuda, bmm_type=BMM.BSTC32)
    assert torch.equal(output_bstc_unpacked, output_bstc_packed)


@pytest.mark.parametrize("num_input_features", [64, 512, 1024])
@pytest.mark.parametrize("num_hidden_fc", [512, 1000])
@pytest.mark.parametrize("batch_size", [1, 32, 128, 512])
def test_cpu_weight_packing(num_input_features, num_hidden_fc, batch_size):
    # setup data
    bits_binary_word = 8

    print("num_input_features:{}, num_hidden:{}, batch_size:{}".format(num_input_features, num_hidden_fc, batch_size))

    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features))
    weight = torch.normal(0, 1, size=(num_hidden_fc, num_input_features))

    # get QLinear output
    qlayer = QLinear(num_input_features, num_hidden_fc, bias=False, weight_quantization="sign", input_quantization="sign")
    qlayer.weight = Parameter(weight)
    # start_time = time.time()
    y_dot = qlayer(input_data)
    # time_dot = time.time() - start_time
    # print("1-bit dot run time: %.6f s" % time_dot)

    ## run python version of "get_binary_col(...)"
    size_binary = int(num_input_features//bits_binary_word)
    binary_col = torch.zeros((num_hidden_fc * size_binary), dtype=torch.uint8)
    start_time = time.time()
    binary_col = get_binary_col(weight.transpose(0, 1).reshape(-1,), binary_col, num_input_features, num_hidden_fc, bits_binary_word)
    time_dot = time.time() - start_time
    print("py-binary_col run time: %.6f s" % time_dot)

    ## run c++ xor inference
    binary_layer = BinaryLinearCPP(num_input_features, num_hidden_fc)
    binary_layer.set_weight_data(weight)
    assert torch.equal(binary_layer.opt_weight, weight)

    # we test the output of cpp and python version of get_binary_col
    start_time = time.time()
    binary_layer.generate_quantized_weight() # will call the cpp weight packing function
    time_dot = time.time() - start_time
    print("cpp-binary_col run time: %.6f s" % time_dot)
    assert torch.equal(binary_layer.qweight, binary_col)

    # start_time = time.time()
    xor_output = binary_layer(input_data)
    # time_xor = time.time() - start_time
    # print("C++ XOR unpacked weight run time: %.6f s" % time_xor)

    # using bit-packed weights
    # start_time = time.time()
    xor_output_binarized_weights = binary_layer(input_data)
    # time_xor_binarized = time.time() - start_time
    # print("C++ XOR binarized weight run time: %.6f s" % time_xor_binarized)

    assert torch.equal(y_dot, xor_output)
    assert torch.equal(xor_output_binarized_weights, xor_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("num_input_features", [768, 1024, 1536, 3072, 8192, 12288])
@pytest.mark.parametrize("num_hidden_fc", [768, 1024, 1536, 3072, 8192, 12288])
@pytest.mark.parametrize("batch_size", [512, 1024, 2048, 4096])
def test_binary_linear_cutlass(num_input_features, num_hidden_fc, batch_size):
    print("\nM:{}, N:{}, K:{}.".format(batch_size, num_hidden_fc, num_input_features))
    num_runs = 100

    dtype = torch.float
    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)

    input_data_cuda = to_device(input_data, device)
    grad_input_data_cuda = to_device(grad_input_data, device)

    linear = torch.nn.Linear(num_input_features, num_hidden_fc, bias=False, dtype=dtype)
    linear.to(device)
    start_time = time.time()
    for i in range(num_runs):
        result = linear(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("pytorch linear forward run time: %.6f s" % (time_engine/num_runs))

    # run CUTLASS
    binary_layer_cutlass = BinaryLinearCutlass(num_input_features, num_hidden_fc, device=device, dtype=dtype)
    binary_layer_cutlass.to(device)
    binary_layer_cutlass.prepare_params()
    weig = binary_layer_cutlass.weight.data
    start_time = time.time()
    for i in range(num_runs):
        result_cutlass = binary_layer_cutlass(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("bitorch-engine 1-bit linear forward (CUTLASS) run time: %.6f s" % (time_engine/num_runs))
    # print(result_cutlass)

    binary_layer_cuda = BinaryLinearCuda(num_input_features, num_hidden_fc, device=device, bmm_type=BMM.BTC32)
    binary_layer_cuda.to(device)
    binary_layer_cuda.prepare_params()
    binary_layer_cuda.weight.data = weig
    binary_layer_cuda.scale_a = binary_layer_cutlass.scale_a
    binary_layer_cuda.scale_w = binary_layer_cutlass.scale_w

    assert torch.equal(binary_layer_cutlass.weight.data, binary_layer_cuda.weight.data)

    start_time = time.time()
    for i in range(num_runs):
        result_cuda = binary_layer_cuda(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("bitorch-engine 1-bit linear forward (CUDA) run time: %.6f s" % (time_engine/num_runs))
    # print(result_cuda)

    assert torch.all(torch.isclose(result_cutlass, result_cuda, rtol=1e-05, atol=1e-08, equal_nan=False))

    # weight packing test
    binary_layer_cutlass.eval()
    binary_layer_cutlass.generate_quantized_weight()
    result_cutlass_packed = binary_layer_cutlass(input_data_cuda)
    # print(result_cutlass_packed)
    assert torch.equal(result_cutlass, result_cutlass_packed)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("num_input_features", [8192, 12288])
@pytest.mark.parametrize("num_hidden_fc", [8192, 12288])
@pytest.mark.parametrize("batch_size", [2048])
def test_binary_linear_backward(num_input_features, num_hidden_fc, batch_size):
    print("\nM:{}, N:{}, K:{}.".format(batch_size, num_hidden_fc, num_input_features))

    dtype = torch.float
    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)

    input_data_cuda = to_device(input_data, device)
    grad_input_data_cuda = to_device(grad_input_data, device)

    # run CUTLASS
    binary_layer_cutlass = BinaryLinearCutlass(num_input_features, num_hidden_fc, device=device, dtype=dtype)
    binary_layer_cutlass.to(device)
    binary_layer_cutlass.prepare_params()
    weig = binary_layer_cutlass.weight.data
    result_cutlass = binary_layer_cutlass(input_data_cuda)

    binary_layer_cuda = BinaryLinearCuda(num_input_features, num_hidden_fc, device=device, bmm_type=BMM.BTC32)
    binary_layer_cuda.to(device)
    binary_layer_cutlass.prepare_params()
    binary_layer_cuda.weight.data = weig
    result_cuda = binary_layer_cuda(input_data_cuda)

    num_runs = 100
    start_time = time.time()
    for i in range(num_runs):
        result_cutlass.backward(grad_input_data_cuda, retain_graph=True)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    # print(binary_layer_cutlass.weight.grad)
    print("bitorch-engine binary linear backward (CUTLASS) run time: %.6f s" % (time_engine/num_runs))

    start_time = time.time()
    for i in range(num_runs):
        result_cuda.backward(grad_input_data_cuda, retain_graph=True)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    # print(binary_layer_cuda.weight.grad)
    print("bitorch-engine binary linear backward (CUDA) run time: %.6f s" % (time_engine/num_runs))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("hidden_size", [768, 1020, 1536, 2040, 3072, 8196])
@pytest.mark.parametrize("seq_length", [128, 256])
@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("head_num", [12])
def test_binary_matmul_cutlass(hidden_size, seq_length, batch_size, head_num):
    print("\nbatch:{}, seq_length:{}, hidden_size:{}, head_num: {}."
          .format(batch_size, seq_length, hidden_size, head_num))
    num_runs = 100

    dtype = torch.float32
    # creating test data
    input_data_k = torch.normal(0, 1, size=(batch_size, head_num, seq_length, int(hidden_size/head_num)),
                                requires_grad=False, dtype=dtype)
    input_data_v = torch.normal(0, 1, size=(batch_size, head_num, seq_length, int(hidden_size/head_num)),
                                requires_grad=False, dtype=dtype)
    input_data_q = torch.normal(0, 1, size=(batch_size, head_num, seq_length, int(hidden_size/head_num)),
                                requires_grad=False, dtype=dtype)
    grad_input_data = torch.normal(0, 1, size=(batch_size, head_num, seq_length, seq_length), requires_grad=False, dtype=dtype)
    grad_input_data_2 = torch.normal(0, 1, size=(batch_size, head_num, int(hidden_size/head_num), seq_length), requires_grad=False,
                                   dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)

    input_data_k_cuda = to_device(input_data_k, device)
    input_data_v_cuda = to_device(input_data_v, device)
    input_data_q_cuda = to_device(input_data_q, device)
    grad_input_data_cuda = to_device(grad_input_data, device)
    grad_input_data_cuda_2 = to_device(grad_input_data_2, device)

    start_time = time.time()
    for i in range(num_runs):
        attention_score_torch = torch.matmul(input_data_q_cuda, input_data_k_cuda.transpose(-1, -2).contiguous())
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("torch matmul run time: %.6f s" % (time_engine/num_runs))

    # run BinaryMatMul
    cutlass_matmul_1 = BinaryMatMul(dtype=input_data_k_cuda.dtype)
    cutlass_matmul_2 = BinaryMatMul(dtype=input_data_k_cuda.dtype)
    cutlass_matmul_1.to(device)
    cutlass_matmul_2.to(device)

    start_time = time.time()
    for i in range(num_runs):
        # this matmul accepts x_shape:(..., m, k), y_shape(..., n, k), the function will transpose the second input
        attention_scores = cutlass_matmul_1(input_data_q_cuda, input_data_k_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("CUTLASS matmul run time: %.6f s" % (time_engine/num_runs))

    attention_scores.backward(grad_input_data_cuda, retain_graph=True)
    # print(cutlass_matmul_1.x_clip.grad)

    attention_scores = attention_scores / \
                       math.sqrt(head_num)

    # attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    context_layer = cutlass_matmul_2(input_data_v_cuda.transpose(-1, -2), attention_scores)
    # print(context_layer)
    context_layer.backward(grad_input_data_cuda_2, retain_graph=True)
    # print(cutlass_matmul_2.x_clip.grad)
