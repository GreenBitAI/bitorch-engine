import time, math
import pytest
import torch
from bitorch_engine.layers.qlinear.nbit.cutlass import Q4LinearCutlass, Q8LinearCutlass, Q4MatMul
from bitorch_engine.layers.qlinear.nbit.cuda import MPQLinearCuda, MBWQLinearCuda
from bitorch_engine.layers.qlinear.nbit.cuda.utils import pack_fp_weight, unpack_qweight
from tests.layers.util import get_cuda_test_device


"""
    Test nbit inference layers
"""

# lower print threshold
torch.set_printoptions(threshold=100)


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("num_input_features", [4096, 8192, 12288])
@pytest.mark.parametrize("num_hidden_fc", [4096, 8192, 12288])
@pytest.mark.parametrize("batch_size", [1024, 4096, 8192, 12288])
def test_q4_linear_cuda(num_input_features, num_hidden_fc, batch_size):
    print("\nM:{}, N:{}, K:{}.".format(batch_size, num_hidden_fc, num_input_features))
    num_runs = 10

    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)
    input_data_cuda = to_device(input_data, device)
    grad_input_data_cuda = to_device(grad_input_data, device)
    b_linear = torch.nn.Linear(num_input_features, num_hidden_fc, bias=False)
    b_linear.to(device)

    start_time = time.time()
    for i in range(num_runs):
        result = b_linear(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("pytorch linear forward run time: %.6f s" %  (time_engine/num_runs))


    q4_linear_layer = Q4LinearCutlass(in_channels=num_input_features,
                                       out_channels=num_hidden_fc,
                                       device=device)
    q4_linear_layer.to(device)
    q4_linear_layer.prepare_params()

    start_time = time.time()
    for i in range(num_runs):
        result1 = q4_linear_layer(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("bitorch-engine 4-bit qlinear forward (CUTLASS) run time: %.6f s" % (time_engine/num_runs))

    # weight packing test
    q4_linear_layer.eval()
    result2 = q4_linear_layer(input_data_cuda)

    assert torch.equal(result1, result2)

    start_time = time.time()
    for i in range(num_runs):
        result1.backward(grad_input_data_cuda, retain_graph=True)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("bitorch-engine 4-bit qlinear backward (CUTLASS) run time: %.6f s" % (time_engine/num_runs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("hidden_size", [384, 768, 1536, 3072, 8448, 12288])
@pytest.mark.parametrize("seq_length", [128, 256])
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("head_num", [12])
def test_q4_matmul_cutlass(hidden_size, seq_length, batch_size, head_num):
    print("\nbatch:{}, seq_length:{}, hidden_size:{}, head_num: {}."
          .format(batch_size, seq_length, hidden_size, head_num))
    # NOTE THAT: the minimal edge size for 4-bit gemm is 32 (128 for 1-bit),
    # therefore hidden_size / head_num >= 32*12=384

    num_runs = 10

    dtype = torch.float
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
    cutlass_matmul_1 = Q4MatMul(dtype=input_data_k_cuda.dtype, device=device)
    cutlass_matmul_2 = Q4MatMul(dtype=input_data_k_cuda.dtype, device=device)
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

    attention_scores = attention_scores / \
                       math.sqrt(head_num)

    # attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    context_layer = cutlass_matmul_2(input_data_v_cuda.transpose(-1, -2), attention_scores)
    context_layer.backward(grad_input_data_cuda_2, retain_graph=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("num_input_features", [4096, 8192])
@pytest.mark.parametrize("num_hidden_fc", [4096, 8192])
@pytest.mark.parametrize("batch_size", [1024, 4096, 8192])
def test_q8_linear_cuda(num_input_features, num_hidden_fc, batch_size):
    print("\nM:{}, N:{}, K:{}.".format(batch_size, num_hidden_fc, num_input_features))
    num_runs = 10

    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)
    input_data_cuda = to_device(input_data, device)
    grad_input_data_cuda = to_device(grad_input_data, device)

    b_linear = torch.nn.Linear(num_input_features, num_hidden_fc, bias=False)
    b_linear.to(device)
    start_time = time.time()
    for i in range(num_runs):
        result = b_linear(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("pytorch linear forward run time: %.6f s" % (time_engine/num_runs))


    q8_linear_layer = Q8LinearCutlass(in_channels=num_input_features,
                                       out_channels=num_hidden_fc,
                                       device=device)
    q8_linear_layer.to(device)
    q8_linear_layer.prepare_params()

    start_time = time.time()
    for i in range(num_runs):
        result1 = q8_linear_layer(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time

    print("bitorch-engine 8-bit qlinear forward (CUTLASS) run time: %.6f s" % (time_engine/num_runs))

    start_time = time.time()
    for i in range(num_runs):
        result1.backward(grad_input_data_cuda, retain_graph=True)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("bitorch-engine 8-bit qlinear backward (CUTLASS) run time: %.6f s" % (time_engine/num_runs))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("w_bit", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
@pytest.mark.parametrize("num_input_features", [8192])
@pytest.mark.parametrize("num_hidden_fc", [8192])
@pytest.mark.parametrize("batch_size", [2048])
@pytest.mark.parametrize("group_size", [8, 32, 128])
@pytest.mark.parametrize("llama_v", [2])
@pytest.mark.parametrize("model_size", ["1.1b", "7b", "30b", "70b"])
def test_mpq_linear_cuda(w_bit, dtype, num_input_features, num_hidden_fc, batch_size, group_size, llama_v, model_size):
    if ((w_bit == 1 or w_bit == 8) and group_size == 32) \
            or (model_size in ["30b", "70b"] and w_bit != 2)\
            or (model_size in ["1.1b"] and w_bit != 2):
        return

    print("\nM:{}, N:{}, K:{}, bits:{}, dtype:{}, groupsize:{}, llama_v:{}, model_size:{}."
          .format(batch_size, num_hidden_fc, num_input_features, w_bit, dtype, group_size, llama_v, model_size))

    num_runs = 1
    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)
    grad_input_data = torch.normal(0, 1, size=(batch_size, num_hidden_fc), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)
    input_data_cuda = to_device(input_data, device)
    grad_input_data_cuda = to_device(grad_input_data, device)

    b_linear = torch.nn.Linear(num_input_features, num_hidden_fc, bias=False, dtype=dtype)
    b_linear.to(device)
    start_time = time.time()
    for i in range(num_runs):
        result = b_linear(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("pytorch linear forward run time: %.6f s" % (time_engine/num_runs))

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
    # =============================================================================================== #

    mpq_linear_layer = MPQLinearCuda(in_channels=num_input_features,
                                     out_channels=num_hidden_fc,
                                     w_bit=w_bit,
                                     dtype=dtype,
                                     group_size=group_size,
                                     dq_group_size=double_groupsize,
                                     dq_mode=1 if v1 else 2,
                                     use_gba_quant=True,
                                     asym=asym)
    mpq_linear_layer.to(device)
    mpq_linear_layer.prepare_params()
    random_int_tensor = torch.randint(1, 1000, size=mpq_linear_layer.qweight.shape,
                                      dtype=mpq_linear_layer.qweight.dtype, device=device)
    mpq_linear_layer.set_qweight_data(random_int_tensor)

    start_time = time.time()
    for i in range(num_runs):
        result1 = mpq_linear_layer(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time

    print("bitorch-engine mpq linear forward (CUDA) run time: %.6f s" % (time_engine/num_runs))

    start_time = time.time()
    for i in range(num_runs):
        result1.backward(grad_input_data_cuda, retain_graph=True)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time

    print("bitorch-engine mpq linear backward (CUDA) run time: %.6f s" % (time_engine/num_runs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("num_input_features", [8192])
@pytest.mark.parametrize("num_hidden_fc", [8192])
@pytest.mark.parametrize("batch_size", [2048])
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
@pytest.mark.parametrize("w_bit", [2, 4])
def test_mbwq_linear_q4_cuda(num_input_features, num_hidden_fc, batch_size, group_size, w_bit):
    # support 4-bit und bf16 only!
    dtype = torch.half
    print("\nM:{}, N:{}, K:{}, bits:{}, dtype:{}, groupsize:{}."
          .format(batch_size, num_hidden_fc, num_input_features, w_bit, dtype, group_size))

    num_runs = 1
    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False, dtype=dtype)

    # to gpu
    device = get_cuda_test_device()
    torch.cuda.set_device(device)
    input_data_cuda = to_device(input_data, device)

    # parallel bitpacking creating packed int weights
    random_int4_tensor = torch.randint(0, 2 ** w_bit - 1, size=(num_input_features, num_hidden_fc), device=device)

    wf = torch.tensor(list(range(0, 32, w_bit)), dtype=torch.int32, device=device).unsqueeze(0)
    intweight = torch.sum(
        torch.bitwise_left_shift(
            random_int4_tensor.reshape(-1, 32 // w_bit, random_int4_tensor.size(-1)),
            wf.unsqueeze(-1)
        ),
        dim=1,
        dtype=torch.int32
    )

    # === following configuration from low_bit_llama https://github.com/GreenBitAI/low_bit_llama === #
    asym = False
    double_groupsize = 1
    v1 = False
    # =============================================================================================== #

    mbwq_linear_layer = MBWQLinearCuda(in_channels=num_input_features,
                                      out_channels=num_hidden_fc,
                                      w_bit=w_bit,
                                      dtype=dtype,
                                      group_size=group_size,
                                      dq_group_size=double_groupsize,
                                      dq_mode=1 if v1 else 2,
                                      use_gba_quant=True,
                                      asym=asym,
                                      use_mbw=False)
    mbwq_linear_layer.to(device)
    mbwq_linear_layer.set_qweight_data(intweight)

    # this function called before forward
    mbwq_linear_layer.prepare_params()

    # random scales and zeros
    scales = torch.randn_like(mbwq_linear_layer.scales).half()
    zeros = torch.randn_like(mbwq_linear_layer.zeros).half()
    mbwq_linear_layer.set_scales(scales)
    mbwq_linear_layer.set_zeros(zeros)

    start_time = time.time()
    for i in range(num_runs):
        result1 = mbwq_linear_layer(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("bitorch-engine mbwq_linear (CUDA) run time: %.6f s" % (time_engine/num_runs))

    # reconstruct the fp weights.
    fp_weights = MBWQLinearCuda.q42fp_weight(mbwq_linear_layer.qweight, mbwq_linear_layer.scales,
                                             mbwq_linear_layer.zeros, group_size, mbwq_linear_layer.w_bit, mbwq_linear_layer.q_perm)
    result1_pt = torch.matmul(input_data_cuda, fp_weights)

    assert torch.all(torch.isclose(result1, result1_pt, rtol=10, atol=10, equal_nan=False))

    mpq_linear_layer = MPQLinearCuda(in_channels=num_input_features,
                                      out_channels=num_hidden_fc,
                                      w_bit=w_bit,
                                      dtype=dtype,
                                      group_size=group_size,
                                      dq_group_size=double_groupsize,
                                      dq_mode=1 if v1 else 2,
                                      use_gba_quant=True,
                                      asym=asym)
    mpq_linear_layer.to(device)
    mpq_linear_layer.prepare_params()
    # small correction for scales to be 1 and zero to be 0
    mpq_linear_layer.scales = scales
    mpq_linear_layer.zeros = zeros

    # ====== TEST reconstruct half weight from qweight ====== #
    mpq_linear_layer.qweight.scales = mpq_linear_layer.scales
    mpq_linear_layer.qweight.zeros = mpq_linear_layer.zeros
    mpq_linear_layer.qweight.w_bit = w_bit
    mpq_linear_layer.qweight.asym = False
    mpq_linear_layer.qweight.g_idx = None
    mpq_linear_layer.qweight.group_size = group_size
    mpq_linear_layer.qweight.data = pack_fp_weight(fp_weights, mpq_linear_layer.qweight)

    # reconstruct the fp weight
    fp_w = unpack_qweight(mpq_linear_layer.qweight)

    # validate the reconstruction result of MBWQLinearCuda.q42fp_weight
    assert torch.all(torch.isclose(fp_weights, fp_w, rtol=1e-1, atol=1e-1, equal_nan=False))

    start_time = time.time()
    for i in range(num_runs):
        result2 = mpq_linear_layer(input_data_cuda)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("bitorch-engine mpq_linear forward (CUDA) run time: %.6f s" % (time_engine/num_runs))

    assert torch.all(torch.isclose(result1, result2, rtol=10, atol=10, equal_nan=False))

