import pytest
import torch
from tests.layers.util import get_cuda_test_device, get_cuda_test_device_id, to_device
from bitorch_engine.utils.quant_operators import nv_tensor_quant
from bitorch_engine.layers.qembedding.binary.layer import pad_embedding_dim

if torch.cuda.is_available():
    from bitorch_engine.functions.cuda import *

"""
    Test nbit inference layers
"""

# lower print threshold
torch.set_printoptions(threshold=100)

TEST_INPUT_DATA = [
    (50000, 8),
    (50000, 100),
    # size according to gpt3
    (50000, 768),
    (50000, 1024),
    (50000, 1280),
    (50000, 1600),
    (50000, 2048),
    (50000, 2304),
    (50000, 3072),
    (50000, 4096),
    (256000, 768),
    (256000, 1024),
    (256000, 1280),
    (256000, 1600),
    (256000, 2048),
    (256000, 2304),
    (256000, 3072),
    (256000, 4096),
]

# TEST_INPUT_DATA = [
#     (100, 100),
#     (128, 64),
#     (64, 128),
# ]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("num_input_features", [32, 64, 96, 128, 160, 192, 224, 256, 512])
@pytest.mark.parametrize("num_hidden_fc", [32, 64, 96, 128, 160, 192, 224, 256, 512])
@pytest.mark.parametrize("batch_size", [32, 64, 96, 128, 160, 192, 224, 256, 512])
def test_fp32toint4_cuda(num_input_features, num_hidden_fc, batch_size):
    # pytest.skip("Skip long-running test for now.")
    print("\nM:{}, N:{}, K:{}.".format(batch_size, num_hidden_fc, num_input_features))

    # creating test data
    input_data = torch.normal(0, 1, size=(batch_size, num_input_features), requires_grad=False)

    # to gpu
    device = get_cuda_test_device()
    input_data_cuda = to_device(input_data, device)
    num_runs = 100

    # warm up
    for i in range(num_runs):
        output_nv_int4 = nv_tensor_quant(input_data_cuda, num_bits=4)[0]

    stream = torch.cuda.current_stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream=stream)
    for i in range(num_runs):
        output_nv_int4 = nv_tensor_quant(input_data_cuda, num_bits=4)[0]
    end.record(stream=stream)
    torch.cuda.synchronize()
    nv_q4_time = start.elapsed_time(end)/1000/num_runs
    # print("nv_q4 function run time:\t%.6f s" % (nv_q4_time))
    # print(output_nv_int4)

    # nv quant function
    start.record(stream=stream)
    for i in range(num_runs):
        output_int4 = fp32toint4(input_data_cuda)
    end.record(stream=stream)
    torch.cuda.synchronize()
    rt = start.elapsed_time(end) / 1000 / num_runs
    # print("BE fp32tpint4 function run time:\t{:.6f} s, {:.2f} times speedup".format(rt, nv_q4_time / rt))
    # print(output_int4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
def test_tensor_to_packed_uint8(vocab_size, embedding_size):
    print("vocab size:{}, embedding dim:{}".format(vocab_size, embedding_size))

    # creating test data
    device = get_cuda_test_device()
    input_data = torch.normal(-1, 1, size=(vocab_size, embedding_size), requires_grad=False)
    print(input_data.size())
    input_data_cuda = to_device(input_data, device)
    input_data_cuda = pad_embedding_dim(input_data_cuda)
    output = tensor_to_packed_uint8(input_data_cuda)
    print(output.size())
    assert (output.dtype == torch.uint8), \
        'The output tensor has a incorrect dtype {}.'.format(output.dtype)

    assert (output.size(1) == (input_data_cuda.size(1) / 8)), \
        'The size of packed dimension should be "{}", but got "{}".' \
            .format(input_data_cuda.size(1) / 8, output.size(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
def test_unpack_uint8_tensor_custom():
    vocab_size = 64
    embedding_size = 32
    print("vocab size:{}, embedding dim:{}".format(vocab_size, embedding_size))
    batch_size = 2
    seq_length = 16

    # embedding_weights shape (batch_size, seq_length, packed_embedding_dim)
    # scale has shape (batch_size, seq_length, 1)
    # r has shape (batch_size, seq_length, packed_embedding_dim * 8)

    device = get_cuda_test_device()

    embedding_weights_last_dim = torch.tensor([[[0, 16, 35, 255]]], dtype=torch.uint8)
    # replicate (keeps last dim)
    embedding_weights = embedding_weights_last_dim.expand(batch_size, seq_length, embedding_size // 8)

    scale = torch.rand(size=(batch_size, seq_length, 1))#.expand(batch_size, seq_length, embedding_weights.size(2))
    print("Scale:", scale.size(), scale)
    emb_cuda = to_device(embedding_weights, device)
    scale_cuda = to_device(scale, device)

    expected_output_last_dim = torch.tensor(
        [[
            # 0
            [-1,] * 8 +
            # 16 (reverse direction of bits)
            [-1, -1, -1, +1, -1, -1, -1, -1][::-1] +
            # 35 (reverse direction of bits)
            [-1, -1, +1, -1, -1, -1, +1, +1][::-1] +
            # 255
            [1,] * 8
        ]],
        dtype=torch.float32)
    print(expected_output_last_dim)
    expected_output = expected_output_last_dim.expand(batch_size, seq_length, embedding_size).to(device) * scale_cuda

    emb_cuda = pad_embedding_dim(emb_cuda)
    print("Embedding Weights:", emb_cuda.size(), emb_cuda)
    output = unpack_uint8_tensor(emb_cuda, scale_cuda)
    print(output.size())
    print("Output:", output[0][0].size(), output[0][0])
    print("ExpOutput:", expected_output[0][0].size(), expected_output[0][0])
    assert (output.size(2) == (emb_cuda.size(2) * 8)), \
        'The size of packed dimension should be "{}", but got "{}".' \
            .format(emb_cuda.size(2) * 8, output.size(2))
    torch.testing.assert_close(output, expected_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
def test_unpack_uint8_tensor(vocab_size, embedding_size):
    print("vocab size:{}, embedding dim:{}".format(vocab_size, embedding_size))
    # embedding_weights shape (batch_size, seq_length, packed_embedding_dim)
    # scale has shape (batch_size, seq_length, 1)
    # r has shape (batch_size, seq_length, packed_embedding_dim * 8)
    batch_size = 8
    seq_length = 16
    packed_dim = (embedding_size // 8 + 1) * 8
    device = get_cuda_test_device()
    embedding_weights = torch.randint(0, 2 ** 8, (batch_size, seq_length, packed_dim), dtype=torch.uint8)
    scale = torch.rand(batch_size, seq_length, 1)
    print(scale)
    emb_cuda = to_device(embedding_weights, device)
    scale_cuda = to_device(scale, device)

    emb_cuda = pad_embedding_dim(emb_cuda)
    print(emb_cuda)
    output = unpack_uint8_tensor(emb_cuda, scale_cuda)
    print(output.size())
    print(output)
    assert (output.size(2) == (emb_cuda.size(2) * 8)), \
        'The size of packed dimension should be "{}", but got "{}".' \
            .format(emb_cuda.size(2) * 8, output.size(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
def test_q4_pack_unpack_cuda():
    num_runs = 10
    for i in range(num_runs):
        # creating test data
        # Generate a random int tensor with values in the range [-8, 7]
        input_data = torch.randint(low=-8, high=8, size=(10, 10), dtype=torch.int)

        # to gpu
        device = get_cuda_test_device()
        input_data_cuda = to_device(input_data, device)

        packed = q4_pack_tensor(input_data_cuda)

        assert packed.dtype == torch.int8
        assert packed.numel() * 2 == input_data_cuda.numel()

        unpacked = q4_unpack_tensor(packed)

        # Mask to keep only the lowest 4 bits
        mask = 0b1111

        bitwise_difference = unpacked & mask ^ input_data_cuda & mask
        assert (bitwise_difference == 0).all()

        scale = 0.045

        # testing if unpack and scale correctly outputs
        unpacked_scale = q4_unpack_and_scaling_tensor(packed, scale)
        unpacked_scale_int = (unpacked_scale/scale).to(torch.int)
        # print(unpacked_scale_int)

        tensor_input_int = input_data_cuda.to(torch.int)
        # print(tensor_input_int)
        assert torch.all(torch.isclose(unpacked_scale_int, tensor_input_int, rtol=1, atol=1, equal_nan=False))
