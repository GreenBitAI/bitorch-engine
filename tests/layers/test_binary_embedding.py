import pytest
import torch
import time
from bitorch_engine.layers.qembedding.binary import BinaryEmbeddingBag
from tests.layers.util import get_cuda_test_device, to_device

from torch.nn import Embedding, EmbeddingBag, MSELoss

if torch.cuda.is_available():
    from bitorch_engine.layers.qembedding.binary import BinaryEmbeddingCuda


NUM_RUNS_CPU = 1
NUM_RUNS_GPU = 1

TEST_INPUT_DATA = [
    (10, 7),
    (50000, 7),
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

# smaller data for faster tests
TEST_INPUT_DATA_SMALL = [
    (10, 7),
    (5000, 7),
    (5000, 100),
    (50000, 768),
    (50000, 1024),
    (50000, 4096),
]


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA_SMALL)
def test_orig_embedding(vocab_size, embedding_size):
    emb = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)

    batch_size = 10
    input = torch.randint(0, vocab_size-1, (batch_size,))

    out = emb(input)

    print(out.size())

    assert(out.size(0) == batch_size)
    assert(out.size(1) == embedding_size)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA_SMALL)
def test_orig_embedding_2_dim(vocab_size, embedding_size):
    emb = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)

    batch_size = 10
    additional_dimension = 42
    input = torch.randint(0, vocab_size-1, (batch_size, additional_dimension))

    out = emb(input)

    print(out.size())

    assert(out.size(0) == batch_size)
    assert(out.size(1) == additional_dimension)
    assert(out.size(2) == embedding_size)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA_SMALL)
def test_orig_embedding_bag(vocab_size, embedding_size):
    emb_bag = EmbeddingBag(num_embeddings=vocab_size, embedding_dim=embedding_size)

    batch_size = 10
    num_embedded = 3
    input = torch.randint(0, vocab_size-1, (batch_size, num_embedded, ))

    out = emb_bag(input)

    print(out.size())

    assert(out.size(0) == batch_size)
    assert(out.size(1) == embedding_size)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA_SMALL)
def test_cpu_bag(vocab_size, embedding_size):
    print("vocab size:{}, embedding dim:{}".format(vocab_size, embedding_size))

    # testing embedding BAG
    qembeddingbag = BinaryEmbeddingBag(num_embeddings=vocab_size, embedding_dim=embedding_size)

    batch_size = 10
    num_embedded = 3
    input = torch.randint(0, vocab_size-1, (batch_size, num_embedded, ))

    start_time = time.time()
    for i in range(NUM_RUNS_CPU):
        out = qembeddingbag(input)
        assert(out.size(0) == batch_size)
        assert(out.size(1) == embedding_size)
        loss = out.sum()
    time_engine = time.time() - start_time
    print("binary embedding bag forward run time (CPU): %.6f s" % (time_engine/NUM_RUNS_CPU))

    # backward test
    start_time = time.time()
    for i in range(NUM_RUNS_CPU):
        result = loss.backward(retain_graph=True)
    time_engine = time.time() - start_time
    print("binary embedding bag backward run time (CPU): %.6f s" % (time_engine/NUM_RUNS_CPU))

    # get grad
    # print("binary embedding bag weight_grad (CPU):")
    # print(qembeddingbag.weight.grad)
    assert qembeddingbag.weight.grad != None
    assert torch.equal(qembeddingbag.weight.active_indices, input)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping CUDA-related test")
@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
def test_gpu(vocab_size, embedding_size):
    print("vocab size:{}, embedding dim:{}".format(vocab_size, embedding_size))

    batch_size = 10
    num_embedded = 5
    input = torch.randint(0, vocab_size-1, (batch_size, num_embedded, ))

    # to gpu
    device = get_cuda_test_device()
    input_cuda = to_device(input, device)

    qembeddingbag = BinaryEmbeddingBag(num_embeddings=vocab_size, embedding_dim=embedding_size)
    qembeddingbag.to(device)

    start_time = time.time()
    for i in range(NUM_RUNS_GPU):
        bag = qembeddingbag(input_cuda)
        assert(bag.size(0) == batch_size)
        assert(bag.size(1) == embedding_size)
        loss = bag.sum()
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("binary embedding bag forward run time (GPU): %.6f s" % (time_engine/NUM_RUNS_GPU))

    start_time = time.time()
    for i in range(NUM_RUNS_GPU):
        result = loss.backward(retain_graph=True)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("binary embedding bag backward run time (GPU): %.6f s" % (time_engine/NUM_RUNS_GPU))

    # get grad
    # print("binary embedding bag weight_grad (GPU):")
    # print(qembeddingbag.weight.grad)
    assert qembeddingbag.weight.grad != None
    assert torch.equal(qembeddingbag.weight.active_indices, input_cuda)

    qembedding = BinaryEmbeddingCuda(num_embeddings=vocab_size, embedding_dim=embedding_size)
    qembedding.to(device)
    qembedding.prepare_params()

    start_time = time.time()
    for i in range(NUM_RUNS_GPU):
        emb = qembedding(input_cuda)
        assert(emb.size(0) == batch_size)
        assert(emb.size(1) == num_embedded)
        assert(emb.size(2) == embedding_size)
        loss = emb.sum()
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("binary embedding forward run time (GPU): %.6f s" % (time_engine / NUM_RUNS_GPU))

    # backward test
    start_time = time.time()
    for i in range(NUM_RUNS_GPU):
        result = loss.backward(retain_graph=True)
    torch.cuda.synchronize()
    time_engine = time.time() - start_time
    print("binary embedding backward run time (GPU): %.6f s" % (time_engine/NUM_RUNS_GPU))

    # get grad
    # print("binary embedding weight_grad (GPU):")
    # print(qembedding.weight.grad)
    assert qembedding.qweight.grad != None
    assert torch.equal(qembedding.qweight.active_indices, input_cuda)



