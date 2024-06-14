
# bash code to run after installation to test for correct package installation

if [ -n "${SKIP_TESTS}" ]; then
    echo "Skipping tests. Done."
    exit 0
fi

# try basic importing, should detect errors of .so loading
python -c "from bitorch_engine.layers.qlinear.binary.cpp import BinaryLinearCPP"
python -c "from bitorch_engine.layers.qembedding.binary import BinaryEmbeddingCuda"
python -c "from bitorch_engine.layers.qlinear.nbit.cutlass import Q4LinearCutlass, Q8LinearCutlass, Q4MatMul"
python -c "from bitorch_engine.layers.qlinear.nbit.cuda import MPQLinearCuda, MBWQLinearCuda"
python -c "from bitorch_engine.layers.qlinear.nbit.cuda.utils import pack_fp_weight, unpack_qweight"
echo "Imports successful!"

set +o errexit
echo "Testing..."
(
    rm -rf bitorch_install_tmp_test_dir
    mkdir bitorch_install_tmp_test_dir
    cd bitorch_install_tmp_test_dir
    git clone https://github.com/GreenBitAI/bitorch-engine.git --depth 1 --branch "v0.2.5" bitorch_engine_git
    mv bitorch_engine_git/tests .
    pip install pytest numpy
    pytest tests/layers/test_nbit_linear.py
    pytest tests/layers/test_nbit_linear_mixbits.py
    pytest tests/functions/test_quant_ops.py
    pytest tests/layers/test_binary_linear.py
)
rm -rf bitorch_install_tmp_test_dir
