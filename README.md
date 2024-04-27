# BITorch Engine (BIE)

Bitorch Engine is a cutting-edge computation library for neural networks that enhances PyTorch by integrating specialized
layers and functions tailored for **Low-Bit** quantized neural network operations.
It harnesses the robust capabilities of high-performance computing platforms, including GPUs and CPUs,
and is designed with future adaptability in mind to extend support to emerging NPU hardware technologies.

## More about BIE

Bitorch Engine offers a suite of optimized neural network components that are designed to leverage the full power of modern GPUs.
This includes custom CUDA kernels, quantization-aware training mechanisms, and a variety of layer types
that are specifically crafted to reduce computational overhead while maintaining high precision and accuracy in deep learning models.

Building on the foundational strengths of Bitorch Engine, the technology has been employed in pioneering projects that
push the boundaries of neural network training and inference.
For instance:

- [green-bit-llm-trainer](https://github.com/GreenBitAI/green-bit-llm/tree/main/green_bit_llm/sft): In this project, BIE represents a significant leap in the field of Large Language Model (LLM) fine-tuning. Unlike traditional approaches that either quantize a fully trained model or introduce a few additional trainable parameters for [LoRA](https://github.com/microsoft/LoRA) style fine-tuning, this project innovates by directly fine-tuning the quantized parameters of LLMs. This paradigm shift allows for the full-scale quantization fine-tuning of LLMs, ensuring that the training process tightly integrates with the quantization schema from the outset.
- [green-bit-llm-inference](https://github.com/GreenBitAI/green-bit-llm/tree/main/green_bit_llm/inference) also showcase the BIE's adeptness at supporting inference for models quantized from 4 to 2-bits without any significant loss in accuracy compared to the original 32 or 16-bits models. It stands as a testament to BIE's capability to maintain the delicate balance between model size, computational efficiency, and accuracy, addressing one of the key challenges in deploying sophisticated neural networks in resource-constrained environments.

These projects exemplify the practical applications of Bitorch Engine and underscore its flexibility and efficiency for modern AI research and development.
However, keep in mind that BIE is still in an early beta stage, see our roadmap below. 

## Roadmap

Our goals for BITorch engine in the future are (not necessarily in this order):

- Add support for (Distributed) Data Parallel training strategies (for selected layers)
- Provide better support for Metal kernels
- Improve our existing code, so it becomes even faster, more memory-efficient and easier to use
- Binary pip releases which include the built extensions

We are planning to release new features and improvements as they become available,
but this also means breaking changes can occur in the API during our beta stage.

## Installation

The requirements are:

- A compiler that fully supports C++17, such as clang or gcc (gcc 9.4.0 or newer is required, but gcc 12.x is not supported yet)
- Python 3.9 or later
- PyTorch 1.8 or later
- CUDA Toolkit 11.8 or 12.1 (optional, for CUDA accelerated layers)

For more detailed information, you can check the [requirements of PyTorch](https://github.com/pytorch/pytorch?tab=readme-ov-file#prerequisites).

Currently, the engine **needs to be built from source**.
We provide instructions how to install Python/PyTorch (and CUDA/MLX) for:

- Conda + Linux (with CUDA)
- Docker (with CUDA)
- Conda + MacOS (with MLX)

We recommend managing your BITorch Engine installation in a conda environment (otherwise you should adapt/remove certain variables, e.g. `CUDA_HOME`).
You may want to keep everything (environment, code, etc.) in one directory or use the default directory for conda environments.
You may wish to adapt the CUDA version to 12.1 where applicable.

### Conda on Linux (with CUDA)

1. Create Environment for Python 3.9 and activate it:
```bash
conda create -y --name bitorch-engine python=3.9
conda activate bitorch-engine
```
2. Install CUDA
```bash
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
3. [Download customized Torch 2.1.0](https://drive.google.com/drive/folders/1T22b8JhN-E3xbn3h332rI1VjqXONZeB7?usp=sharing) (it allows gradients on INT tensors, built for Python 3.9 and CUDA 11.8) and install it with pip:
```bash
pip install torch-2.1.0-cp39-cp39-linux_x86_64.whl
# optional: install corresponding torchvision (check https://github.com/pytorch/vision?tab=readme-ov-file#installation in the future)
pip install "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu118
```

Alternatively, you can also save the environment and clone the repository within the same directory.

<details><summary>Click to here to expand the instructions for this.</summary>

0. Set workspace dir (use an absolute path!):
```bash
export BITORCH_WORKSPACE="${HOME}/bitorch-workspace"
mkdir -p "${BITORCH_WORKSPACE}" && cd "${BITORCH_WORKSPACE}"
```
1. Create Environment for Python 3.9 and activate it:
```bash
conda create -y --prefix ./conda-env python=3.9
conda activate ./conda-env
```
2. Install CUDA
```bash
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
3. [Download customized Torch 2.1.0](https://drive.google.com/drive/folders/1T22b8JhN-E3xbn3h332rI1VjqXONZeB7?usp=sharing),
select the package fit for the cuda version you installed in the previous step
(it allows gradients on INT tensors, built for Python 3.9 and CUDA 11.8) and install it with pip:
```bash
pip install torch-2.1.0-cp39-cp39-linux_x86_64.whl
# optional: install corresponding torchvision (check https://github.com/pytorch/vision?tab=readme-ov-file#installation in the future)
pip install "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu118
```
</details>

After setting up the environment, clone the code and build with pip (to hide the build output remove `-v`):

```bash
git clone --recursive https://github.com/GreenBitAI/bitorch-engine
cd bitorch-engine
# only gcc versions 9.x, 10.x, 11.x are supported
# to select the correct gcc, use:
# export CC=gcc-11 CPP=g++-11 CXX=g++-11
CUDA_HOME="${CONDA_PREFIX}" pip install -e . -v
```

### Docker (with CUDA) 

You can also use our prepared Dockerfile to build a docker image (which includes building the engine under `/bitorch-engine`):

```bash
cd docker
docker build -t bitorch/engine .
docker run -it --rm --gpus all --volume "/path/to/your/project":"/workspace" bitorch/engine:latest
```

Check the [docker readme](docker/README.md) for options and more details.

### Conda on MacOS (with MLX)

1. We recommend to create a virtual environment for and activate it. In the following example we use a conda environment for python 3.9, 
but virtualenv should work as well.
```bash
conda create -y --name bitorch-engine python=3.9
conda activate bitorch-engine
```
2. Download [customized Torch for MacOS/arm](https://drive.google.com/drive/folders/1T22b8JhN-E3xbn3h332rI1VjqXONZeB7?usp=sharing) (it allows gradients on INT tensors, built for Python 3.9 and CUDA 11.8) and install it with pip:
```bash
pip install path/to/torch-2.2.1-cp39-none-macosx_11_0_arm64.whl
# optional: install corresponding torchvision (check https://github.com/pytorch/vision?tab=readme-ov-file#installation in the future)
pip install "torchvision==0.17.1"
```
3. For MacOS users and to use OpenMP acceleration, install OpenMP with Homebrew and configure the environment:
```bash
brew install libomp
# during libomp installation it should remind you, you need something like this:
export LDFLAGS="-L$(brew --prefix)/opt/libomp/lib"
export CPPFLAGS="-I$(brew --prefix)/opt/libomp/include"
```
4. To use the [mlx](https://github.com/ml-explore/mlx) accelerated `MPQLinearLayer`, you need to install the python library.
```bash
# use one of the following, to either install with pip or conda:
pip install mlx==0.4.0
conda install conda-forge::mlx=0.4.0
```
 Currently, we only tested version 0.4.0. However, newer versions might also work.
 To train the `MPQLinearLayer` you need to install our custom PyTorch version (see steps above).
 Without it, you need to specify `requires_grad=False` when initializing `MPQLinearLayer`.
5. You should now be able to build with:
```bash
git clone --recursive https://github.com/GreenBitAI/bitorch-engine
cd bitorch-engine
pip install -e . -v
```

## Build options

### Building Specific Extensions

While developing, a specific cpp/cuda extension can be (re-)build, by using the environment variable `BIE_BUILD_ONLY`,
like so:
```bash
BIE_BUILD_ONLY="bitorch_engine/layers/qlinear/binary/cpp" pip install -e . -v
```
It needs to a relative path to one extension directory.

### Building for a Specific CUDA Architecture

To build for a different CUDA Arch, use the environment variable `BIE_CUDA_ARCH` (e.g. use 'sm_75', 'sm_80', 'sm_86'):
```bash
BIE_CUDA_ARCH="sm_86" pip install -e . -v
```

### Force Building CUDA Modules

If you have CUDA development libraries installed, but `torch.cuda.is_available()` is False, e.g. in HPC or docker environments,
you can still build the extensions that depend on CUDA, by setting `BIE_FORCE_CUDA="true"`:
```bash
BIE_FORCE_CUDA="true" pip install -e . -v
```

## Development

To adjust the build options or address build failures, modify the configurations in 
[cpp_extension.py](bitorch_engine/utils/cpp_extension.py)/
[cuda_extension.py](bitorch_engine/utils/cuda_extension.py).

You may want to clean the build output before rebuilding, which may help to avoid errors and/or install development requirements:
```bash
python setup.py clean
# now build like usually, use ".[dev]" for development requirements, e.g.
CUDA_HOME="${CONDA_PREFIX}" pip install -e ".[dev]" -v
```

You can run our tests with pytest:
```bash
pytest
```

### Cuda Device Selection

To select a certain CUDA device, set the environment variable `BIE_DEVICE`, e.g.:
```bash
export BIE_DEVICE=1  # This selects the second CUDA device, as indexing starts from 0.
```

## Documentation

Check out the [Documentation](https://greenbitai.github.io/bitorch-engine) for API reference.

## Examples

- Basic example scripts can be found directly in [examples](examples).
- [green-bit-llm-trainer](https://github.com/GreenBitAI/green-bit-llm/tree/main/green_bit_llm/sft) showcases the fine-tuning training of LLMs with quantized parameters.
- [green-bit-llm-inference](https://github.com/GreenBitAI/green-bit-llm/tree/main/green_bit_llm/inference) showcases the BIE's adeptness at supporting fast inference for 4 to 2-bits LLMs.

## Contributors

BIE is under active development and currently maintained by contributors: [Haojin Yang](https://github.com/yanghaojin), [Joseph Bethge](https://github.com/Jopyth), [Nianhui Guo](https://github.com/NicoNico6), [Maximilian Schulze](https://github.com/max-3l), Hong Guo, [Paul Mattes](https://github.com/Snagnar).

Check our [contributing guide](CONTRIBUTING.md) to learn about how to contribute to the project.

## License

Bitorch Engine is made available under the [Apache 2.0 License](LICENSE). See the LICENSE file for details.

## Citation
If you use our approach in your research, please cite our work as follows:
```
@article{bitorch_engine,
  title={Bitorch Engine: Streamlining AI with Open-Source Low-Bit Quantization},
  author={Yang, Haojin and Bethge, Joseph and Guo, Nianhui and Schulze, Maximilian and Guo, Hong},
  journal={https://github.com/GreenBitAI/bitorch-engine},
  year={2024}
}
```

## References and Acknowledgements

This project builds upon or uses concepts from the following open-source projects:

- **[PyTorch](https://github.com/pytorch/pytorch)**
- **[CUTLASS](https://github.com/NVIDIA/cutlass)**
- **[MLX](https://github.com/ml-explore/mlx)**
- **[ExLlamaV2](https://github.com/turboderp/exllamav2)**
- **[TCBNN](https://github.com/pnnl/TCBNN)**
- **[GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)**

We extend our heartfelt gratitude to the developers of these projects for their invaluable contributions to the open-source community. Without their exceptional work, none of this would be possible.
The corresponding licenses of the reference projects can be found in the [licenses](licenses) directory of the source tree.

### Open Source Software Acknowledgment

This project makes use of open source software (OSS) components. The original code of these components is kept under their respective licenses and copyrights. We are grateful to the open-source community for making these resources available. For specific information about each component's license, please refer to the corresponding sections within our project documentation or the direct references provided in the "References" section of this document.

We endeavor to comply with all open source licenses and their requirements, including proper acknowledgment and notice. If there are any concerns or questions regarding our license acknowledgments, please reach out to us for clarification.
