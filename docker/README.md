# Project Setup with Docker

## Build Docker Image

Use the following commands to build a [Docker](https://www.docker.com/) image with a Bitorch Engine installation.
This is currently only targeted and tested for CUDA 11.8 or 12.1 and _Torch 2.2.x_.

```bash
# cd docker
# you should be in this `docker` directory
docker build -t bitorch/engine .
# if you do not want to include installation of example requirements, use this instead:
docker build --target no-examples -t bitorch/engine .
```

After building, the docker image should contain:
  - The selected torch package (limited to those that we modified to support gradients for non-floating-point tensors)
  - A ready-built bitorch engine, and its requirements
  - Everything is installed in a conda environment with Python (currently 3.10)

## Build Options

Depending on your setup, you may want to adjust some options through build arguments:
- CUDA version, e.g. for CUDA 11.8 add
  - `--build-arg FROM_IMAGE="pytorch/manylinux-builder:cuda11.8-2.3"`
  - `--build-arg CUSTOM_TORCH_URL="https://packages.greenbit.ai/whl/cu118/torch/torch-2.3.0-cp310-cp310-linux_x86_64.whl"`
  - `--build-arg TORCHVISION_INDEX_URL="https://download.pytorch.org/whl/cu118"`
- repository URL, e.g. add `--build-arg GIT_URL="https://accesstoken:tokenpassword@github.com/MyFork/bitorch-engine.git"`
- Bitorch Engine branch or tag, e.g. add `--build-arg GIT_BRANCH="v1.2.3"`
- installing requirements for development, e.g. `--build-arg BUILD_TARGET=".[dev]"`
- if there is a problem, set the environment variable `BUILDKIT_PROGRESS=plain` to see all output

Here is an example:
```bash
BUILDKIT_PROGRESS=plain docker build -t bitorch/engine --build-arg BUILD_TARGET=".[dev]" --build-arg GIT_BRANCH="mybranch" .
```

## Run Docker Container

After building the image you can run a container based on it with:
```bash
docker run -it --rm --gpus all bitorch/engine:latest
```

## For Development

A docker image without the code cloned, e.g. for mounting a local copy of the code, can be made easily with the target `build-ready`:
```bash
# cd docker
# you should be in this `docker` directory
docker build -t bitorch/engine:build-ready --target build-ready .
docker run -it --rm --gpus all --volume "$(pwd)/..":/bitorch-engine bitorch/engine:build-ready
# in docker container:
cd /bitorch-engine
pip install -e ".[dev]" -v
```
However, this means the build results will not be persisted in the image, so you probably want to mount the same directory every time.
