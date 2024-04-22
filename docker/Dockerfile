ARG FROM_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel
FROM ${FROM_IMAGE} as pytorch-base

FROM pytorch-base as cutlass-install
ARG CUTLASS_VERSION="2.8.0"
RUN apt-get update && \
  apt-get install -y git && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  git clone --depth 1 --branch "v${CUTLASS_VERSION}" "https://github.com/NVIDIA/cutlass.git" --recursive /cutlass && \
  mkdir /cutlass/build && \
  cd /cutlass/build && \
  cmake .. -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DBUILD_TESTING=OFF -DCUTLASS_NVCC_ARCHS='75;80;86' && \
  make -j `$(nproc)` && \
  cmake --install .

FROM cutlass-install as build-ready
ARG FROM_IMAGE
COPY "build_scripts" "/build_scripts"
RUN bash "/build_scripts/install_modified_pytorch.sh" "${FROM_IMAGE}" && \
  rm -r "/build_scripts"

# clone instead of mounting makes the code in the image independent from local changes
# to mount your code before building, use the target above, and check the "For Development" section in docs/README.md
FROM build-ready as no-examples
ARG GIT_URL="https://github.com/GreenBitAI/bitorch-engine.git"
ARG GIT_BRANCH="main"
ARG BUILD_TARGET="."
RUN git clone \
    --depth 1 \
    --branch "${GIT_BRANCH}" \
    "${GIT_URL}" \
    /bitorch-engine && \
  cd /bitorch-engine && \
  BIE_FORCE_CUDA="true" pip install -e ${BUILD_TARGET} -v && \
  rm -rf build/ bitorch_engine.egg-info/

FROM no-examples as example-ready
RUN pip install -r /bitorch-engine/examples/mnist-lightning/requirements.txt
