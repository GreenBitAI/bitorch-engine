ARG FROM_IMAGE=pytorch/manylinux-builder:cuda12.1-2.3
FROM ${FROM_IMAGE} as builder-base
RUN mkdir "/build_scripts"
RUN mkdir "/workspace"

FROM builder-base as pytorch-base
#COPY "build_scripts/prepare_builder_image.sh" "/build_scripts/"
#RUN bash "/build_scripts/prepare_builder_image.sh" "${FROM_IMAGE}" && \
#  rm "/build_scripts/prepare_builder_image.sh"

FROM pytorch-base as cutlass-install
ARG CUTLASS_VERSION="2.8.0"
ARG CUTLASS_HOME="/opt/cutlass"
RUN git clone --depth 1 --branch "v${CUTLASS_VERSION}" "https://github.com/NVIDIA/cutlass.git" --recursive "${CUTLASS_HOME}/source" && \
  mkdir "${CUTLASS_HOME}/build" && \
  cd "${CUTLASS_HOME}/build" && \
  cmake ../source \
    -DCMAKE_INSTALL_PREFIX="${CUTLASS_HOME}/install" \
    -DCUTLASS_ENABLE_HEADERS_ONLY=ON \
    -DCUTLASS_ENABLE_TOOLS=ON \
    -DCUTLASS_ENABLE_LIBRARY=OFF \
    -DCUTLASS_ENABLE_PROFILER=OFF \
    -DCUTLASS_NVCC_ARCHS='75;80;86' && \
  cmake --install . && \
  rm -rf "${CUTLASS_HOME}/build" "${CUTLASS_HOME}/source"

FROM cutlass-install as build-ready
ARG PYTHON_HOME="/opt/python/cp310-cp310"
ENV PATH="${PYTHON_HOME}/bin:${PATH}"
ARG CUSTOM_TORCH_URL="https://packages.greenbit.ai/whl/cu121/torch/torch-2.3.0-cp310-cp310-linux_x86_64.whl"
ARG TORCHVISION_VERSION="0.18.0"
ARG TORCHVISION_INDEX_URL="https://download.pytorch.org/whl/cu121"
RUN pip install "${CUSTOM_TORCH_URL}" && \
  pip install "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCHVISION_INDEX_URL}" && \
  pip cache purge && \
  rm -rf /build_scripts

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
  BIE_FORCE_CUDA="true" CPATH="${CUTLASS_HOME}/install/include" pip install -e ${BUILD_TARGET} -v && \
  rm -rf build/ bitorch_engine.egg-info/
WORKDIR "/workspace"

FROM no-examples as example-ready
RUN pip install -r /bitorch-engine/examples/mnist-lightning/requirements.txt
