#!/usr/bin/env bash

set -o xtrace

function usage() {
    echo "./.dev-scripts/publish-with-docker.sh BIE_VERSION [CUDA_VERSION]"
    echo "builds a package and publishes it to (test-)pypi"
    echo
    echo "BIE_VERSION must be a version string like 'v1.2.3'."
    echo "optional: CUDA_VERSION can be either '11.8' (default) or '12.1'."
}

trap exit INT

if ! ((1 <= $# && $# <= 2)) || [ "${1}" = "-h" ]; then
    usage
    exit
fi

export PUBLISH_BIE_VERSION="${1}"
CUDA_VERSION="${2:-11.8}"

if ! [[ "${PUBLISH_BIE_VERSION}" =~ ^v[0-9].[0-9].[0-9]$ ]]; then
    echo "Invalid BIE_VERSION '${PUBLISH_BIE_VERSION}' given."
    echo
    usage
    exit
fi

cuda_known="false"
add_build_arg=""
cuda_abbrev="unknown"
if [ "${CUDA_VERSION}" = "11.8" ]; then
    cuda_known="true"
    cuda_abbrev="cu118"
    torch_requirement="torch==2.2.2"
fi
if [ "${CUDA_VERSION}" = "12.1" ]; then
    cuda_known="true"
    cuda_abbrev="cu121"
    add_build_arg="--build-arg FROM_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"
    torch_requirement="torch==2.2.2"
fi
if [ "${cuda_known}" = "false" ]; then
    echo "Unknown CUDA_VERSION '${CUDA_VERSION}' given."
    echo
    usage
    exit
fi

echo "building bitorch engine ${PUBLISH_BIE_VERSION}"
echo "building for cuda ${CUDA_VERSION}"

bie_image_tag="bitorch/engine:publish-${cuda_abbrev}-${PUBLISH_BIE_VERSION}"
bie_container_name="bie-${cuda_abbrev}-${PUBLISH_BIE_VERSION}"
output_folder="./dist/${cuda_abbrev}"

# build/tag docker image
pushd docker
docker build --target no-examples ${add_build_arg} --build-arg GIT_BRANCH="${PUBLISH_BIE_VERSION}" -t "${bie_image_tag}" .
popd

docker container create -it \
    --rm \
    -it \
    -v "${output_folder}:/bitorch-engine/dist" \
    --name "${bie_container_name}" \
    -e PUBLISH_BIE_VERSION \
    -e BIE_FORCE_CUDA="true" \
    -e BIE_SKIP_BUILD="true" \
    -e BIE_TORCH_REQUIREMENT="${torch_requirement}" \
    -e BIE_WHEEL_PLATFORM="linux_x86_64" \
    -w /bitorch-engine \
    "${bie_image_tag}" \
    /workspace/publish-docker-internal.sh release

# make sure correct version is set
echo "${PUBLISH_BIE_VERSION#v}" > version.txt && docker container cp version.txt "${bie_container_name}":/bitorch-engine
docker container cp .dev-scripts/publish-docker-internal.sh "${bie_container_name}":/workspace

# for previous versions, we need to manually overwrite setup.py:
# TODO: can (hopefully) be removed later on
docker container cp setup.py "${bie_container_name}":/bitorch-engine

docker start -ai "${bie_container_name}"
