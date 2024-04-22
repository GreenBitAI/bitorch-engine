#!/usr/bin/env bash

from_image="${1}"
action="${2:-install}"

function usage() {
    echo "./install_modified_pytorch.sh DOCKER_IMAGE ACTION"
    echo "verify or install a modified torch version suitable for the chosen docker image"
    echo
    echo "ACTION can be 'install' or 'verify' and is optional (default: install)"
}

gdrive_id="unknown"
file="custom_torch.whl"

## list of known docker images and the corresponding google drive id to download modified torch packages
## adding them here individually is tedious, but we need to build them manually and ensure compatibility anyway

if [ "${from_image}" == "pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel" ]; then
    gdrive_id="1sS3LS_8wEm2CJ-oCHZAWYeuXjJHXPINP"
    file="torch-2.2.2-cp310-cp310-linux_x86_64.whl"
    checksum="1a7e8f1c315d3aefcc65b0a6676857b9cde4877737a134cf1423a048d8938985"
fi
if [ "${from_image}" == "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel" ]; then
    gdrive_id="18DP0P9MJ4U211HR5-1ss6NogFPcIOJDR"
    file="torch-2.2.2-cp310-cp310-linux_x86_64.whl"
    checksum="5f89163d910e1e1ee6010e4ea5d478756c021abab1e248be9716d3bee729b9e7"
fi
if [ "${from_image}" == "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel" ]; then
    gdrive_id="1QK_QqlPubFNgitiOkSABZ3AZyg7M0ezc"
    file="torch-2.1.0-cp39-cp39-linux_x86_64.whl"
    checksum="6600c130395b66bd047ca01b077f702703924eb3eaab2d3d04d9eb51154d9080"
fi
if [ "${from_image}" == "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel" ]; then
    gdrive_id="1fguT0jRJwRE1126rPpEvL9G6F246CLar"
    file="torch-2.1.0-cp39-cp39-linux_x86_64.whl"
    checksum="10b95aaca45558f3b80ee331677ddd925f3891ef542ab419ae68dd57641b9a12"
fi
#if [ "${from_image}" == "pytorch/pytorch:X.X.X-cudaXX.X-cudnn8-devel" ]; then
#    gdrive_id="xxx"
#    file="torch-X.X.X-cp310-cp310-linux_x86_64.whl"
#    checksum="xxx"
#fi

function check_error() {
    # shows and then runs a command. if the exit code is not zero, aborts the script
    # usage: check_error mv foo bar

    echo + $@
    "$@"
    local exit_code=$?
    if [ "${exit_code}" -ne 0 ]; then
        echo "! > An error occured, aborting."
        exit 1
    fi
}

if [ "${gdrive_id}" == "unknown" ]; then
    echo "Unknown image '${from_image}', could not choose modified torch accordingly."
    echo "Please add your base image to install_modified_pytorch.sh or request official support for your image via Github."
    echo
    usage
    exit 1
fi

check_error pip install gdown
check_error gdown "${gdrive_id}" -O "${file}"
check_error pip uninstall -y gdown

if [ -n "${checksum}" ]; then
    check_error sha256sum --check --status <<< "${checksum} ${file}"
fi

if [ "${action}" == "verify" ]; then
    exit 0
fi

check_error pip install "${file}"
check_error rm "${file}"
check_error pip cache purge
