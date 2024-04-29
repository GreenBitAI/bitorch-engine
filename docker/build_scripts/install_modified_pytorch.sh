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
    gdrive_id="1PoVor85-RF3s0KpOP19mFV5hNUnHERa1"
    file="torch-2.2.2-cp310-cp310-linux_x86_64.whl"
    checksum="6646519e5e7b4af8f99b79eb9be3e6460b0d05c4695bbf86de02568f37ff3fea"
fi
if [ "${from_image}" == "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel" ]; then
    gdrive_id="1LjFNImboq8QeFSompMS2gPjBRYtP2Dsz"
    file="torch-2.2.2-cp310-cp310-linux_x86_64.whl"
    checksum="2a5953dab7be6c1640112e38ae7519ad88180d9fa79faab6c86dbee6b1cc210e"
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
