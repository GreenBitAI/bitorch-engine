#!/usr/bin/env bash

function usage() {
    echo "./.dev-scripts/publish.sh VERSION"
    echo "builds a package and publishes it to (test-)pypi"
    echo
    echo "VERSION must be either 'pre-release' or 'release'."
}

if ! [ "$#" = "1" ] || [ "${1}" = "-h" ]; then
    usage
    exit
fi

if ! [ "${1}" = "release" ] && ! [ "${1}" = "pre-release" ]; then
    usage
    exit
fi

# set SCRIPT_ROOT:
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set SRC_ROOT and make sure that is our working directory
SRC_ROOT="$(readlink -f "${SCRIPT_ROOT}/..")"
cd "${SRC_ROOT}"

trap exit INT

function check_yes() {
    # asks the given yes or no question, returns true if they answer YES
    # usage:
    # if check_yes "Do you really want to delete foo?"; then
    #     rm foo
    # fi

    local prompt="${1}"
    read -p "${prompt} [y/N] " REPLY
    echo ""
    if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
        return 1
    fi
    return 0
}

function check_no() {
    # asks the given yes or no question, returns false if they answer NO
    # usage:
    # if check_no "Do you want to exit the script?"; then
    #     exit 0
    # fi

    local prompt="${1}"
    read -p "${prompt} [Y/n] " REPLY
    echo ""
    if [[ "${REPLY}" =~ ^[Nn]$ ]]; then
        return 1
    fi
    return 0
}

function check_error() {
    # shows and then runs a command. if the exit code is not zero, asks the user whether to continue
    # usage: check_error mv foo bar

    echo + $@
    "$@"
    local exit_code=$?
    if [ "${exit_code}" -ne 0 ]; then
        if ! check_yes "! > An error occurred, continue with the script?"; then
            if [ "${1}" = "pre-release" ]; then
                git checkout "${version_file}"
            fi
            exit 1
        fi
    fi
}

# main script content

if [ -z "$(git status --porcelain)" ]; then
    echo "Git seems clean."
else
    echo "There are uncommitted changes, aborting."
    exit 1
fi

if [ "${1}" = "release" ]; then
    version_file="${SRC_ROOT}/version.txt"
    version_content="$(cat "${version_file}")"
    major_minor_patch="$(cut -d '.' -f 1,2,3 <<< "${version_content}")"
    version_str="${major_minor_patch}"
else
    version_file="${SRC_ROOT}/version.txt"
    version_content="$(cat "${version_file}")"
    major_minor_patch="$(cut -d '.' -f 1,2,3 <<< "${version_content}")"
    date_str="$(date +"%Y%m%d")"
    git_ref="$(git rev-parse --short HEAD)"
    version_str="${major_minor_patch}.dev${date_str}+${git_ref}"
fi

if [ "${1}" = "release" ] && ! [ "${version_content}" = "${version_str}" ]; then
    echo "The file version.txt does not seem to contain a release version."
    exit 1
else
    if ! check_no "Publish version ${version_str} ?"; then
        exit 0
    fi
fi

if [ "${1}" = "pre-release" ]; then
    echo "${version_str}" > "${version_file}"
fi

check_error pip uninstall -y -r <(pip freeze)
check_error pip install --upgrade pip
check_error pip install -e ".[dev]" -v

pip install build twine

check_error pytest .
check_error python3 -m build --sdist

if check_yes "Publish to real PyPi?"; then
    echo "Do:"
    echo '    python3 -m twine dist/*'
else
    echo "Do:"
    echo '    python3 -m twine upload --repository testpypi dist/*'
fi

if [ "${1}" = "pre-release" ]; then
    git checkout "${version_file}"
fi
