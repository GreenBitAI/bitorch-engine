#!/usr/bin/env bash

function usage() {
    echo "./.dev-scripts/publish-docker-internal.sh OPTION"
    echo "builds a package and publishes it to (test-)pypi"
    echo
    echo "OPTION must be either 'pre-release' or 'release'."
}

if ! [ "$#" = "1" ] || [ "${1}" = "-h" ]; then
    usage
    exit
fi

if ! [ "${1}" = "release" ] && ! [ "${1}" = "pre-release" ]; then
    usage
    exit
fi

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

retry () {
    "$@" || (sleep 2 && "$@") || (sleep 4 && "$@") || (sleep 8 && "$@")
}

SRC_ROOT="${BITORCH_ENGINE_ROOT:-/bitorch-engine}"
check_error [ -f "${SRC_ROOT}/setup.py" ]
cd "${SRC_ROOT}"

set -o xtrace

# main script content

if [ -z "$(git status --porcelain)" ]; then
    echo "Git seems clean."
else
    if check_yes "Git not clean. Do you want to see the diff to proceed?"; then
        git diff
        if ! check_yes "Proceed with these differences?"; then
            echo "There are uncommitted changes, aborting."
            exit 1
        fi
    else
        echo "Git not clean, aborting."
        exit 1
    fi
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
fi

echo "Building version ${version_str}."

if [ "${1}" = "pre-release" ]; then
    echo "${version_str}" > "${version_file}"
fi

pip install build

check_error python3 -m build --no-isolation --wheel

# set patchelf binary:
export PATCHELF_BIN=/usr/local/bin/patchelf
patchelf_version=$($PATCHELF_BIN --version)
echo "patchelf version: ${patchelf_version} (${PATCHELF_BIN})"

# install openssl and zip
yum clean all
retry yum install -q -y zip openssl

LIB_DIR="extensions"
PREFIX="bitorch_engine"
LIB_SO_RPATH='$ORIGIN/../../torch/lib:$ORIGIN'
FORCE_RPATH="--force-rpath"

#######################################################################
# do not remove (license notice):
# code below was copied and modified from:
# https://github.com/pytorch/builder/blob/main/manywheel/build_common.sh
# it was released under BSD-2-Clause License
#######################################################################
# ADD DEPENDENCIES INTO THE WHEEL
#
# auditwheel repair doesn't work correctly and is buggy
# so manually do the work of copying dependency libs and patchelfing
# and fixing RECORDS entries correctly
######################################################################

fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    DIRNAME=$(dirname $1)
    BASENAME=$(basename $1)
    # Do not rename nvrtc-builtins.so as they are dynamically loaded
    # by libnvrtc.so
    # Similarly don't mangle libcudnn and libcublas library names
    if [[ $BASENAME == "libnvrtc-builtins.s"* || $BASENAME == "libcudnn"* || $BASENAME == "libcublas"*  ]]; then
        echo $1
    else
        INITNAME=$(echo $BASENAME | cut -f1 -d".")
        ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
        echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
    fi
}

fname_without_so_number() {
    LINKNAME=$(echo $1 | sed -e 's/\.so.*/.so/g')
    echo "$LINKNAME"
}

make_wheel_record() {
    FPATH=$1
    if echo $FPATH | grep RECORD >/dev/null 2>&1; then
        # if the RECORD file, then
        echo "$FPATH,,"
    else
        HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
        FSIZE=$(ls -nl $FPATH | awk '{print $5}')
        echo "$FPATH,sha256=$HASH,$FSIZE"
    fi
}

replace_needed_sofiles() {
    find $1 -name '*.so*' | while read sofile; do
        origname=$2
        patchedname=$3
        if [[ "$origname" != "$patchedname" ]] || [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
            set +e
            origname=$($PATCHELF_BIN --print-needed $sofile | grep "$origname.*")
            ERRCODE=$?
            set -e
            if [ "$ERRCODE" -eq "0" ]; then
                echo "patching $sofile entry $origname to $patchedname"
                $PATCHELF_BIN --replace-needed $origname $patchedname $sofile
            fi
        fi
    done
}

echo 'Built this wheel:'
ls /bitorch-engine/dist

mkdir /tmp_dir
pushd /tmp_dir

for pkg in /bitorch-engine/dist/bitorch_engine*linux*.whl; do

    # if the glob didn't match anything
    if [[ ! -e $pkg ]]; then
        continue
    fi

    rm -rf tmp
    mkdir -p tmp
    cd tmp
    cp $pkg .

    unzip -q $(basename $pkg)
    rm -f $(basename $pkg)

    # set RPATH of $LIB_DIR/ files to $ORIGIN
    find $PREFIX/$LIB_DIR -maxdepth 1 -type f -name "*.so*" | while read sofile; do
        echo "Setting rpath of $sofile to ${LIB_SO_RPATH:-'$ORIGIN'}"
        $PATCHELF_BIN --set-rpath ${LIB_SO_RPATH:-'$ORIGIN'} ${FORCE_RPATH:-} $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # regenerate the RECORD file with new hashes
    record_file=$(echo $(basename $pkg) | sed -e 's/-cp.*$/.dist-info\/RECORD/g')
    if [[ -e $record_file ]]; then
        echo "Generating new record file $record_file"
        : > "$record_file"
        # generate records for folders in wheel
        find * -type f | while read fname; do
            make_wheel_record "$fname" >>"$record_file"
        done
    fi

    # zip up the wheel back
    # todo: determine if typo should be fixed or not
    zip -rq $(basename $pkg) $PREIX*

    # replace original wheel
    rm -f $pkg
    mv $(basename $pkg) $pkg
    chown "${USER_ID}:${GROUP_ID}" $pkg
    cd ..
    rm -rf tmp
done
########################################################################
# do not remove (license notice):
# code above was copied and modified from https://github.com/pytorch/builder/blob/main/manywheel/build_common.sh
# released under BSD-2-Clause License
#######################################################################

echo "To publish to real PyPI use:"
echo "    python3 -m twine dist-${PUBLISH_BIE_VERSION}/*"
echo "To publish to test PyPI use:"
echo "    python3 -m twine upload --repository testpypi dist-${PUBLISH_BIE_VERSION}/*"

if [ "${1}" = "pre-release" ]; then
    git checkout "${version_file}"
fi
