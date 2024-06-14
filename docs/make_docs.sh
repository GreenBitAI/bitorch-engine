#!/usr/bin/env bash

if ! [ -d "docs" ]; then
    cd ..
fi
if ! [ -d "docs" ]; then
    echo "Could not locate docs directory. Please run the script from the root or the docs directory."
    exit 1
fi

function package_required() {
    pip freeze | grep "${1}" &> /dev/null
    if ! [ $? == "0" ]; then
        echo "Package '${1}' not found. Please install it, e.g. with: $ pip install ${1}"
        exit 1
    fi
}
# check new packages are installed
package_required pandoc
package_required sphinx_design

pandoc --from=markdown --to=rst --output=README.rst README.md
python docs/scripts/convert_docs.py README.rst
sphinx-build -b html docs/source/ docs/build/ -a
