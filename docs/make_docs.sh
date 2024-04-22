#!/usr/bin/env bash

if ! [ -d "docs" ]; then
    cd ..
fi
if ! [ -d "docs" ]; then
    echo "Could not locate docs directory. Please run the script from the root or the docs directory."
    exit 1
fi

pip freeze | grep pandoc &> /dev/null
if ! [ $? == "0" ]; then
    echo "Pandoc not found. Please install it, e.h. with 'pip install pandoc'."
    exit 1
fi

pandoc --from=markdown --to=rst --output=README.rst README.md
python docs/scripts/convert_docs.py README.rst
sphinx-build -b html docs/source/ docs/build/ -a
