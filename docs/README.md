# Bitorch Engine Documentation 

## Requirements

First you should install BITorch Engine as normal.
Then, install the additional requirements with:
```bash
# in the docs folder
pip install -r requirements.txt
```

## Magic Build Script

The script `make_docs.sh` will try to automagically build the documentation for you:
```bash
./docs/make_docs.sh
```
If there is a problem, see the script and manual steps below.

## Manual Build

The docs for `bitorch_engine` are generated using the [sphinx](https://www.sphinx-doc.org/en/master/>) package.
To build the docs, `cd` into the repository root and execute.

```bash
sphinx-build -b html docs/source/ docs/build/ -a
```

The generated `HTML` files will be put into `docs/build`.

## Synchronize from Readmes

To synchronize information from the Readme, we can use pandoc to convert the markdown file to RST:
```bash
pip install pandoc
pandoc --from=markdown --to=rst --output=README.rst README.md
```

Afterward, we need to fix a few issues, such as incorrect URLs, collapsible sections, etc.
and then move those sections to the appropriate place in the documentation.
You can try automatically doing so by running `python docs/scripts/convert_docs.py README.rst` before building with sphinx.
