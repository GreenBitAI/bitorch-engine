# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Bitorch Engine'
copyright = '2024, Haojin Yang, Joseph Bethge, Nianhui Guo, Maximilian Schulze, Hong Guo, Paul Mattes'
author = 'Haojin Yang, Joseph Bethge, Nianhui Guo, Maximilian Schulze, Hong Guo, Paul Mattes'

root_path = Path(__file__).resolve().parent.parent.parent
release = "unknown"
version_file = root_path / "version.txt"
if version_file.exists():
    with open(root_path / "version.txt") as handle:
        version_content = handle.read().strip()
        if version_content:
            release = version_content


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "autodocsumm",
    "sphinx.ext.intersphinx",
    "sphinx_toolbox.code",
    "sphinx_toolbox.collapse",
    "sphinx_design",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['.so', '_build', 'Thumbs.db', '.DS_Store']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
