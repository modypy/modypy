# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: disable=missing-module-docstring,redefined-builtin

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import modypy

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "MoDyPy"
copyright = "2021, Ralf Gerlich"
author = "Ralf Gerlich"

release = modypy.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", "modypy-sphinx-style/_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.rst"]

# Enable numeric reference for figures
numfig = True

# Intersphinx mappings
intersphinx_mapping = {
    "Python 3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static", "modypy-sphinx-style/_static"]

# Logo and Favicon configuration
html_logo = "modypy-sphinx-style/_static/logo.svg"
html_favicon = "modypy-sphinx-style/_static/logo.ico"

html_css_files = ["modypy.css"]

html_context = {}

# Add Permalinks for the sections
html_permalinks = True

if "deployment" in tags:
    # Specific for docs.modypy.org deployment
    html_context["deployment"] = True
    html_context["static_url"] = "https://docs.modypy.org/"
