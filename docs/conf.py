#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions Reduction Program (MERP)
# and National Energy Technology Laboratory's (NETL) National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government and the U.S.
# Government consequently retains certain rights. As such, the U.S. Government has been granted for
# itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare derivative works, and perform
# publicly and display publicly, and to permit others to do so.
#################################################################################

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# Standard libs
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "primo")))
# User-defined libs
from primo import RELEASE, VERSION

# -- Project information -----------------------------------------------------

project = "PRIMO"
copyright = "2023-2026, PRIMO"
author = (
    "The PRIMO team:\\\\ Dev Kakkad\\\\ Tyler Jaffe\\\\ Ruonan Li\\\\ Sangbum Lee\\\\"
    "Radhkakrishna Tumbalam Gooty\\\\ Miguel Zamarripa\\\\ Yash Puranik\\\\"
    "Markus Drouven"
)

latex_elements = {"maketitle": "\\author{" + author + "}\\sphinxmaketitle"}

# The full version, including alpha/beta/rc tags
release = RELEASE
# The short X.Y version
version = VERSION
# -- General configuration ---------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "nbsphinx",  # Jupyter notebooks as docs
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",  # Google and NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosectionlabel_prefix_document = True
autodoc_warningiserror = False  # suppress warnings during autodoc

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "apidoc/*tests*"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
html_logo = "_static/logo-print-hd.jpg"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#
html_favicon = "_static/PRIMO_favicon_300x300.svg"

# Options for LaTeX build
# change default engine (pdflatex) to correctly display Unicode characters in source
# https://docs.readthedocs.io/en/stable/guides/pdf-non-ascii-languages.html#sphinx-pdfs-with-unicode
# latex_engine = "xelatex"
# latex_elements = {
#    "preamble": r"""
# \usepackage[mono=false]{libertinus-otf}
# """
# }

## for MyST (Markdown)

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 2
myst_footnote_transition = True
myst_dmath_double_inline = True
panels_add_bootstrap_css = False
numfig = True
numfig_secnum_depth = 1
