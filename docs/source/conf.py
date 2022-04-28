# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
dn = os.path.dirname
sys.path.insert(0, dn(dn(dn(os.path.abspath(__file__)))))


# -- Project information -----------------------------------------------------

project = 'qudit-sim'
copyright = '2022, Yutaro Iiyama'
author = 'Yutaro Iiyama'

# The full version, including alpha/beta/rc tags
# TODO make this read setup.cfg
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for autodoc ------------------------------------------------------

import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    has_jax = False
else:
    has_jax = True

def _typehints_formatter(annotation, config):
    if annotation is np:
        return ':py:mod:`numpy`'
    elif has_jax and annotation is jnp:
        return ':py:mod:`jax.numpy`'
    elif annotation is np.ndarray:
        return ':py:class:`numpy.ndarray`'
    elif has_jax and annotation is jnp.ndarray:
        return ':py:class:`jax.numpy.ndarray`'

typehints_formatter = _typehints_formatter

autodoc_preserve_defaults = True
