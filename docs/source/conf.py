# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# import pydata_sphinx_theme

import os
import sys


# for the auto documentation to work
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../../pyramids"))

# General information about the project.
project = "pyramids"
copyright = "2024, Mostafa Farrag"
author = "Mostafa Farrag"
release = "0.7.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Enables autodoc
    "sphinx.ext.viewcode",  # Adds links to the source code
    "sphinx.ext.graphviz",  # Allows rendering of graphviz diagrams
    "sphinx.ext.napoleon",  # Allows for Google-style and Numpy docstrings
]

templates_path = ["_templates"]
exclude_patterns = []

root_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# Set the theme name
# Optionally, you can customize the theme's configuration
html_theme_options = {
    "logo_link": "index",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Serapieum-of-alex/pyramids",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": ["search-field.html", "navbar-icon-links"],
    "search_bar_text": "Search this site...",
    "navbar_align": "content",
    "navigation_depth": 4,
    "show_prev_next": False,
    "show_toc_level": 2,
    # Toc options
    "collapse_navigation": True,
    # "external_links": [
    #     {"name": "External Link", "url": "https://example.com"},
    # ],
    "header_links_before_dropdown": 4,
}

html_static_path = ["_static"]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}

# -- Options for autodoc -----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Ensure that the path to the Graphviz `dot` command is correct
graphviz_dot = "dot"
