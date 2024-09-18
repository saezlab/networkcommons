import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information

project = 'NetworkCommons'
author = 'SaezLab'
release = 'alpha'
version = '0.0.1'
copyright = f"{datetime.now():%Y}, NetworkCommons developers"

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'numpydoc',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
master_doc = 'contents'

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "alphabetical"

autodoc_typehints = "signature"
autodoc_docstring_signature = True
autodoc_follow_wrapped = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False

html_theme = 'sphinx_rtd_theme'
html_static_path = ["_static"]
html_theme_options = dict(
    logo_only=True,
    display_version=True,
)
html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user='saezlab',  # Username
    github_repo='networkcommons',  # Repo name
    github_version='master',  # Version
    conf_py_path='/docs/src/',  # Path in the checkout to the docs root
)
html_show_sphinx = False
html_logo = '_static/nc_logo.svg'
html_favicon = '_static/nc_logo.svg'
html_css_files = [
    'css/custom.css',
]

# -- Options for EPUB output
epub_show_urls = 'footnote'