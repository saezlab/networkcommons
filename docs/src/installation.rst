############
Installation
############

``NetworkCommons`` requires ``Python`` version >= 3.10 and < 3.13 to run.

-------------------
Requirements
-------------------

NetworkCommons requires `graphviz <https://graphviz.gitlab.io/download/>`_ to visualize networks.

Conda
-------------------

If you are using ``conda``, you can install the system-level graphviz dependency with:

.. code-block:: console

   conda install graphviz


Ubuntu
-------------------
``graphviz`` is also available via the APT package manager in Ubuntu:

.. code-block:: console

   sudo apt-get install -y graphviz graphviz-dev

MacOS
-------------------

In MacOS, it can be installed using `Homebrew <https://brew.sh/>`_.

.. code-block:: console

   brew install graphviz

In some of our local MacOS tests, we also needed to set the following environment variables:

.. code-block:: console

   export PATH="$(brew --prefix graphviz)/bin:$PATH"
   export CFLAGS="-I$(brew --prefix graphviz)/include"
   export LDFLAGS="-L$(brew --prefix graphviz)/lib"

Please note, if you are using a different architecture or operating system within a subsystem, you may need to make sure that the installed binaries of graphviz were compiled for the correct architecture and that they are visible to the Python interpreter.

-------------------
PIP
-------------------

Install the latest stable release from PyPI:

.. code-block:: console

   pip install networkcommons

Optional extras
~~~~~~~~~~~~~~~

NetworkCommons ships several optional extras for additional functionality:

.. code-block:: console

   # ILP solvers for CORNETO (Gurobi, SCIP, pygraphviz)
   pip install networkcommons[corneto-backends]

   # igraph support
   pip install networkcommons[igraph]

   # PyTorch-based models (e.g. LEMBAS)
   pip install networkcommons[torch]

   # All extras at once
   pip install 'networkcommons[corneto-backends,igraph,torch]'

GPU / CUDA support (PyTorch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default ``networkcommons[torch]`` installs the CPU-only PyTorch wheel from
PyPI. To enable GPU acceleration, install the matching CUDA wheel **after**
the main install by pointing pip at the PyTorch CUDA index.
Replace ``cu128`` with your local CUDA version (``cu118``, ``cu124``, ``cu126``, …):

.. code-block:: console

   pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
       "torch>=2.1+cu128"

Development version
~~~~~~~~~~~~~~~~~~~

To install the latest development version directly from GitHub:

.. code-block:: console

   pip install 'networkcommons[corneto-backends] @ git+https://github.com/saezlab/networkcommons@dev'

-------------------
Pixi
-------------------

`Pixi <https://prefix.dev/>`_ manages both conda and PyPI dependencies and is
the recommended environment manager for development.

.. code-block:: console

   # install pixi (if not already installed)
   curl -fsSL https://pixi.sh/install.sh | sh

   # clone the repo and install the dev environment
   git clone https://github.com/saezlab/networkcommons.git
   cd networkcommons
   pixi install -e dev

The ``dev`` environment includes all optional extras (torch, igraph,
corneto-backends, pygraphviz).

GPU / CUDA support with Pixi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pixi resolves PyPI packages via `uv <https://docs.astral.sh/uv/>`_ internally.
Because the correct CUDA variant of PyTorch depends on your local driver, it is
not pinned in the shared ``pyproject.toml``. Instead, create a personal
``requirements-local.txt`` (already in ``.gitignore``) with your CUDA version:

.. code-block:: text

   --extra-index-url https://download.pytorch.org/whl/cu128
   torch>=2.1+cu128

Then install it into the pixi environment using pixi's bundled uv:

.. code-block:: console

   pixi run -e dev uv pip install -r requirements-local.txt

This needs to be re-run after any ``pixi install -e dev`` that upgrades torch.
