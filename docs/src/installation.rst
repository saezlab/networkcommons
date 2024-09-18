############
Installation
############

``NetworkCommons`` requires ``Python`` version >= 3.10 to run.

-------------------
Requirements
-------------------

NetworkCommons requires `graphviz <https://graphviz.gitlab.io/download/>`_ to visualize networks.

In Ubuntu, it can be installed via using the APT package manager:

.. code-block:: console

   sudo apt-get update
   sudo apt-get install -y graphviz graphviz-dev

In MacOS, it can be installed using `Homebrew <https://brew.sh/>`_. In our local tests, we also needed to set the following environment variables:

.. code-block:: console

   brew install graphviz

.. code-block:: console

   echo 'export PATH="$(brew --prefix graphviz)/bin:$PATH"' >> ~/.zshrc
   echo 'export CFLAGS="-I$(brew --prefix graphviz)/include"' >> ~/.zshrc
   echo 'export LDFLAGS="-L$(brew --prefix graphviz)/lib"' >> ~/.zshrc


-------------------
PIP
-------------------

Currently, NetworkCommons can be installed via pip.

.. code-block:: console

   pip install networkcommons


Additionally, users can install backends for CORNETO via:

.. code-block:: console

   pip install networkcommons[corneto-backends]


To install the development version (with the CORNETO solvers), run:

.. code-block:: console

   pip install 'networkcommons[corneto-backends] @ git+https://github.com/saezlab/networkcommons@dev'


