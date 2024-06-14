NetworkCommons
==============================================================

|MainBuild| |Issues| |PyPIDownloads| |Docs| |Codecov|

|Conda| |CondaDownloads|

.. |MainBuild| image:: https://github.com/saezlab/networkcommons/actions/workflows/main.yml/badge.svg
   :target: https://github.com/saezlab/networkcommons/actions
   
.. .. |Issues| image:: https://img.shields.io/github/issues/saezlab/networkcommons.svg
..    :target: https://github.com/saezlab/networkcommons/issues/

.. .. |PyPIDownloads| image:: https://static.pepy.tech/badge/decoupler
..    :target: https://pepy.tech/project/decoupler
   
.. .. |Docs| image:: https://readthedocs.org/projects/networkcommons/badge/?version=latest
..    :target: https://networkcommons.readthedocs.io/en/latest/?badge=latest

.. |Codecov| image:: https://codecov.io/github/saezlab/networkcommons/graph/badge.svg?token=RH438ALJC2
   :target: https://codecov.io/gh/saezlab/networkcommons

.. .. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/decoupler-py.svg
..    :target: https://anaconda.org/conda-forge/decoupler-py

.. .. |CondaDownloads| image:: https://img.shields.io/conda/dn/conda-forge/decoupler-py.svg
..    :target: https://anaconda.org/conda-forge/decoupler-py

``NetworkCommons`` is a package which allows users to download data and prior knowledge to perform static network inference using different methodologies. The package also provides different visualization and evaluation strategies.

.. .. figure:: graphical_abstract.png
..    :height: 500px
..    :alt: decoupler’s workflow
..    :align: center
..    :class: no-scaled-link

..    decoupler contains a collection of computational methods that coupled with 
..    prior knowledge resources estimate biological activities from omics data.

.. Check out the `Usage <https://decoupler-py.readthedocs.io/en/latest/notebooks/usage.html>`_ or any other tutorial for further information.

.. If you have any question or problem do not hesitate to open an `issue <https://github.com/saezlab/decoupler-py/issues>`_.

.. scverse
.. -------
.. ``decoupler`` is part of the `scverse <https://scverse.org>`_ ecosystem, a collection of tools for single-cell omics data analysis in python.
.. For more information check the link.

License
-------
The data redistributed by OmniPath does not have a license, each original resource carries their own. 
`Here <https://omnipathdb.org/info>`_ one can find the license information of all the resources in OmniPath.

.. Citation
.. -------
.. Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D., Müller-Dott S., Taus P., Dugourd A., Holland C.H., 
.. Ramirez Flores R.O. and Saez-Rodriguez J. 2022. decoupleR: ensemble of computational methods to infer biological activities 
.. from omics data. Bioinformatics Advances. https://doi.org/10.1093/bioadv/vbac016

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Main

   installation
   api
   release_notes
   reference

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Vignettes

   notebooks/a_simple_example
   notebooks/MOON
   notebooks/progeny
   notebooks/dorothea
   notebooks/msigdb
   notebooks/pseudobulk
   notebooks/spatial
   notebooks/bulk
   notebooks/benchmark
   notebooks/translate