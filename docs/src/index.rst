.. image:: nc_banner.png

##############
Bridging the gap between data, methods and knowledge in network inference
##############

|MainBuild| |Codecov| |Docs|

.. |MainBuild| image:: https://github.com/saezlab/networkcommons/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/saezlab/networkcommons/actions
   
.. .. |Issues| image:: https://img.shields.io/github/issues/saezlab/networkcommons.svg
..    :target: https://github.com/saezlab/networkcommons/issues/

.. .. |PyPIDownloads| image:: https://static.pepy.tech/badge/decoupler
..    :target: https://pepy.tech/project/decoupler
   
.. |Docs| image:: https://readthedocs.org/projects/networkcommons/badge/?version=22-add-documentation
   :target: https://networkcommons.readthedocs.io/en/22-add-documentation/

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

Mission statement
=================
Thirty years ago, microarrays revolutionized the study of biological processes, making the computational analysis of vast molecular data essential for understanding phenotypes systematically. This shift gave rise to **network biology**, a field that has developed numerous approaches based on diverse networks, algorithmic assumptions, and omics data types.
At SaezLab, our focus is on large-scale studies of signaling processes, typically represented as directed, and sometimes signed, networks. These networks encode the transmission of molecular activation states among proteins, enabling us to map cellular functions based on data and prior knowledge.

In a **recent review**, we identified several challenges in large-scale modeling of signaling networks, including the **lack of benchmarks** and the need for a **unified technical implementation** comprising data, methods, and evaluation strategies.

The **NetworkCommons** initiative seeks to fill this gap, and aims to bring together the field around a common technical framework, aimed at standardising and advancing network biology methods. Our prototype focuses on directed and/or signed signaling networks, which we integrate with transcriptomics or phosphoproteomics data using various algorithms to create context-specific networks for evaluation and visualization.
We envision this initiative as a critical technical advancement that will facilitate the **comparison**, **development**, and **reuse of novel and existing methods**. Our vignettes demonstrate how to utilize the implemented elements in this prototype for analyzing different types of omics data. 

.. Note::
   We are in the early stages of development and welcome any contributions aligned with our mission. 
   Please use our GitHub Issues for discussions and questions.


License
=======
The data redistributed by OmniPath does not have a license, each original resource carries their own. 
`Here <https://omnipathdb.org/info>`_ one can find the license information of all the resources in OmniPath.

.. Citation
.. -------
.. Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D., Müller-Dott S., Taus P., Dugourd A., Holland C.H., 
.. Ramirez Flores R.O. and Saez-Rodriguez J. 2022. decoupleR: ensemble of computational methods to infer biological activities 
.. from omics data. Bioinformatics Advances. https://doi.org/10.1093/bioadv/vbac016

