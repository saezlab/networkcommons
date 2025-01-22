.. image:: _static/nc_banner.svg
   :width: 1200px
   :align: center

##############
Bridging the gap between data, methods and knowledge in the network inference field
##############

|MainBuild| |Codecov| |Docs|

.. |MainBuild| image:: https://github.com/saezlab/networkcommons/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/saezlab/networkcommons/actions
   
.. .. |Issues| image:: https://img.shields.io/github/issues/saezlab/networkcommons.svg
..    :target: https://github.com/saezlab/networkcommons/issues/

.. .. |PyPIDownloads| image:: https://static.pepy.tech/badge/decoupler
..    :target: https://pepy.tech/project/decoupler
   
.. |Docs| image:: https://readthedocs.org/projects/networkcommons/badge/?version=main
   :target: https://networkcommons.readthedocs.io/en/main/?badge=main
   :alt: Documentation Status

.. |Codecov| image:: https://codecov.io/github/saezlab/networkcommons/graph/badge.svg?token=RH438ALJC2
   :target: https://codecov.io/gh/saezlab/networkcommons

NetworkCommons is a community-driven platform designed to simplify access to tools and resources for 
inferring context-specific protein interaction networks by integrating context-agnostic prior knowledge with omics data. 
These networks have a wide range of applications, from omics data interpretation to identifying drug targets and key driver 
mutations.

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_graphabs.svg" width="1000px" alt="Networkcommons API" class="no-scaled-link" style="display: block; margin: 0 auto;"></object>


Addressing the complexity and fragmentation of data, methods, and tools across multiple repositories, NetworkCommons 
offers an API that provides access to raw and preprocessed omics data, directed and/or signed template graphs, and various network 
inference methods. By streamlining accessibility, we aim to support systematic benchmarking, enhance critical assessment, and foster 
collaborative advancements in network biology.

Mission statement
=================
Thirty years ago, microarrays revolutionized the study of biological processes, making the computational analysis of vast molecular data essential for understanding phenotypes systematically. This shift increased usage of **network biology**, a field that has developed numerous approaches based on diverse networks, algorithmic assumptions, and omics data types.
At SaezLab, our focus is on large-scale studies of signaling processes, typically represented as directed, and sometimes signed, networks. These networks encode the transmission of molecular activation states among proteins, enabling us to map cellular functions based on data and prior knowledge.

In a  `recent review <https://www.embopress.org/doi/full/10.15252/msb.202211036>`_, we identified several challenges in large-scale modeling of signaling networks, including the **lack of benchmarks** and the need for a **unified technical implementation** comprising data, methods, and evaluation strategies.

The **NetworkCommons** initiative seeks to fill this gap, and aims to bring together the field around a common technical framework, aimed at standardising and advancing network biology methods. Our prototype focuses on directed and/or signed signaling networks, which we integrate with transcriptomics or phosphoproteomics data using various algorithms to create context-specific networks for evaluation and visualization.
We envision this initiative as a critical technical advancement that will facilitate the **comparison**, **development**, and **reuse of novel and existing methods**. Our vignettes demonstrate how to utilize the implemented elements in this prototype for analyzing different types of omics data. 

.. Note::
   We are in the early stages of development and welcome any contributions aligned with our mission. 
   Please use our GitHub Issues for discussions and questions.


License
=======
The data redistributed by OmniPath does not have a license, each original resource carries their own. 
`Here <https://omnipathdb.org/info>`_ one can find the license information of all the resources in OmniPath.

Citation
-------
Victor Paton, Denes Türei, Olga Ivanova, Sophia Müller-Dott, Pablo Rodriguez-Mier, Veronica Venafra, Livia Perfetto, Martin Garrido-Rodriguez, Julio Saez-Rodriguez
bioRxiv 2024.11.22.624823; doi: https://doi.org/10.1101/2024.11.22.624823

