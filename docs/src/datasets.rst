####
Data
####
NetworkCommons provides a collection of omics datasets and prior knowledge resources. The datasets are available in the form of files that can be downloaded and used for further analysis. The prior knowledge resources are available in the form of networks (either Network objects or pd.DataFrames).
All the data can be accessed via the NetworkCommons API. 
If you want to contribute with your own, please check our :doc:`Contribution guidelines <guidelines/guide_1_data>`.

.. _details-omics:

----------
Omics data
----------
Below, we provide a list of all the omics datasets currently available in NetworkCommons. For each data, we provide a link to the original publication, a description, processing (if applicable), and a link to the data location.

.. _details-decryptm:

DecryptM
--------

**Alias:** decryptm

**Description:** Drug perturbation proteomics and phosphoproteomics data

**Publication Link:** `Jana Zecha et al. Decrypting drug actions and protein modifications by dose- and time-resolved proteomics. Science 380,93-101(2023). <https://doi.org/10.1126/science.ade3925>`_

**Data location:** `PRIDE <https://www.ebi.ac.uk/pride/archive/projects/PXD037285>`_

**Detailed Description:** This dataset contains the profiling of 31 cancer drugs in 13 human cancer cell line models, resulting in 1.8 million dose-response curves. The data includes 47,502 regulated phosphopeptides, 7316 ubiquitinylated peptides, and 546 regulated acetylated peptides.
Networkcommons contains the files containing, per phosphosite, EC50 values obtained from fitting the intensity values of the 10 drug concentration points to a four-parameter logistic function.

**Functions:** See API documentation for :ref:`DecryptM <api-decryptm>`.

.. _details-panacea:

PANACEA
-------

**Alias:** panacea

**Description:** Pancancer Analysis of Chemical Entity Activity RNA-Seq data

**Publication Link:** `Eugene F. Douglass et al. A community challenge for a pancancer drug mechanism of action inference from perturbational profile data. Cell Reports Medicine (2022). <https://doi.org/10.1016/j.xcrm.2021.100492>`_

**Data location:** `NCBI GEO <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186341>`_

**Detailed Description:** 	1728 RNAseq profiles of a cell-line perturbed with 32 kinase inhibitors. Cell-lines were treated at a 48-hour IC20 for each drug and RNA was collected at 24 hours to minimize cell-death.
Samples were prepared and sequences in batches of 96 that included 6 vehicle (DMSO). 2 replicates were taken for each drug-cell-line pair. Originally, this resource served as the basis for a DREAM Challenge assessing
the accuracy and sensitivity of computational algorithms for de novo drug polypharmacology predictions. 
NetworkCommons provides raw files for countdata and metadata, as retrieved in the original page. In addition, differential expression and TF activity tables are provided. 

**Data processing:** The differential expression statistics were obtained via `FLOP <https://doi.org/10.1093/nar/gkae552>`_, using FilterbyExpr and DESeq2, one of the top performer combinations in the benchmarking study.
The contrasts were set, per cell line, between each drug and the DMSO control. The TF activity tables were obtained also via `FLOP <https://doi.org/10.1093/nar/gkae552>`_, using univariate linear models as implemented 
in `decoupler <https://doi.org/10.1093/bioadv/vbac016>`_.

**Functions:** See API documentation for :ref:`PANACEA <api-panacea>`.

.. _details-cptac:

CPTAC
-----

**Alias:** CPTAC

**Description:** Clinical Proteomic Tumor Analysis Consortium data

**Publication Link:** `Ellis, M. J. et al. Connecting genomic alterations to cancer biology with proteomics: the NCI Clinical Proteomic Tumor Analysis Consortium. Cancer Discov. 3, 1108–1112 (2013). <https://doi.org/10.1158/2159-8290.CD-13-0219>`_

**Data location:** `NIH NCI Proteommic Data Commons <https://pdc.cancer.gov/pdc/cptac-pancancer>`_

**Detailed Description:** This dataset contains data from the Clinical Proteomic Tumor Analysis Consortium. It includes various cancer types and proteomic data.
We included only the data processed by the University of Michigan team's pipeline, and then post-processed by the Baylor College of Medicine's pipeline. Details 
can be found in the STAR Methods of `'Proteogenomic Data and Resources for Pan-Cancer Analysis' <https://doi.org/10.1016/j.ccell.2023.06.009>`_ (i.e., 'BCM pipeline for pan-cancer multi-omics data harmonization').

**Functions:** See API documentation for :ref:`CPTAC <api-cptac>`.

.. _details-nci60:

NCI60
-----

**Alias:** NCI60

**Description:** NCI-60 cell line data

**Publication Link:** `Shoemaker, R. The NCI60 human tumour cell line anticancer drug screen. Nat Rev Cancer 6, 813–823 (2006). <https://doi.org/10.1038/nrc1951>`_

**Data location:** `COSMOS R package - Bioconductor <https://www.bioconductor.org/packages/release/bioc/html/cosmosR.html>`_

**Detailed Description:** This dataset contains data from the NCI-60 cell line panel. It includes three files: TF activities from transcriptomics data, metabolite abundances, and gene reads.

**Functions:** See API documentation for :ref:`NCI60 <api-nci60>`.

.. _details-pk:


Phosphoproteomics in response to EGF
-----

**Alias:** PhosphoEGF

**Description:** A meta-analysis of phosphoproteomics data in response to EGF stimulation

**Publication Link:** `Garrido-Rodriguez et al. Evaluating signaling pathway inference from kinase-substrate interactions and phosphoproteomics data. bioRxiv (2024). <https://www.biorxiv.org/content/10.1101/2024.10.21.619348v1>`_

**Data location:** `Supplementary Data files of the manuscript <https://www.biorxiv.org/content/10.1101/2024.10.21.619348v1.supplementary-material>`_

**Detailed Description:** This dataset the results of a meta-analysis of phosphoproteomics data in response to EGF stimulation across different labs and stimulation times. The data is available at two different levels. First, the phosphosite differential abundance is provided for every combination of study and treatment time. In the table, 'This study' refers to the data generated in the manuscript. Second, we offer access to the kinase-level activities inerred using decoupleR and the different kinase-substrate networks described in the paper. Briefly, four different networks were employed: A first one based on literature (literature), one based on kinase-substrate interaction prediction via protein language models (phosformer), one based on positionl peptide array screening (kinlibrary) and a combination of all of them (combined).

**Functions:** See API documentation for :ref:`Phospho-EGF meta-analysis<api-phosphoegf>`.

.. _details-pk:


---------------
Prior Knowledge
---------------
Below, we provide a list of all the prior knowledge resources currently available in NetworkCommons. For each resource, we provide a description and a link to the original publication.

.. _details-omnipath:

OmniPath
--------

**Alias:** omnipath

**Description:** OmniPath database

**Publication Link:** `Türei, D. et al. OmniPath: guidelines and gateway for literature-curated signaling pathway resources. Nat Methods 13, 966–967 (2016). <https://doi.org/10.1038/nmeth.4077>`_

**Detailed Description:** OmniPath is a comprehensive collection of signaling pathways and regulatory interactions. Currently, NetworkCommons include the signed and directed PPI network that can be obtained from Omnipath.Interactions. 
Our aim is to expand the API to more data sources within OmniPath. For more information, please refer to the `OmniPath website <https://omnipathdb.org/>`_ and the `OmniPath documentation page <https://omnipath.readthedocs.io/>`_.

**Functions:** See API documentation for :ref:`Prior knowledge <api-pk>`.

.. _details-liana:

Liana
-----

**Alias:** liana

**Description:** Liana database

**Publication Link:** `Dimitrov, D., Türei, D., Garrido-Rodriguez, M. et al. Comparison of methods and resources for cell-cell communication inference from single-cell RNA-Seq data. Nat Commun 13, 3224 (2022). <https://doi.org/10.1038/s41467-022-30755-0>`_

**Detailed Description:** The Prior Knowledge from Liana contains ligand-receptor interactions. For more information, please refer to the `Liana documentation page <https://liana-py.readthedocs.io/en/latest/>`_.

**Functions:** See API documentation for :ref:`Prior knowledge <api-pk>`.

.. _details-phosphositeplus:

PhosphositePlus 
---------------

**Alias:** phosphositeplus

**Description:** PhosphositePlus database

**Publication Link:** `Hornbeck, P. V. et al. PhosphoSitePlus, 2014: mutations, PTMs and recalibrations. Nucleic Acids Res 43, D512–D520 (2015). <https://doi.org/10.1093/nar/gku1267>`_

**Detailed Description:** PhosphositePlus is a comprehensive resource that contains, among other PTM interactions, kinase-subsrate interactions, which can then be useful to infer kinase activities from phosphoproteomics data. 
For more information, please refer to the `PhosphositePlus website <https://www.phosphosite.org/>`_.

**Functions:** See API documentation for :ref:`Prior knowledge <api-pk>`.