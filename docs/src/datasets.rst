####
Data
####
NetworkCommons provides a collection of omics datasets and prior knowledge resources. The datasets are available in the form of files that can be downloaded and used for further analysis. The prior knowledge resources are available in the form of networks (either Network objects or pd.DataFrames).
All the data can be accessed via the NetworkCommons API.

----------
Omics data
----------
Below, we provide a list of all the omics datasets currently available in NetworkCommons. For each data, we provide a link to the original publication, a description, processing (if applicable), and a link to the data location.


DecryptM
--------

**Alias:** decryptm

**Description:** Drug perturbation proteomics and phosphoproteomics data

**Publication Link:** `Jana Zecha et al. Decrypting drug actions and protein modifications by dose- and time-resolved proteomics. Science 380,93-101(2023). <https://doi.org/10.1126/science.ade3925>`_

**Data location:** `PRIDE <https://ftp.pride.ebi.ac.uk/pride/data/archive/2023/03/PXD037285/>`_

**Detailed Description:** This dataset contains the profiling of 31 cancer drugs in 13 human cancer cell line models, resulting in 1.8 million dose-response curves. The data includes 47,502 regulated phosphopeptides, 7316 ubiquitinylated peptides, and 546 regulated acetylated peptides.
Networkcommons contains the files containing, per phosphosite, EC50 values obtained from fitting the intensity values of the 10 drug concentration points to a four-parameter logistic function.


PANACEA
-------

**Alias:** panacea

**Description:** Pancancer Analysis of Chemical Entity Activity RNA-Seq data

**Publication Link:** `Eugene F. Douglass et al. A community challenge for a pancancer drug mechanism of action inference from perturbational profile data. Cell Reports Medicine (2022). <https://doi.org/10.1016/j.xcrm.2021.100492>`_

**Data location:** `NCBI GEO <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186341>`_

**Detailed Description:** PANACEA contains dose-response and perturbational profiles for 32 kinase inhibitors in 11 cancer cell lines, in addition to a DMSO control. Originally, this resource served as the basis for a DREAM Challenge assessing the accuracy and sensitivity of computational algorithms for de novo drug polypharmacology predictions.
NetworkCommons provides raw files for countdata and metadata, as retrieved in the original page. In addition, differential expression and TF activity tables are provided.

**Data processing:** The differential expression statistics were obtained via `FLOP <https://doi.org/10.1093/nar/gkae552>`_, using FilterbyExpr and DESeq2, one of the top performer combinations in the benchmarking study.
The contrasts were set, per cell line, between each drug and the DMSO control. The TF activity tables were obtained also via `FLOP <https://doi.org/10.1093/nar/gkae552>`_, using univariate linear models as implemented 
in `decoupler <https://doi.org/10.1093/bioadv/vbac016>`_.


CPTAC
-----

**Alias:** CPTAC

**Description:** Clinical Proteomic Tumor Analysis Consortium data

**Publication Link:** `Ellis, M. J. et al. Connecting genomic alterations to cancer biology with proteomics: the NCI Clinical Proteomic Tumor Analysis Consortium. Cancer Discov. 3, 1108–1112 (2013). <https://doi.org/10.1158/2159-8290.CD-13-0219>`_

**Data location:** `NIH NCI Proteommic Data Commons <https://pdc.cancer.gov/pdc/cptac-pancancer>`_

**Detailed Description:** This dataset contains data from the Clinical Proteomic Tumor Analysis Consortium. It includes various cancer types and proteomic data.
We included only the data processed by the University of Michigan team's pipeline, and then post-processed by the Baylor College of Medicine's pipeline. Details 
can be found in the STAR Methods of `'Proteogenomic Data and Resources for Pan-Cancer Analysis' <https://doi.org/10.1016/j.ccell.2023.06.009>`_ (i.e., 'BCM pipeline for pan-cancer multi-omics data harmonization').


NCI60
-----

**Alias:** NCI60

**Description:** NCI-60 cell line data

**Publication Link:** `Shoemaker, R. The NCI60 human tumour cell line anticancer drug screen. Nat Rev Cancer 6, 813–823 (2006). <https://doi.org/10.1038/nrc1951>`_

**Data location:** `COSMOS R package - Bioconductor <https://www.bioconductor.org/packages/release/bioc/html/cosmosR.html>`_

**Detailed Description:** This dataset contains data from the NCI-60 cell line panel. It includes three files: TF activities from transcriptomics data, metabolite abundances, and gene reads.

---------------
Prior Knowledge
---------------
Below, we provide a list of all the prior knowledge resources currently available in NetworkCommons. For each resource, we provide a description and a link to the original publication.

OmniPath
--------

**Alias:** omnipath

**Description:** OmniPath database

**Publication Link:** `Türei, D. et al. OmniPath: guidelines and gateway for literature-curated signaling pathway resources. Nat Methods 13, 966–967 (2016). <https://doi.org/10.1038/nmeth.4077>`_

**Detailed Description:** OmniPath is a comprehensive collection of signaling pathways and regulatory interactions. Currently, NetworkCommons include the signed and directed PPI network that can be obtained from Omnipath.Interactions. 
Our aim is to expand the API to more data sources within OmniPath. For more information, please refer to the `OmniPath website <https://omnipathdb.org/>`_ and the `OmniPath documentation page <https://omnipath.readthedocs.io/>`_.

Liana
-----

**Alias:** liana

**Description:** Liana database

**Publication Link:** `Dimitrov, D., Türei, D., Garrido-Rodriguez, M. et al. Comparison of methods and resources for cell-cell communication inference from single-cell RNA-Seq data. Nat Commun 13, 3224 (2022). <https://doi.org/10.1038/s41467-022-30755-0>`_

**Detailed Description:** The Prior Knowledge from Liana contains ligand-receptor interactions. For more information, please refer to the `Liana documentation page <https://liana-py.readthedocs.io/en/latest/>`_.

PhosphositePlus
---------------

**Alias:** phosphositeplus

**Description:** PhosphositePlus database

**Publication Link:** `Hornbeck, P. V. et al. PhosphoSitePlus, 2014: mutations, PTMs and recalibrations. Nucleic Acids Res 43, D512–D520 (2015). <https://doi.org/10.1093/nar/gku1267>`_

**Detailed Description:** PhosphositePlus is a comprehensive resource that contains, among other PTM interactions, kinase-subsrate interactions, which can then be useful to infer kinase activities from phosphoproteomics data. 
For more information, please refer to the `PhosphositePlus website <https://www.phosphosite.org/>`_.