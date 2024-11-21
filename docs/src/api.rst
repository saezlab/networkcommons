===
API
===

Import NetworkCommons as::

    import networkcommons as nc

.. _api-methods:

Methods
=======

.. _api-moon:

MOON
~~~~
.. module::networkcommons.methods
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    methods.prepare_metab_inputs
    methods.is_expressed
    methods.filter_pkn_expressed_genes
    methods.filter_input_nodes_not_in_pkn
    methods.keep_controllable_neighbours
    methods.keep_observable_neighbours
    methods.compress_same_children
    methods.run_moon_core
    methods.filter_incoherent_TF_target
    methods.decompress_moon_result
    methods.reduce_solution_network
    methods.translate_res

.. _api-topological:

Topological methods
~~~~~~~~~~~~~~~~~~~
.. module::networkcommons.methods
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    methods.run_shortest_paths
    methods.run_sign_consistency
    methods.run_reachability_filter
    methods.run_all_paths
    methods.compute_all_paths

.. _api-rwr:

Random Walk with Restart
~~~~~~~~~~~~~~~~~~~~~~~~
.. module::networkcommons.methods
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    methods.add_pagerank_scores
    methods.compute_ppr_overlap

.. _api-corneto:

CORNETO
~~~~
.. module::networkcommons.methods
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    methods.run_corneto_carnival


.. _api-signalingprofiler:

SignalingProfiler
=================
.. module::networkcommons.methods
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    methods.run_signalingprofiler

.. _api-pk:

Prior Knowledge
===============

.. module::networkcommons.data.network
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.network.get_omnipath
    data.network.get_lianaplus
    data.network.get_phosphositeplus
    data.network.get_cosmos_pkn

.. _api-data:

Datasets
========

.. _api-data-utils:

Utils
~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.datasets
    data.omics.deseq2
    data.omics.get_ensembl_mappings
    data.omics.convert_ensembl_to_gene_symbol

.. _api-decryptm:

DecryptM
~~~~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.decryptm_datasets
    data.omics.decryptm_table
    data.omics.decryptm_experiment

.. _api-panacea:

PANACEA
~~~~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.panacea_experiments
    data.omics.panacea_datatypes
    data.omics.panacea_tables
    data.omics.panacea_gold_standard

.. _api-scperturb:

scPerturb
~~~~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.scperturb
    data.omics.scperturb_metadata
    data.omics.scperturb_datasets

.. _api-cptac:

CPTAC
~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.cptac_cohortsize
    data.omics.cptac_fileinfo
    data.omics.cptac_table
    data.omics.cptac_datatypes
    data.omics.cptac_extend_dataframe

.. _api-nci60:

NCI60
~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.nci60_datasets
    data.omics.nci60_datatypes
    data.omics.nci60_table

.. _api-phosphoegf:

Phospho-EGF meta-analysis
~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.phospho_egf_datatypes
    data.omics.phospho_egf_tables

.. _api-eval:

Evaluation and description
==========================

.. module::networkcommons.eval
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    eval.get_number_nodes
    eval.get_number_edges
    eval.get_mean_degree
    eval.get_mean_betweenness
    eval.get_mean_closeness
    eval.get_connected_targets
    eval.get_recovered_offtargets
    eval.get_graph_metrics
    eval.get_metric_from_networks
    eval.get_ec50_evaluation
    eval.run_ora
    eval.get_phosphorylation_status
    eval.perform_random_controls
    eval.shuffle_dict_keys

.. _api-vis:

Visualization
=============

.. module::networkcommons.visual
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    visual.NetworkXVisualizer
    visual.YFilesVisualizer
    visual.get_styles
    visual.set_style_attributes
    visual.merge_styles
    visual.get_comparison_color
    visual.get_edge_color
    visual.update_node_property
    visual.update_edge_property
    visual.apply_node_style
    visual.apply_edge_style
    visual.build_volcano_plot
    visual.build_ma_plot
    visual.plot_pca
    visual.build_heatmap_with_tree
    visual.lollipop_plot
    visual.create_heatmap
    visual.plot_density
    visual.plot_scatter
    visual.plot_rank

.. _api-utils:

Utilities
=========

.. module::networkcommons.utils
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:


    utils.to_cornetograph
    utils.to_networkx
    utils.read_network_from_file
    utils.network_from_df
    utils.get_subnetwork
    utils.decoupler_formatter
    utils.targetlayer_formatter
    utils.handle_missing_values
    utils.subset_df_with_nodes
    utils.node_attrs_from_corneto
    utils.edge_attrs_from_corneto

