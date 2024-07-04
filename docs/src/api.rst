===
API
===

Import NetworkCommons as::

    import networkcommons as nc

Methods
=======

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


Topological methods
~~~~
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
    methods.add_pagerank_scores
    methods.compute_ppr_overlap


CORNETO
~~~~
.. module::networkcommons.methods
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    methods.run_corneto_carnival
    methods.to_cornetograph
    methods.to_networkx   



Prior Knowledge
===============

.. module::networkcommons.data.network
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.network.get_omnipath
    data.network.get_lianaplus


Datasets
========

Utils
~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.datasets
    data.omics.deseq2

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

PANACEA
~~~~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.panacea

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

Other
~~~~~~~~
.. module::networkcommons.data.omics
.. currentmodule:: networkcommons

.. autosummary::
    :toctree: api
    :recursive:

    data.omics.moon


Evaluation
==========

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
    eval.get_metrics_from_networks
    

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
    visual.get_comparison_color
    visual.get_edge_color
    visual.update_node_property
    visual.update_edge_property
    visual.apply_node_style
    visual.apply_edge_style
    visual.build_volcano_plot
    visual.build_ma_plot
    visual.build_pca_plot
    visual.build_heatmap_with_tree
    vlsual.visualize_graph_simple
    visual.lollipop_plot
