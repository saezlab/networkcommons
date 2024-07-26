#!/usr/bin/env python

#
# This file is part of the `networkcommons` Python module
#
# Copyright 2024
# Heidelberg University Hospital
#
# File author(s): Saez Lab (omnipathdb@gmail.com)
#
# Distributed under the GPLv3 license
# See the file `LICENSE` or read a copy at
# https://www.gnu.org/licenses/gpl-3.0.txt
#

"""
Causal network inference methods.
"""

from __future__ import annotations

__all__ = [
    'to_cornetograph',
    'to_networkx',
    'run_corneto_carnival',
]

import lazy_import
import networkx as nx
# cn = lazy_import.lazy_module('corneto')
# cn_nx = lazy_import.lazy_module('corneto.contrib.networkx')
import corneto as cn
import corneto.contrib.networkx as cn_nx

from .. import _utils


def run_corneto_carnival(network,
                         source_dict,
                         target_dict,
                         betaWeight=0.2,
                         solver=None,
                         verbose=True):
    """
    Run the Vanilla Carnival algorithm via CORNETO.

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign
            of perturbation.
        target_dict (dict): A dictionary containing the targets and sign
            of measurements.

    Returns:
        nx.Graph: The subnetwork containing the paths found by CARNIVAL.
        list: A list containing the paths found by CARNIVAL.
    """
    corneto_net = _utils.to_cornetograph(network)

    problem, graph = cn.methods.runVanillaCarnival(
        perturbations=source_dict,
        measurements=target_dict,
        priorKnowledgeNetwork=corneto_net,
        betaWeight=betaWeight,
        solver=solver,
        verbose=verbose
    )

    network_sol = graph.edge_subgraph(
        cn.methods.carnival.get_selected_edges(problem, graph),
    )

    network_nx = _utils.to_networkx(network_sol, skip_unsupported_edges=True)

    network_nx.remove_nodes_from(['_s', '_pert_c0', '_meas_c0'])

    return network_nx
