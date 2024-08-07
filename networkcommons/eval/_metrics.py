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
Basic graph metrics.
"""

from __future__ import annotations

__all__ = [
    'get_number_nodes',
    'get_number_edges',
    'get_mean_degree',
    'get_mean_betweenness',
    'get_mean_closeness',
    'get_connected_targets',
    'get_recovered_offtargets',
    'get_graph_metrics',
    'get_metrics_from_networks'
]

import pandas as pd
import networkx as nx
import numpy as np


def get_number_nodes(network: nx.Graph) -> int:
    """
    Get the number of nodes in the network.

    Args:
        network (nx.Graph): The network to get the number of nodes from.

    Returns:
        int: The number of nodes in the network.
    """

    return network.number_of_nodes()


def get_number_edges(network: nx.Graph) -> int:
    """
    Get the number of edges in the network.

    Args:
        network (nx.Graph): The network to get the number of edges from.

    Returns:
        int: The number of edges in the network.
    """

    return network.number_of_edges()


def get_mean_degree(network: nx.Graph) -> float:
    """
    Get the mean degree centrality of the network.

    Args:
        network (nx.Graph): The network to get the mean degree from.

    Returns:
        float: The mean degree of the network.
    """

    return np.mean(list(dict(nx.degree(network)).values()))


def get_mean_betweenness(network: nx.Graph) -> float:
    """
    Get the mean betweenness centrality of the network.

    Args:
        network (nx.Graph): The network to get the mean betweenness from.

    Returns:
        float: The mean betweenness of the network.
    """

    return np.mean(list(nx.betweenness_centrality(network).values()))


def get_mean_closeness(network: nx.Graph) -> float:
    """
    Get the mean closeness centrality of the network.

    Args:
        network (nx.Graph): The network to get the mean closeness from.

    Returns:
        float: The mean closeness of the network.
    """

    return np.mean(list(nx.closeness_centrality(network).values()))


def get_connected_targets(network, target_dict, percent=False):
    """
    Get the number of connected targets in the network.

    Args:
        network (nx.Graph): The network to get the connected targets from.
        source_dict (dict): A dictionary containing the sources and sign
            of perturbation.
        target_dict (dict): A dictionary containing the targets and sign
            of measurements.
        percent (bool, optional): If True, return the percentage of connected
            targets. Defaults to False.

    Returns:
        int: The number of connected targets in the network.
    """

    target_nodes = list(target_dict.keys())

    connected_nodes = len(set(target_nodes).intersection(set(network.nodes())))

    if percent:
        return connected_nodes / len(target_nodes) * 100
    else:
        return connected_nodes


def get_recovered_offtargets(network, offtargets, percent=False):
    """
    Get the number of off-targets recovered by the network.

    Args:
        network (nx.Graph): The network to get the off-targets from.
        offtargets (list): A list of off-targets.
        percent (bool, optional): If True, return the percentage of
            off-targets recovered. Defaults to False.

    Returns:
        int: The number/percentage of off-targets recovered by the network.
    """

    recovered_offtargets = len(set(offtargets).intersection(
        set(network.nodes()))
        )

    if percent:
        return recovered_offtargets / len(offtargets) * 100
    else:
        return recovered_offtargets


def get_graph_metrics(network, target_dict):
    """
    Get the graph metrics of a network.

    Args:
        network (nx.Graph): The network to get the graph metrics from.
        target_dict (dict): A dictionary containing the targets and sign
            of measurements.

    Returns:
        DataFrame: The graph metrics of the network.
    """

    metrics = {
        'Number of nodes': get_number_nodes(network),
        'Number of edges': get_number_edges(network),
        'Mean degree': get_mean_degree(network),
        'Mean betweenness': get_mean_betweenness(network),
        'Mean closeness': get_mean_closeness(network),
        'Connected targets': get_connected_targets(network, target_dict)
    }

    return pd.DataFrame(metrics, index=[0])


def get_metrics_from_networks(networks, target_dict):
    """
    Get the graph metrics of multiple networks.

    Args:
        networks (Dict[str, nx.Graph]): A dictionary of network names and
            their corresponding graphs.
        target_dicts (Dict[str, dict]): A dictionary of target dictionaries
            for each network.

    Returns:
        DataFrame: The graph metrics of the networks.
    """
    metrics = pd.DataFrame()
    for network_name, graph in networks.items():
        network_df = get_graph_metrics(graph, target_dict)
        network_df['network'] = network_name
        metrics = pd.concat([metrics, network_df])

    metrics.reset_index(inplace=True, drop=True)

    return metrics
