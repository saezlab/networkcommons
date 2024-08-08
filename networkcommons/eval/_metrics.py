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
    'get_metric_from_networks',
    'get_ec50_evaluation',
    'run_ora',
    'perform_random_controls',
    'get_phosphorylation_status',
    'shuffle_dict_keys'
]

import pandas as pd
import networkx as nx
import decoupler as dc
import numpy as np

import networkcommons._utils as utils

import random


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
        network (nx.Graph, dict): The network to get the graph metrics from. If a dictionary, will iterate over it.
        target_dict (dict): A dictionary containing the targets and sign
            of measurements.

    Returns:
        DataFrame: The graph metrics of the network.
    """

    if isinstance(network, dict):
        metrics = pd.DataFrame()
        for network_name, graph in network.items():
            network_df = get_graph_metrics(graph, target_dict)
            network_df['network'] = network_name
            metrics = pd.concat([metrics, network_df])

        metrics.reset_index(inplace=True, drop=True)

    elif isinstance(network, nx.DiGraph):
        metrics = pd.DataFrame({
            'Number of nodes': get_number_nodes(network),
            'Number of edges': get_number_edges(network),
            'Mean degree': get_mean_degree(network),
            'Mean betweenness': get_mean_betweenness(network),
            'Mean closeness': get_mean_closeness(network),
            'Connected targets': get_connected_targets(network, target_dict)
        }, index=[0])
    else:
        raise TypeError("The network must be a networkx graph or a dictionary of networkx graphs.")

    return metrics


def get_metric_from_networks(networks, function, **kwargs):
    """
    Get the graph metrics of multiple networks.

    Args:
        networks (Dict[str, nx.Graph]): A dictionary of network names and
            their corresponding graphs.
        function (function): The function to get the graph metrics.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        DataFrame: The graph metrics of the networks.
    """
    metrics = pd.DataFrame()
    try:
        callable(function)
        if not callable(function):
            raise NameError(f"Function {function} not callable from the given environment.")
    except (KeyError, AttributeError, NameError):
        raise NameError(f"Function {function} not found in available functions.")

    for network_name, graph in networks.items():
        network_df = function(graph, **kwargs)
        network_df['network'] = network_name
        if 'random' in network_name:
            network_df['type'] = 'random'
        else:
            network_df['type'] = 'real'
        network_df['method'] = network_name.split('__')[0]
        metrics = pd.concat([metrics, network_df])

    metrics.reset_index(inplace=True, drop=True)

    return metrics


def get_ec50_evaluation(network, ec50_dict):
    """
    Get the EC50 evaluation of a network.
    This evaluation approach is based on the assumption that those
    elements important to explain the measurements will be more sensitive
    to the perturbation (lower EC50) than those less related to said
    perturbation.
    Produces three columns
    - 'avg_EC50': The average EC50 value of the network.
    - 'nodes_with_EC50': The number of nodes with an EC50 value.
    - 'coverage': The percentage of nodes with an EC50 value.


    Args:
        network (nx.Graph): The network to get the nodes from.
        ec50_dict (dict): A dictionary containing the EC50 values.

    Returns:
        DataFrame: The EC50 evaluation of the network.
    """

    ec50_values_in = [ec50_dict[node] for node in ec50_dict.keys() if node in network.nodes()]
    ec50_values_out = [ec50_dict[node] for node in ec50_dict.keys() if node not in network.nodes()]

    num_nodes = len(network.nodes())
    coverage = (len(ec50_values_in) / num_nodes * 100) if num_nodes > 0 else np.nan

    return pd.DataFrame({
        'avg_EC50_in': np.mean(ec50_values_in),
        'avg_EC50_out': np.mean(ec50_values_out),
        'nodes_with_EC50': len(ec50_values_in),
        'coverage': coverage
    }, index=[0])


def run_ora(graph, net, metric='ora_Combined score', ascending=False, **kwargs):
    """
    Run over-representation analysis on a custom set of genes.

    Args:
        graph (nx.Graph, nx.DiGraph): A (contextualised) graph.
        net (pd.DataFrame): A DataFrame containing source (gene set name) and
        target columns (elements which will be mapped to the network nodes),
        and containing the gene sets of interest.
        **kwargs: Additional keyword arguments to pass to the function
        decoupler.get_ora_df().

    Returns:
        pd.DataFrame: The results of the over-representation analysis.
    """
    custom_set = list(graph.nodes())

    ora_results = dc.get_ora_df(
        df=custom_set,
        net=net,
        **kwargs)

    # append ora_ to colnames
    ora_results.columns = ['ora_' + col for col in ora_results.columns]

    ora_results['ora_rank'] = ora_results[metric].rank(ascending=ascending, method='min')

    return ora_results


def perform_random_controls(graph,
                            inference_function,
                            n_iterations,
                            network_name,
                            randomise_measurements=True,
                            item_list=None,
                            **kwargs):
    """
    Performs random controls of a network by shuffling node labels and running the inference function.

    Parameters:
    graph (nx.DiGraph): The original directed graph.
    inference_function (function): The network inference function to apply.
    n_iterations (int): The number of iterations to perform.
    network_name (str): The base name for the networks in the resulting dictionary.
    randomise_measurements (bool, optional): Whether to randomise the measurements. Defaults to True.
    Requires a target_dict in the kwargs and an item_list with the possible labels.
    item_list (list, optional): A dictionary containing the measurements to randomise. Defaults to None.
    If randomise_measurements is True, this is required.
    **kwargs: Additional keyword arguments to pass to the inference function.

    Returns:
    dict: A dictionary containing the inferred networks.
    """
    # Initialize the dictionary to store the networks
    inferred_networks = {}

    # Get the list of nodes
    nodes = list(graph.nodes)

    for i in range(n_iterations):
        # Shuffle the node labels
        shuffled_nodes = nodes[:]
        random.shuffle(shuffled_nodes)

        # Create a mapping from original to shuffled node labels
        mapping = {original: shuffled for original, shuffled in zip(nodes, shuffled_nodes)}

        # Relabel the nodes in the graph
        shuffled_graph = nx.relabel_nodes(graph, mapping, copy=True)

        # Shuffle the target_dict if required
        if randomise_measurements and item_list:
            shuffled_target_dict = shuffle_dict_keys(kwargs['target_dict'], item_list)
            # Update kwargs with the shuffled target_dict
            kwargs['target_dict'] = shuffled_target_dict

        # Perform the network inference on the shuffled graph
        inferred_network, _ = inference_function(shuffled_graph, **kwargs)

        # Add the inferred network to the dictionary with a unique name
        network_label = f"{network_name}__random{i+1:03d}"
        inferred_networks[network_label] = inferred_network

    return inferred_networks


def shuffle_dict_keys(dictionary, items):
    """
    Shuffle the keys of a dictionary.

    Args:
        dictionary (dict): The dictionary to shuffle.
        items (list): The list of items to shuffle.

    Returns:
        dict: The dictionary with shuffled keys.
    """
    old_labels = list(dictionary.keys())
    new_labels = random.sample(items, len(dictionary))

    random_dict = {new_label: dictionary[old_label] for old_label, new_label in zip(old_labels, new_labels)}

    return random_dict


def get_phosphorylation_status(network, dataframe, col='stat'):
    """
    Calculates the phosphorylation status metrics for a given network and dataframe.

    Parameters:
    network (nx.DiGraph): The network graph.
    dataframe (pandas DataFrame): The dataframe containing the phosphorylation data.
    col (str, optional): The column name in the dataframe to use to infer the metrics. Defaults to 'stat' 
    (from nc.data.omics.deseq2()).

    Returns:
    pandas DataFrame: A dataframe containing the calculated metrics:
    - avg_relabundance: The average value of the nodes with phosphorylation information, in absolute values
    (we don't consider the sign of the dysregulation, but how severe it is).
    - avg_relabundance_overall: The average value of all the elements, irrespective of whether
    they are present in the network or not, in absolute values.
    - diff_dysregulation: The overall difference of dysregulation between nodes in the network and those
    outside of it.
    - nodes_with_phosphoinfo: The number of nodes with phosphorylation information.
    - coverage: The percentage of nodes with phosphorylation information.
    """

    subset_df = utils.subset_df_with_nodes(network, dataframe)
    metric_in = abs(subset_df[col].values)
    metric_out = abs(dataframe[col].values)
    coverage = len(subset_df) / len(network.nodes) * 100 if len(network.nodes) > 0 else 0

    return pd.DataFrame({
        'avg_relabundance': np.mean(metric_in),
        'avg_relabundance_overall': np.mean(metric_out),
        'diff_dysregulation': np.mean(metric_in) - np.mean(metric_out),
        'nodes_with_phosphoinfo': len(subset_df),
        'coverage': coverage
    }, index=[0])