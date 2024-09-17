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

from __future__ import annotations

"""
Graph based methods.
"""

__all__ = [
    'run_shortest_paths',
    'run_sign_consistency',
    'run_reachability_filter',
    'run_all_paths',
    'compute_all_paths',
    'add_pagerank_scores',
    'compute_ppr_overlap',
]

import networkx as nx
import numpy as np

from networkcommons import utils
from networkcommons._session import session as _session

from collections import defaultdict, Counter
import random

from networkcommons._session import _log


def run_shortest_paths(network, source_dict, target_dict, verbose=False):
    """
    Calculate the shortest paths between sources and targets.

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign
            of perturbation.
        target_dict (dict): A dictionary containing the targets and sign
            of measurements.
        verbose (bool): If True, print warnings when no path is found to
            a given target.

    Returns:
        nx.Graph: The subnetwork containing the shortest paths.
        list: A list containing the shortest paths.
    """

    _log('Shortest paths: Running...')

    shortest_paths_res = []

    sources = source_dict.keys()
    targets = target_dict.keys()

    for source_node in sources:
        for target_node in targets:
            try:
                shortest_paths_res.extend([p for p in nx.all_shortest_paths(
                    network,
                    source=source_node,
                    target=target_node,
                    weight='weight'
                )])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # _session.log_traceback(console = verbose)
                pass
        
        _log(f'Shortest paths: Found {len(shortest_paths_res)} paths for source {source_node}')

    subnetwork = utils.get_subnetwork(network, shortest_paths_res)

    _log(f'Shortest paths: Network solution with {subnetwork.number_of_nodes()} nodes and {subnetwork.number_of_edges()} edges.')
    _log('Shortest paths: finished.')

    return subnetwork, shortest_paths_res


def run_sign_consistency(network, paths, source_dict, target_dict=None):
    """
    Calculate the sign consistency between sources and targets. If the target
    sign is not provided, infer the sign by majority consensus from the paths.

    Args:
        network (nx.Graph): The network.
        paths (list): A list containing the shortest paths.
        source_dict (dict): A dictionary containing the sources and sign
            of perturbation.
        target_dict (dict, optional): A dictionary containing the targets and sign
            of measurements. If None, infer the target signs by consensus.

    Returns:
        nx.Graph: The subnetwork containing the sign consistent paths.
        list: A list containing the sign consistent paths.
        dict: A dictionary containing the inferred target signs (if target_dict is None).
    """
    _log('Sign consistency: Running...')

    directed = nx.is_directed(network)
    subnetwork = nx.DiGraph() if directed else nx.Graph()
    sign_consistency_res = []
    inferred_target_signs = defaultdict(list)

    for path in paths:
        source = path[0]
        product_sign = 1
        target = path[-1]

        source_sign = source_dict[source]

        for i in range(len(path) - 1):
            edge_sign = network.get_edge_data(path[i], path[i + 1])['sign']
            product_sign *= edge_sign

        if target_dict:
            target_sign = target_dict[target]
            if np.sign(source_sign * product_sign) == np.sign(target_sign):
                sign_consistency_res.append(path)
        else:
            inferred_target_signs[target].append(np.sign(source_sign * product_sign))

    if not target_dict:
        _log('Sign consistency: No target sign provided. Inferring target signs by majority consensus.')
        inferred_target_sign = {}
        for target, signs in inferred_target_signs.items():
            sign_count = Counter(signs)
            most_common_signs = sign_count.most_common()
            if len(most_common_signs) > 1 and most_common_signs[0][1] == most_common_signs[1][1]:
                chosen_sign = random.choice([most_common_signs[0][0], most_common_signs[1][0]])
            else:
                chosen_sign = most_common_signs[0][0]
            inferred_target_sign[target] = chosen_sign

        for path in paths:
            source = path[0]
            product_sign = 1
            target = path[-1]

            source_sign = source_dict[source]
            target_sign = inferred_target_sign[target]

            for i in range(len(path) - 1):
                edge_sign = network.get_edge_data(path[i], path[i + 1])['sign']
                product_sign *= edge_sign

            if np.sign(source_sign * product_sign) == np.sign(target_sign):
                sign_consistency_res.append(path)

    subnetwork = utils.get_subnetwork(network, sign_consistency_res)

    _log(f'Sign consistency: Number of sign-inconsistent paths: {len(paths) - len(sign_consistency_res)}')
    _log(f'Sign consistency: Network solution with {subnetwork.number_of_nodes()} nodes and {subnetwork.number_of_edges()} edges.')
    _log('Sign consistency: finished.')

    if not target_dict:
        return subnetwork, sign_consistency_res, inferred_target_sign
    else:
        return subnetwork, sign_consistency_res


def run_reachability_filter(network, source_dict):
    """
    Filters out all nodes from the graph which cannot be reached from
        source(s).

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign
            of perturbation.

    Returns:
        None
    """
    _log('Reachability filter: Running...')

    source_nodes = set(source_dict.keys())
    reachable_nodes = source_nodes.copy()
    for source in source_nodes:
        reachable_nodes.update(nx.descendants(network, source))

    subnetwork = network.subgraph(reachable_nodes)

    _log(f'Reachability filter: Network solution with {subnetwork.number_of_nodes()} nodes and {subnetwork.number_of_edges()} edges.')
    _log('Reachability filter: finished.')

    return subnetwork


def run_all_paths(network,
                  source_dict,
                  target_dict,
                  depth_cutoff=None,
                  verbose=False):
    """
    Calculate all paths between sources and targets.

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign
            of perturbation.
        target_dict (dict): A dictionary containing the targets and sign
            of measurements.
        depth_cutoff (int, optional): Cutoff for path length. If None,
            there's no cutoff.
        verbose (bool): If True, print warnings when no path is found to
            a given target.

    Returns:
        list: A list containing all paths.
    """
    _log('All paths: Running...')

    all_paths_res = []
    sources = list(source_dict.keys())
    targets = list(target_dict.keys())

    for source in sources:
        try:
            all_paths_res.extend(compute_all_paths(network,
                                                   source,
                                                   targets,
                                                   depth_cutoff))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # _session.log_traceback(console = verbose)
            pass

    subnetwork = utils.get_subnetwork(network, all_paths_res)

    _log(f'All paths: Network solution with {subnetwork.number_of_nodes()} nodes and {subnetwork.number_of_edges()} edges.')
    _log('All paths: finished.')

    return subnetwork, all_paths_res


def compute_all_paths(network, source, targets, cutoff):
    """
    Compute all paths between source and targets.

    Args:
        args (tuple): Tuple containing the network, source, targets and cutoff.

    Returns:
        list: A list containing all paths.
    """
    paths_for_source = []
    for target in targets:
        paths = list(nx.all_simple_paths(network,
                                         source=source,
                                         target=target,
                                         cutoff=cutoff))
        paths_for_source.extend(paths)

    _log(f'All paths: Found {len(paths_for_source)} paths for source {source}')

    return paths_for_source


def add_pagerank_scores(network,
                        source_dict,
                        target_dict,
                        alpha=0.85,
                        max_iter=100,
                        tol=1.0e-6,
                        nstart=None,
                        weight='weight',
                        personalize_for=None):
    """
    Add PageRank scores to the nodes of the network.

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign
            of perturbation.
        target_dict (dict): A dictionary containing the targets and sign
            of measurements.
        percentage (int): Percentage of nodes to keep.
        alpha (float): Damping factor for the PageRank algorithm.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance to determine convergence.
        nstart (dict): Starting value of PageRank iteration for all nodes.
        weight (str): Edge data key to use as weight.
        personalize_for (str): Personalize the PageRank by setting initial
            probabilities for either sources or targets.

    Returns:
        tuple: Contains nodes above threshold from sources, nodes above
            threshold from targets, and overlapping nodes.
    """
    _log('PPR: Adding PageRank scores...')

    sources = source_dict.keys()
    targets = target_dict.keys()

    if nx.is_empty(network):
        _log('PPR: WARNING: Empty network, no scores added.')
        return network

    if personalize_for == "source":
        personalized_prob = {n: 1/len(sources) for n in sources}
        attribute_name = 'pagerank_from_sources'
    elif personalize_for == "target":
        personalized_prob = {n: 1/len(targets) for n in targets}
        network = network.reverse()
        attribute_name = 'pagerank_from_targets'
    else:
        personalized_prob = None
        attribute_name = 'pagerank'

    pagerank = nx.pagerank(network,
                           alpha=alpha,
                           max_iter=max_iter,
                           personalization=personalized_prob,
                           tol=tol, nstart=nstart,
                           weight=weight,
                           dangling=personalized_prob)

    if personalize_for == "target":
        network = network.reverse()

    for node, pr_value in pagerank.items():
        network.nodes[node][attribute_name] = pr_value
    
    _log('PPR: PageRank scores added.')

    return network


def compute_ppr_overlap(network, percentage=20):
    """
    Compute the overlap of nodes that exceed the personalized PageRank
        percentage threshold from sources and targets.

    Args:
        network (nx.Graph): The network.
        percentage (int): Percentage of top nodes to keep.

    Returns:
        tuple: Contains nodes above threshold from sources, nodes above
            threshold from targets, and overlapping nodes.
    """
    _log('PPR: Computing personalized PageRank overlap with percentage', percentage)
    # Sorting nodes by PageRank score from sources and targets
    if nx.is_empty(network):
        _log('PPR: WARNING: Empty network, no PPR overlap computed.')
        return network

    try:
        nodes_sources = [(node, data['pagerank_from_sources']) for node, data in network.nodes(data=True)]
        nodes_targets = [(node, data['pagerank_from_targets']) for node, data in network.nodes(data=True)]

    except KeyError:
        _log('PPR: ERROR: No personalized PageRank scores found.')
        _log('PPR: Please run the add_pagerank_scores method first with personalization options.')
        raise KeyError("Please run the add_pagerank_scores method first with \
                        personalization options.")
    
    sorted_nodes_sources = sorted(nodes_sources,
                                key=lambda x: x[1],
                                reverse=True)
    sorted_nodes_targets = sorted(nodes_targets,
                                key=lambda x: x[1],
                                reverse=True)

    # Calculating the number of nodes to keep
    num_nodes_to_keep_sources = int(
        len(sorted_nodes_sources) * (percentage / 100))
    num_nodes_to_keep_targets = int(
        len(sorted_nodes_targets) * (percentage / 100))

    # Selecting the top nodes
    nodes_above_threshold_from_sources = {
        node[0] for node in sorted_nodes_sources[:num_nodes_to_keep_sources]
    }
    _log(f"PPR: Number of nodes above threshold from sources: {len(nodes_above_threshold_from_sources)}")
    nodes_above_threshold_from_targets = {
        node[0] for node in sorted_nodes_targets[:num_nodes_to_keep_targets]
    }
    _log(f"PPR: Number of nodes above threshold from targets: {len(nodes_above_threshold_from_targets)}")

    nodes_to_include = nodes_above_threshold_from_sources.union(
        nodes_above_threshold_from_targets
        )

    subnetwork = network.subgraph(nodes_to_include)

    _log('PPR: finished.')
    _log(f'PPR: Network solution with {subnetwork.number_of_nodes()} nodes and {subnetwork.number_of_edges()} edges.')

    return subnetwork
