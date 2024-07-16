import pytest

import networkx as nx
import pandas as pd
import numpy as np

from networkcommons.eval import _metrics


@pytest.fixture
def network():

    network = nx.DiGraph()
    network.add_edge('A', 'B', weight=1)
    network.add_edge('B', 'C', weight=2)
    network.add_edge('C', 'D', weight=3)
    network.add_edge('C', 'E', weight=3)
    network.add_edge('A', 'D', weight=6)
    network.add_edge('A', 'E', weight=4)
    network.add_edge('E', 'F', weight=5)

    return network


def test_get_number_nodes():

    assert _metrics.get_number_nodes(nx.Graph()) == 0
    assert _metrics.get_number_nodes(nx.Graph([(1, 2)])) == 2
    assert _metrics.get_number_nodes(nx.Graph([(1, 2), (2, 3)])) == 3


def test_get_number_edges():

    assert _metrics.get_number_edges(nx.Graph()) == 0
    assert _metrics.get_number_edges(nx.Graph([(1, 2)])) == 1
    assert _metrics.get_number_edges(nx.Graph([(1, 2), (2, 3)])) == 2


def test_get_mean_degree(network):

    assert _metrics.get_mean_degree(network) == 7/3


def test_get_mean_betweenness(network):

    assert _metrics.get_mean_betweenness(network) == 0.05833333333333334


def test_get_mean_closeness(network):

    assert _metrics.get_mean_closeness(network) == 0.29444444444444445


def test_get_connected_targets(network):

    target_dict = {'D': 1, 'F': 1, 'W': 1}

    assert _metrics.get_connected_targets(network, target_dict) == 2
    assert (
        _metrics.get_connected_targets(network, target_dict, percent=True) ==
        2 / 3 * 100
    )


def test_get_recovered_offtargets(network):

    offtargets = ['B', 'D', 'W']

    assert _metrics.get_recovered_offtargets(network, offtargets) == 2
    assert (
        _metrics.get_recovered_offtargets(network, offtargets, percent=True) ==
        2 / 3 * 100
    )# noqa: E501


def test_get_graph_metrics(network):

    target_dict = {'D': 1, 'F': 1, 'W': 1}

    metrics = pd.DataFrame({
        'Number of nodes': 6,
        'Number of edges': 7,
        'Mean degree': 7/3,
        'Mean betweenness': 0.05833333333333334,
        'Mean closeness': 0.29444444444444445,
        'Connected targets': 2
    }, index=[0])

    assert _metrics.get_graph_metrics(network, target_dict).equals(metrics)


def test_all_nodes_in_ec50_dict():
    network = nx.Graph([(1, 2), (2, 3)])
    ec50_dict = {1: 5.0, 2: 10.0, 3: 15.0}
    expected_result = pd.DataFrame({
        'avg_EC50_in': [10.0],
        'avg_EC50_out': [np.nan],
        'nodes_with_EC50': [3],
        'coverage': [100.0]
    })

    result = _metrics.get_ec50_evaluation(network, ec50_dict)
    pd.testing.assert_frame_equal(result, expected_result)


def test_some_nodes_in_ec50_dict():
    network = nx.Graph([(1, 2), (2, 3)])
    ec50_dict = {1: 5.0, 2: 10.0, 4: 20.0}
    expected_result = pd.DataFrame({
        'avg_EC50_in': [7.5],
        'avg_EC50_out': [20.0],
        'nodes_with_EC50': [2],
        'coverage': [2/3 * 100]
    })

    result = _metrics.get_ec50_evaluation(network, ec50_dict)
    pd.testing.assert_frame_equal(result, expected_result)


def test_no_nodes_in_ec50_dict():
    network = nx.Graph([(1, 2), (2, 3)])
    ec50_dict = {4: 20.0, 5: 25.0}
    expected_result = pd.DataFrame({
        'avg_EC50_in': [np.nan],
        'avg_EC50_out': [22.5],
        'nodes_with_EC50': [0],
        'coverage': [0.0]
    })

    result = _metrics.get_ec50_evaluation(network, ec50_dict)
    pd.testing.assert_frame_equal(result, expected_result)


def test_empty_network():
    network = nx.Graph()
    ec50_dict = {1: 5.0, 2: 10.0}
    expected_result = pd.DataFrame({
        'avg_EC50_in': [np.nan],
        'avg_EC50_out': [7.5],
        'nodes_with_EC50': [0],
        'coverage': [np.nan]
    })

    result = _metrics.get_ec50_evaluation(network, ec50_dict)
    pd.testing.assert_frame_equal(result, expected_result)

