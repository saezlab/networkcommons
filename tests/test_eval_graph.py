import pytest

import networkx as nx
import pandas as pd

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
