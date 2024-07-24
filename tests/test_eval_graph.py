import pytest

import networkx as nx
import pandas as pd
import numpy as np

from networkcommons.eval import _metrics

from unittest.mock import patch
import networkcommons._utils as utils

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
        'network': 'Network1',
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


def test_run_ora():
    # Create an example graph
    graph = nx.DiGraph()
    graph.add_nodes_from(["geneA", "geneB", "geneC", "geneD", "geneE", "geneF"])

    # Create an example net DataFrame
    net = pd.DataFrame({
        'source': ['gene_set_1', 'gene_set_1', 'gene_set_2', 'gene_set_2', 'gene_set_2'],
        'target': ['geneA', 'geneB', 'geneC', 'geneD', 'geneE']
    })

    # Expected output DataFrame (you need to adjust this based on expected results)
    expected_results = pd.DataFrame({
        'ora_Term': ["gene_set_1", "gene_set_2"],
        'ora_Set size': [2, 3],
        'ora_Overlap ratio': [1.0, 1.0],
        'ora_p-value': [7.500375e-08, 1.500225e-11],
        'ora_FDR p-value': [7.500375e-08, 3.000450e-11],
        'ora_Odds ratio': [4444.111111, 5713.571429],
        'ora_Combined score': [72908.876856, 142398.317463],
        'ora_Features': ["geneA;geneB", "geneC;geneD;geneE"],
        'ora_rank': [2.0, 1.0]
    })

    # Run the ORA function
    ora_results = _metrics.run_ora(graph, net, metric='ora_Combined score', ascending=False)

    # Assertions to verify the results
    pd.testing.assert_frame_equal(ora_results, expected_results)


def test_get_phosphorylation_status():
    # Create a sample network graph
    network = nx.DiGraph()
    network.add_nodes_from(['node1', 'node2', 'node3'])

    # Create a sample dataframe
    data = {
        'stat': [0.5, 1.5, -0.5, 0.0],
    }
    dataframe = pd.DataFrame(data, index=['node1', 'node2', 'node3', 'node4'])

    result_df = _metrics.get_phosphorylation_status(network, dataframe, col='stat')

    expected_data = {
        'avg_relabundance': [np.mean([0.5, 1.5, -0.5])],
        'avg_relabundance_overall': [np.mean([0.5, 1.5, -0.5, 0.0])],
        'diff_avg_abundance': [abs(np.mean([0.5, 1.5, -0.5])) - abs(np.mean([0.0]))],
        'nodes_with_phosphoinfo': [3],
        'coverage': [3 / 3 * 100]
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)
