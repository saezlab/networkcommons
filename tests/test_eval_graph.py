import pytest
import networkx as nx
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import random

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
    assert _metrics.get_mean_degree(network) == 7 / 3


def test_get_mean_betweenness(network):
    assert _metrics.get_mean_betweenness(network) == 0.05833333333333334


def test_get_mean_closeness(network):
    assert _metrics.get_mean_closeness(network) == 0.29444444444444445


def test_get_connected_targets(network):
    target_dict = {'D': 1, 'F': 1, 'W': 1}
    assert _metrics.get_connected_targets(network, target_dict) == 2
    assert _metrics.get_connected_targets(network, target_dict, percent=True) == 2 / 3 * 100


def test_get_recovered_offtargets(network):
    offtargets = ['B', 'D', 'W']
    result = _metrics.get_recovered_offtargets(network, offtargets)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2)
    assert result['n_offtargets'][0] == 2
    assert result['perc_offtargets'][0] == 2 / 3 * 100



@patch('networkcommons.data.omics.panacea_experiments')
@patch('networkcommons.data.omics.panacea_gold_standard')
@patch('networkcommons.data.network.get_omnipath')
@patch('networkcommons.utils.network_from_df')
@patch('networkcommons.data.omics.panacea_tables')
@patch('networkcommons.methods.run_shortest_paths')
@patch('networkcommons.methods.run_sign_consistency')
@patch('networkcommons.methods.run_all_paths')
@patch('networkcommons.methods.add_pagerank_scores')
@patch('networkcommons.methods.run_corneto_carnival')
@patch('networkcommons.eval._metrics.get_metric_from_networks')
@patch('networkcommons.eval._metrics._log')
def test_get_offtarget_panacea_evaluation(
    mock_log,
    mock_get_metric_from_networks,
    mock_run_corneto_carnival,
    mock_add_pagerank_scores,
    mock_run_all_paths,
    mock_run_sign_consistency,
    mock_run_shortest_paths,
    mock_panacea_tables,
    mock_network_from_df,
    mock_get_omnipath,
    mock_panacea_gold_standard,
    mock_panacea_experiments
):
    # Mocks
    mock_panacea_experiments.return_value = pd.DataFrame({
        'group': ['CellA_DrugA', 'CellA_DrugB'],
        'tf_scores': [True, True]
    })

    mock_panacea_gold_standard.return_value = pd.DataFrame({
        'cmpd': ['DrugA', 'DrugA', 'DrugA'],
        'target': ['TargetA', 'TargetB', 'TargetC'],
        'rank': [1, 2, 3]
    })

    mock_network_df = pd.DataFrame({
        'source': ['TargetA', 'Node1', 'TargetC', 'TargetD'],
        'target': ['Node1', 'TargetC', 'TargetC', 'Node2'],
        'interaction': [1, 1, 1, 1]
    })
    mock_network_df = mock_network_df.astype({'source': str, 'target': str})
    mock_graph = nx.from_pandas_edgelist(mock_network_df, source='source', target='target', create_using=nx.DiGraph)
    mock_network_from_df.return_value = mock_graph

    mock_panacea_tables.side_effect = lambda **kwargs: pd.DataFrame({
        'items': ['Node1', 'Node2'],
        'act': [0.5, -0.3]
    })

    mock_run_shortest_paths.return_value = (MagicMock(), [])
    mock_run_sign_consistency.return_value = (MagicMock(), [])
    mock_run_all_paths.return_value = (MagicMock(), [])
    mock_add_pagerank_scores.return_value = MagicMock()
    mock_run_corneto_carnival.return_value = MagicMock()

    mock_get_metric_from_networks.return_value = pd.DataFrame({
        'perc_offtargets': [10, 25],
        'network': ['shortest_path', 'all_paths']
    })

    # Test
    _metrics.get_offtarget_panacea_evaluation(cell='CellY') # No info
    mock_log.assert_called_with("EVAL: no cell-drug combinations found with TF activity scores. Exiting...")

    result_df = _metrics.get_offtarget_panacea_evaluation(cell='CellA') # Only cell

    # Assertins
    assert isinstance(result_df, pd.DataFrame)
    assert 'perc_offtargets' in result_df.columns
    assert len(result_df) > 0

    result_df = _metrics.get_offtarget_panacea_evaluation(cell='CellA', drug='DrugA') # both

    assert isinstance(result_df, pd.DataFrame)
    assert 'perc_offtargets' in result_df.columns
    assert len(result_df) > 0

    result_df = _metrics.get_offtarget_panacea_evaluation() # None

    assert isinstance(result_df, pd.DataFrame)
    assert 'perc_offtargets' in result_df.columns
    assert len(result_df) > 0

    mock_panacea_experiments.assert_called()
    mock_panacea_gold_standard.assert_called()
    mock_get_omnipath.assert_called()
    mock_network_from_df.assert_called()
    mock_panacea_tables.assert_called()
    mock_run_shortest_paths.assert_called()
    mock_run_sign_consistency.assert_called()
    mock_run_all_paths.assert_called()
    mock_add_pagerank_scores.assert_called()
    mock_run_corneto_carnival.assert_called()
    mock_get_metric_from_networks.assert_called()
    mock_log.assert_called_with("EVAL: finished offtarget recovery evaluation using PANACEA TF activity scores.")


def test_all_nodes_in_ec50_dict():
    network = nx.Graph([(1, 2), (2, 3)])
    ec50_dict = {1: 5.0, 2: 10.0, 3: 15.0}
    expected_result = pd.DataFrame({
        'avg_EC50_in': [10.0],
        'avg_EC50_out': [np.nan],
        'diff_EC50': [np.nan],
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
        'diff_EC50': [-12.5],
        'nodes_with_EC50': [2],
        'coverage': [2 / 3 * 100]
    })
    result = _metrics.get_ec50_evaluation(network, ec50_dict)
    pd.testing.assert_frame_equal(result, expected_result)


def test_no_nodes_in_ec50_dict():
    network = nx.Graph([(1, 2), (2, 3)])
    ec50_dict = {4: 20.0, 5: 25.0}
    expected_result = pd.DataFrame({
        'avg_EC50_in': [np.nan],
        'avg_EC50_out': [22.5],
        'diff_EC50': [np.nan],
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
        'diff_EC50': [np.nan],
        'nodes_with_EC50': [0],
        'coverage': [np.nan]
    })
    result = _metrics.get_ec50_evaluation(network, ec50_dict)
    pd.testing.assert_frame_equal(result, expected_result)


def test_run_ora():
    graph = nx.DiGraph()
    graph.add_nodes_from(["geneA", "geneB", "geneC", "geneD", "geneE", "geneF"])

    net = pd.DataFrame({
        'source': ['gene_set_1', 'gene_set_1', 'gene_set_2', 'gene_set_2', 'gene_set_2'],
        'target': ['geneA', 'geneB', 'geneC', 'geneD', 'geneE']
    })

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

    ora_results = _metrics.run_ora(graph, net, metric='ora_Combined score', ascending=False)

    pd.testing.assert_frame_equal(ora_results, expected_results)


def test_get_phosphorylation_status():
    network = nx.DiGraph()
    network.add_nodes_from(['node1', 'node2', 'node3'])

    data = {
        'stat': [0.5, 1.5, -0.5, 0.0],
    }
    dataframe = pd.DataFrame(data, index=['node1', 'node2', 'node3', 'node4'])

    subset_df = dataframe.loc[~dataframe.index.isin(['node4'])]
    metric_in = abs(subset_df['stat'].values)
    metric_overall = abs(dataframe['stat'].values)

    result_df = _metrics.get_phosphorylation_status(network, dataframe, col='stat')

    expected_data = {
        'avg_relabundance': np.mean(metric_in),
        'avg_relabundance_overall': np.mean(metric_overall),
        'diff_dysregulation': np.mean(metric_in) - np.mean(metric_overall),
        'nodes_with_phosphoinfo': [3],
        'coverage': [3 / 3 * 100]
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


def test_get_metric_from_networks():
    real_graph = nx.DiGraph()
    real_graph.add_edges_from([
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('D', 'E')
    ])

    random_graph = nx.DiGraph()
    random_graph.add_edges_from([
        ('W', 'X'),
        ('X', 'Y'),
        ('Y', 'Z'),
        ('Z', 'W'),
        ('W', 'Y')
    ])

    networks = {
        'shortest_path__real': real_graph,
        'shortest_path__random_1': random_graph
    }

    target_dict = {'D': 1, 'F': 1, 'W': 1}

    expected_data = {
        'Number of nodes': [5, 4],
        'Number of edges': [4, 5],
        'Mean degree': [1.6, 2.5],
        'Mean betweenness': [0.166667, 0.375000],
        'Mean closeness': [0.271667, 0.587500],
        'Connected targets': [1, 1],
        'network': ['shortest_path__real', 'shortest_path__random_1'],
        'type': ['real', 'random'],
        'method': ['shortest_path', 'shortest_path']
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = _metrics.get_metric_from_networks(
        networks,
        _metrics.get_graph_metrics,
        target_dict=target_dict
    )
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


def test_function_not_found():
    networks = {
        'real_network__1': nx.path_graph(5),
    }
    with pytest.raises(NameError):
        _metrics.get_metric_from_networks(networks, nonexistent_function)


@patch('random.shuffle')
def test_perform_random_controls(mock_shuffle, network):
    mock_shuffle.side_effect = lambda x: x.reverse()
    inference_function = lambda g, **kw: (g, None)
    n_iterations = 2
    network_name = 'test_network'
    item_list = ['A', 'B', 'C', 'D', 'E', 'F']
    target_dict = {'A': 1, 'B': 1, 'C': 1}

    results = _metrics.perform_random_controls(
        network,
        inference_function,
        n_iterations,
        network_name,
        randomise_measurements=True,
        item_list=item_list,
        target_dict=target_dict
    )

    assert len(results) == n_iterations
    for i in range(n_iterations):
        assert f"{network_name}__random{i+1:03d}" in results


def test_get_graph_metrics():
    target_dict = {'D': 1, 'F': 1, 'W': 1}
    network1 = nx.DiGraph()
    network1.add_edges_from([
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('D', 'E')
    ])

    network2 = nx.DiGraph()
    network2.add_edges_from([
        ('W', 'X'),
        ('X', 'Y'),
        ('Y', 'Z'),
        ('Z', 'W'),
        ('W', 'Y')
    ])
    networks = {'network1': network1, 'network2': network2}

    expected_metrics = pd.DataFrame({
        'Number of nodes': [5, 4],
        'Number of edges': [4, 5],
        'Mean degree': [1.6, 2.5],
        'Mean betweenness': [0.166667, 0.375000],
        'Mean closeness': [0.271667, 0.587500],
        'Connected targets': [1, 1],
        'network': ['network1', 'network2']
    })

    metrics = _metrics.get_graph_metrics(networks, target_dict)
    pd.testing.assert_frame_equal(metrics.reset_index(drop=True), expected_metrics)
    assert 'network' in metrics.columns

    expected_metrics = pd.DataFrame({
        'Number of nodes': [5],
        'Number of edges': [4],
        'Mean degree': [1.6],
        'Mean betweenness': [0.166667],
        'Mean closeness': [0.271667],
        'Connected targets': [1]
    })
    print(type(network1))
    metrics = _metrics.get_graph_metrics(network1, target_dict)
    pd.testing.assert_frame_equal(metrics.reset_index(drop=True), expected_metrics)
    assert 'network' not in metrics.columns


def test_get_graph_metrics_invalid_type():
    with pytest.raises(TypeError, match="The network must be a networkx graph or a dictionary of networkx graphs."):
        _metrics.get_graph_metrics(123, {})


def test_shuffle_dict_keys():
    original_dict = {'A': 1, 'B': 2, 'C': 3}
    items = ['A', 'B', 'C', 'X', 'Y', 'Z']

    random.seed(42)
    shuffled_dict = _metrics.shuffle_dict_keys(original_dict, items)

    expected_dict = {'A': 2, 'Y': 3, 'Z': 1}
    assert shuffled_dict == expected_dict


def test_get_metric_from_networks_non_callable():
    networks = {
        'real_network__1': nx.path_graph(5),
    }
    non_callable = "I am not callable"
    with pytest.raises(NameError):
        _metrics.get_metric_from_networks(networks, non_callable)


def test_perform_random_controls_with_item_list(network):
    inference_function = lambda g, **kw: (g, None)
    n_iterations = 2
    network_name = 'test_network'
    item_list = ['A', 'B', 'C', 'D', 'E', 'F']
    target_dict = {'A': 1, 'B': 1, 'C': 1}

    results = _metrics.perform_random_controls(
        network,
        inference_function,
        n_iterations,
        network_name,
        randomise_measurements=True,
        item_list=item_list,
        target_dict=target_dict
    )

    assert len(results) == n_iterations
    for i in range(n_iterations):
        assert f"{network_name}__random{i+1:03d}" in results

@patch("networkcommons.eval._metrics.shuffle_dict_keys")
def test_perform_random_controls_without_item_list(mock_shuffle, network):
    inference_function = lambda g, **kw: (g, None)
    n_iterations = 2
    network_name = 'test_network'
    target_dict = {'A': 1, 'B': 1, 'C': 1}

    results = _metrics.perform_random_controls(
        network,
        inference_function,
        n_iterations,
        network_name,
        randomise_measurements=False,
        target_dict=target_dict
    )

    assert len(results) == n_iterations
    for i in range(n_iterations):
        assert f"{network_name}__random{i+1:03d}" in results
    mock_shuffle.assert_not_called()
    
    
    


