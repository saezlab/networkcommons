import pandas as pd
import networkx as nx
import numpy as np
import corneto as cn
from unittest.mock import patch
import pytest
import networkcommons._utils as utils
import pygraphviz as pgv


def test_to_cornetograph():
    nx_graph = nx.DiGraph()
    nx_graph.add_edge('a', 'b', sign=1)

    corneto_graph = utils.to_cornetograph(nx_graph)

    assert isinstance(corneto_graph, cn._graph.Graph)

    for data in corneto_graph.get_attr_edges():
        assert 'interaction' in data.keys()
        assert 'sign' not in data.keys()

    corneto_graph = cn.Graph.from_sif_tuples([('node1', 1, 'node2')])
    result = utils.to_cornetograph(corneto_graph)
    assert isinstance(result, cn._graph.Graph)


def test_to_cornetograph_when_cornetograph():
    corneto_graph = cn.Graph.from_sif_tuples([('node1', 1, 'node2')])

    result = utils.to_cornetograph(corneto_graph)
    assert isinstance(result, cn._graph.Graph)


def test_to_cornetograph_when_not_supported():
    multi_graph = nx.MultiDiGraph()
    with pytest.raises(NotImplementedError, match="Only nx.DiGraph graphs and corneto graphs are supported."):
        utils.to_cornetograph(multi_graph)

    undir_graph = nx.Graph()
    with pytest.raises(NotImplementedError, match="Only nx.DiGraph graphs and corneto graphs are supported."):
        utils.to_cornetograph(undir_graph)

    graphviz_grpah = pgv.AGraph()
    with pytest.raises(NotImplementedError, match="Only nx.DiGraph graphs and corneto graphs are supported."):
        utils.to_cornetograph(graphviz_grpah)


def test_to_networkx():
    corneto_graph = cn.Graph.from_sif_tuples([('node1', 1, 'node2')])

    # Convert to networkx graph using the function
    networkx_graph = utils.to_networkx(corneto_graph)

    # Expected networkx graph
    expected_graph = nx.DiGraph()
    expected_graph.add_node('node1', attr1='value1')
    expected_graph.add_node('node2', attr1='value2')
    expected_graph.add_edge('node1', 'node2', sign=1)

    assert isinstance(networkx_graph, nx.DiGraph)

    assert nx.is_isomorphic(networkx_graph, expected_graph)
    for u, v, data in networkx_graph.edges(data=True):
        assert data['sign'] == expected_graph.get_edge_data(u, v)['sign']
        assert 'interaction' not in data.keys()

    nx_graph = nx.DiGraph()
    nx_graph.add_edge('a', 'b', sign=1)

    # Convert to networkx graph using the function
    networkx_graph = utils.to_networkx(nx_graph)

    # Expected networkx graph
    expected_graph = nx.DiGraph()
    expected_graph.add_edge('a', 'b', sign=1)

    assert nx.is_isomorphic(networkx_graph, expected_graph)
    for u, v, data in networkx_graph.edges(data=True):
        assert data['sign'] == expected_graph.get_edge_data(u, v)['sign']


def test_to_networkx_when_networkx_graph():
    nx_graph = nx.DiGraph()
    nx_graph.add_edge('a', 'b', sign=1)

    result = utils.to_networkx(nx_graph)
    assert isinstance(result, nx.DiGraph)
    assert nx.is_isomorphic(result, nx_graph)
    for u, v, data in nx_graph.edges(data=True):
        assert data['sign'] == result.get_edge_data(u, v)['sign']
        assert 'interaction' not in data.keys()


def test_to_networkx_when_not_supported():
    multi_graph = nx.MultiDiGraph()
    with pytest.raises(NotImplementedError, match="Only nx.DiGraph graphs and corneto graphs are supported."):
        utils.to_networkx(multi_graph)

    undir_graph = nx.Graph()
    with pytest.raises(NotImplementedError, match="Only nx.DiGraph graphs and corneto graphs are supported."):
        utils.to_networkx(undir_graph)

    graphviz_grpah = pgv.AGraph()
    with pytest.raises(NotImplementedError, match="Only nx.DiGraph graphs and corneto graphs are supported."):
        utils.to_networkx(graphviz_grpah)


def test_read_network_from_file():
    with patch('pandas.read_csv') as mock_read_csv, patch('networkcommons._utils.network_from_df') as mock_network_from_df:
        mock_read_csv.return_value = pd.DataFrame({'source': ['a'], 'target': ['b']})
        utils.read_network_from_file('dummy_path')
        mock_network_from_df.assert_called_once()


def test_network_from_df():
    df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})
    result = utils.network_from_df(df)
    assert isinstance(result, nx.DiGraph)
    assert list(result.edges(data=True)) == [('a', 'b', {'sign': 1})]


def test_network_from_df_no_attrs():
    df = pd.DataFrame({'source': ['a'], 'target': ['b']})
    result = utils.network_from_df(df)
    assert isinstance(result, nx.DiGraph)
    assert list(result.edges(data=True)) == [('a', 'b', {})]


def test_network_from_df_negative_weights():
    df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'weight':-3})
    result = utils.network_from_df(df)
    assert isinstance(result, nx.DiGraph)
    assert list(result.edges(data=True)) == [('a', 'b', {'weight': 3, 'sign': -1})]


def test_get_subnetwork():
    G = nx.path_graph(4)
    paths = [[0, 1, 2], [2, 3]]
    subnetwork = utils.get_subnetwork(G, paths)
    assert list(subnetwork.edges) == [(0, 1), (1, 2), (2, 3)]


def test_decoupler_formatter():
    df = pd.DataFrame({'ID': ['Gene1', 'Gene2', 'Gene3'], 'stat': [3.5, 4, 3]}).set_index('ID')
    result = utils.decoupler_formatter(df, ['stat'])
    expected = df.T
    pd.testing.assert_frame_equal(result, expected)


def test_decoupler_formatter_string():
    df = pd.DataFrame({'ID': ['Gene1', 'Gene2', 'Gene3'], 'stat': [3.5, 4, 3]}).set_index('ID')
    result = utils.decoupler_formatter(df, 'stat')
    expected = df.T
    pd.testing.assert_frame_equal(result, expected)


def test_targetlayer_formatter():
    df = pd.DataFrame({'TF': ['A', 'B', 'C', 'D'], 'sign': [1.5, -2, 0, 3]}).set_index('TF')
    result = utils.targetlayer_formatter(df, n_elements=2)
    expected = {'D': 1, 'B': -1}
    assert result == expected


def test_subset_df_with_nodes():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    df = pd.DataFrame({'value': [10, 20, 30]}, index=[1, 2, 4])
    result = utils.subset_df_with_nodes(G, df)
    expected = pd.DataFrame({'value': [10, 20]}, index=[1, 2])
    pd.testing.assert_frame_equal(result, expected)


def test_handle_missing_values_fill():
    df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [3, 2, np.nan], 'C': [np.nan, 7, 8]})
    result = utils.handle_missing_values(df, 0.5, fill=True)
    expected = pd.DataFrame({'index': [0, 1], 'A': [1.0, 2.0], 'B': [3.0, 2.0], 'C': [2.0, 7.0]}).astype({'index': 'int64'})
    pd.testing.assert_frame_equal(result, expected)


def test_handle_missing_values_fill_and_drop():
    df = pd.DataFrame({'A': [1, np.nan, np.nan], 'B': [np.nan, 2, np.nan], 'C': [np.nan, 7, np.nan]})
    result = utils.handle_missing_values(df, 0.5, fill=True)
    expected = pd.DataFrame({'index': [1], 'A': [4.5], 'B': [2.0], 'C': [7.0]}).astype({'index': 'int64'})
    pd.testing.assert_frame_equal(result, expected)


def test_handle_missing_values_drop():
    df = pd.DataFrame({'A': [1, np.nan, np.nan], 'B': [np.nan, np.nan, np.nan], 'C': [np.nan, np.nan, 8]})
    result = utils.handle_missing_values(df, 0.1, fill=False)
    expected = pd.DataFrame({'index': [], 'A': [], 'B': [], 'C': []}).astype({'index': 'int64'})
    pd.testing.assert_frame_equal(result, expected)


def test_handle_missing_values_non_numeric_column():
    df = pd.DataFrame({'id': ['a', 'b', 'c'], 'A': [1, 2, np.nan], 'B': [3, 2, np.nan], 'C': [np.nan, 7, 8]})
    result = utils.handle_missing_values(df, 0.5)
    expected = pd.DataFrame({'id': ['a', 'b'], 'A': [1.0, 2.0], 'B': [3.0, 2.0], 'C': [2.0, 7.0]})
    pd.testing.assert_frame_equal(result, expected)


def test_handle_missing_values_more_than_one_non_numeric_column():
    df = pd.DataFrame({'id1': ['a', 'b', 'c'], 'id2': ['x', 'y', 'z'], 'A': [1, 2, np.nan], 'B': [3, 2, np.nan]})
    try:
        utils.handle_missing_values(df, 0.5)
    except ValueError as e:
        assert str(e) == "More than one non-numeric column found: Index(['id1', 'id2'], dtype='object')"


def test_handle_missing_values_no_missing_values():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = utils.handle_missing_values(df, 0.5)
    expected = df.reset_index().rename(columns={'index': 'index'})
    expected = expected.astype({'index': 'int64'})
    pd.testing.assert_frame_equal(result, expected)


def test_handle_missing_values_with_inf():
    df = pd.DataFrame({'A': [1, 2, -np.inf], 'B': [3, 2, np.nan], 'C': [np.nan, 7, 8]})
    result = utils.handle_missing_values(df, 0.5)
    expected = pd.DataFrame({'index': [0, 1], 'A': [1.0, 2.0], 'B': [3.0, 2.0], 'C': [2.0, 7.0]}).astype({'index': 'int64'})
    pd.testing.assert_frame_equal(result, expected)