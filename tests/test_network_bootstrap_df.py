import pytest
import pandas as pd

from networkcommons.network._bootstrap import BootstrapDf

@pytest.fixture
def toy_network_df():
    edges = pd.DataFrame({
        'source': ['A', 'B', 'C'],
        'target': ['B', 'C', 'A'],
        'weight': [1, 2, 3],
    })

    nodes = pd.DataFrame({
        'id': ['A', 'B', 'C'],
        'attr1': [10, 20, 30],
    })

    return edges, nodes


def test_bootstrap_edges(toy_network_df):
    edges, nodes = toy_network_df
    bootstrap_df = BootstrapDf(edges=edges, nodes=nodes, node_key="id")
    assert len(bootstrap_df._edges) == 3
    assert len(bootstrap_df._nodes) == 3


def test_bootstrap_nodes(toy_network_df):
    edges, nodes = toy_network_df
    bootstrap_df = BootstrapDf(edges=edges, nodes=nodes, node_key="id")
    assert len(bootstrap_df._node_attrs) == 3
    assert len(bootstrap_df._nodes) == 3
    assert 'attr1' in bootstrap_df._node_attrs.columns


def test_proc_nodes_in_edge(toy_network_df):
    edges, nodes = toy_network_df
    bootstrap_df = BootstrapDf(edges=edges, nodes=nodes, node_key="id")
    nodes_processed = bootstrap_df._proc_nodes_in_edge('A;B;C')
    assert nodes_processed == {'A', 'B', 'C'}


def test_proc_node_key(toy_network_df):
    edges, nodes = toy_network_df
    bootstrap_df = BootstrapDf(edges=edges, nodes=nodes, node_key='id')
    node_key = bootstrap_df._proc_node_key('A')
    assert node_key == ('A',)


def test_edge_processing_with_missing_columns(toy_network_df):
    edges, nodes = toy_network_df
    minimal_edges = pd.DataFrame({
        'source': ['A', 'B'],
        'target': ['C', 'A'],
    })
    bootstrap_df = BootstrapDf(edges=minimal_edges, nodes=nodes, ignore=['weight'])
    assert bootstrap_df._edge_attrs.empty


def test_edge_processing_with_custom_separator(toy_network_df):
    edges, nodes = toy_network_df
    custom_edges = pd.DataFrame({
        'source': ['A,B', 'B,C'],
        'target': ['C,A', 'A,B'],
    })
    bootstrap_df = BootstrapDf(edges=custom_edges, nodes=nodes, inner_sep=',')
    assert len(bootstrap_df._edges) == 2
    assert 'A' in bootstrap_df._nodes
    assert 'B' in bootstrap_df._nodes
    assert 'C' in bootstrap_df._nodes
