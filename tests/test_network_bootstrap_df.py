import pytest
import pandas as pd

from networkcommons.network._bootstrap import BootstrapDf
from networkcommons.network import _constants as _c


@pytest.fixture
def toy_network_binary_str():

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


@pytest.fixture
def toy_network_binary_tuple():

    edges = pd.DataFrame({
        'source': [('A', 'e'), ('B', 'e'), ('C', 'f')],
        'target': [('B', 'f'), ('C', 'e'), ('A', 'e')],
        'weight': [1, 2, 3],
    })

    nodes = pd.DataFrame({
        'id': ['A', 'B', 'C'],
        'attr1': [10, 20, 30],
    })

    return edges, nodes


def test_bootstrap_binary_str(toy_network_binary_str):

    edges, nodes = toy_network_binary_str
    bootstrap_df = BootstrapDf(edges = edges, nodes = nodes, node_key = 'id')

    assert len(bootstrap_df._edges) == 3
    assert len(bootstrap_df._nodes) == 3
    assert len(bootstrap_df._node_attrs) == 3
    assert 'attr1' in bootstrap_df._node_attrs.columns
    assert set(bootstrap_df._nodes.keys()) == {'A', 'B', 'C'}


def test_bootstrap_binary_tuple(toy_network_binary_tuple):

    edges, nodes = toy_network_binary_tuple
    bootstrap_df = BootstrapDf(
        edges = edges,
        node_key = ('id', 'comp'),
    )

    assert len(bootstrap_df._edges) == 3
    assert len(bootstrap_df._nodes) == 5
    assert len(bootstrap_df._node_attrs) == 5
    assert 'comp' in bootstrap_df._node_attrs.columns
    assert _c.NODE_KEY in bootstrap_df._node_attrs.columns
    assert set(bootstrap_df._nodes.keys()) == {
        ('A', 'e'),
        ('B', 'e'),
        ('B', 'f'),
        ('C', 'e'),
        ('C', 'f'),
    }
    assert bootstrap_df._edges[0][0] == {('A', 'e')}
    assert bootstrap_df._nodes[('A', 'e')] == ({0}, {2})


def test_proc_node_key(toy_network_binary_str):

    edges, nodes = toy_network_binary_str
    bootstrap_df = BootstrapDf(edges = edges, nodes = nodes, node_key = 'id')
    node_key = bootstrap_df._proc_node_key('A')

    assert node_key == 'A'

    edges, nodes = toy_network_binary_str
    custom_edges = pd.DataFrame({
        'source': ['A_x;B_y', 'B_x;C_y'],
        'target': ['C_z;A_u', 'A_z;B_y'],
    })

    try:

        bootstrap_df = BootstrapDf(
            edges = custom_edges,
            nodes = nodes,
            node_key = ('id', 'species'),
            node_key_sep = '_',
        )

    except ValueError as e:

        assert str(e).startswith('Columns in `node_key` not found')

    bootstrap_df = BootstrapDf(
        edges = custom_edges,
        node_key = ('id', 'species'),
        node_key_sep = '_',
    )
    node_key = bootstrap_df._proc_node_key('A_x')

    assert node_key == ('A', 'x')
    assert bootstrap_df._edges[0][0] == {('A', 'x'), ('B', 'y')}


def test_edge_processing_with_custom_separator(toy_network_binary_str):

    edges, nodes = toy_network_binary_str
    custom_edges = pd.DataFrame({
        'source': ['A,B', 'B,C'],
        'target': ['C,A', 'A,B'],
    })
    bootstrap_df = BootstrapDf(
        edges = custom_edges,
        nodes = nodes,
        inner_sep = ',',
        node_key = 'id',
    )

    assert len(bootstrap_df._edges) == 2
    assert 'A' in bootstrap_df._nodes
    assert 'B' in bootstrap_df._nodes
    assert 'C' in bootstrap_df._nodes
