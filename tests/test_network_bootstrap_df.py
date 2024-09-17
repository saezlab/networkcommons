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
        'id': [('A', 'e'), ('B', 'e'), ('B', 'f'), ('C', 'f'), ('C', 'f')],
        'attr1': [10, 20, 30, 20, 30],
    })

    return edges, nodes


def test_bootstrap_binary_str(toy_network_binary_str):

    edges, nodes = toy_network_binary_str
    bs = BootstrapDf(edges = edges, nodes = nodes, node_key = 'id')

    assert len(bs._edges) == 3
    assert len(bs._nodes) == 3
    assert len(bs._node_attrs) == 3
    assert 'attr1' in bs._node_attrs.columns
    assert set(bs._nodes.keys()) == {'A', 'B', 'C'}


def test_bootstrap_binary_tuple(toy_network_binary_tuple):

    edges, nodes = toy_network_binary_tuple
    bs = BootstrapDf(
        edges = edges,
        node_key = ('id', 'comp'),
    )

    assert len(bs._edges) == 3
    assert len(bs._nodes) == 5
    assert len(bs._node_attrs) == 5
    assert 'comp' in bs._node_attrs.columns
    assert _c.NODE_KEY in bs._node_attrs.columns
    assert set(bs._nodes.keys()) == {
        ('A', 'e'),
        ('B', 'e'),
        ('B', 'f'),
        ('C', 'e'),
        ('C', 'f'),
    }
    assert bs._edges[0][0] == {('A', 'e')}
    assert bs._nodes[('A', 'e')] == ({0}, {2})


def test_bootstrap_hyper_set_str(toy_network_binary_str):

    edges, nodes = toy_network_binary_str
    edges = pd.DataFrame(
        {
            'source': [{'A', 'B'}, {'A', 'B'}, {'B', 'C'}],
            'target': [{'B', 'D'}, {'C', 'D'}, {'A', 'B'}],
            'weight': [1, 2, 3],
            'style': ['solid', 'dashed', 'dotted'],
        },
    )
    bs = BootstrapDf(edges = edges, nodes = nodes, node_key = 'id')

    assert len(bs._edges) == 3
    assert len(bs._nodes) == 4
    assert len(bs._node_attrs) == 4
    assert set(bs._nodes.keys()) == {'A', 'B', 'C', 'D'}
    assert bs._nodes['D'] == (set(), {0, 1})


def test_bootstrap_hyper_set_tuple():

    edges = pd.DataFrame(
        {
            'source': [
                {('A', 'e'), ('B', 'f')},
                {('A', 'e'), ('B', 'e')},
                {('B', 'e'), ('C', 'e')},
            ],
            'target': [
                {('D', 'e'), ('B', 'e')},
                {('C', 'e'), ('B', 'f')},
                {('D', 'e'), ('C', 'f')},
            ],
            'weight': [1, 2, 3],
            'style': ['solid', 'dashed', 'dotted'],
        },
    )
    bs = BootstrapDf(edges = edges, node_key = ('name', 'class'))

    assert len(bs._edges) == 3
    assert len(bs._nodes) == 6
    assert len(bs._node_attrs) == 6
    assert set(bs._nodes.keys()) == {
        ('A', 'e'),
        ('B', 'e'),
        ('B', 'f'),
        ('C', 'e'),
        ('C', 'f'),
        ('D', 'e'),
    }
    assert bs._nodes[('A', 'e')] == ({0, 1}, set())
    assert bs._edges[0] == (
        {('A', 'e'), ('B', 'f')},
        {('D', 'e'), ('B', 'e')},
    )
    assert set(bs._edge_attrs.columns) == {_c.EDGE_ID, 'weight', 'style'}


def test_proc_node_key(toy_network_binary_str):

    edges, nodes = toy_network_binary_str
    bs = BootstrapDf(edges = edges, nodes = nodes, node_key = 'id')
    node_key = bs._proc_node_key('A')

    assert node_key == 'A'

    edges, nodes = toy_network_binary_str
    custom_edges = pd.DataFrame({
        'source': ['A_x;B_y', 'B_x;C_y'],
        'target': ['C_z;A_u', 'A_z;B_y'],
    })

    try:

        bs = BootstrapDf(
            edges = custom_edges,
            nodes = nodes,
            node_key = ('id', 'species'),
            node_key_sep = '_',
        )

    except ValueError as e:

        assert str(e).startswith('Columns in `node_key` not found')

    bs = BootstrapDf(
        edges = custom_edges,
        node_key = ('id', 'species'),
        node_key_sep = '_',
    )
    node_key = bs._proc_node_key('A_x')

    assert node_key == ('A', 'x')
    assert bs._edges[0][0] == {('A', 'x'), ('B', 'y')}


def test_edge_processing_with_custom_separator(toy_network_binary_str):

    edges, nodes = toy_network_binary_str
    custom_edges = pd.DataFrame({
        'source': ['A,B', 'B,C'],
        'target': ['C,A', 'A,B'],
    })
    bs = BootstrapDf(
        edges = custom_edges,
        nodes = nodes,
        inner_sep = ',',
        node_key = 'id',
    )

    assert len(bs._edges) == 2
    assert 'A' in bs._nodes
    assert 'B' in bs._nodes
    assert 'C' in bs._nodes
