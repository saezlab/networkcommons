"""
NetworkBase is instantiated here in three ways resulting identical contents.
"""

import itertools

import pandas as pd

from networkcommons.network import _network
from networkcommons.network import _constants as _c


def test_edgelist():

    edges = [('a', 'b'), ('b', 'c'), ('a', 'c')]
    nodes = {
        'a': {'color': 'blue'},
        'b': {'color': 'red'},
        'c': {'color': 'green'},
    }
    result = _network.NetworkBase(edges = edges, nodes = nodes)

    # Ensure the edges are stored correctly
    assert len(result.edges) == len(edges)
    assert result.nodes['b'] == ({1}, {0})
    assert set(result.edge_attrs.columns) == {_c.EDGE_ID}
    assert set(result.edge_attrs[_c.EDGE_ID]) == set(range(len(edges)))
    assert set(result.edges.keys()) == set(range(len(edges)))
    assert (
        sorted(result.edges.values()) ==
        sorted(({s}, {t}) for s, t in edges)
    )

    # Ensure the nodes are extracted from the edges
    nodes = set(itertools.chain(*edges))
    assert set(result.nodes.keys()) == nodes
    assert result.node_attrs.shape[0] == len(nodes)
    assert result.node_attrs[_c.NODE_KEY].isin(nodes).all()
    assert set(result.node_attrs.columns) == {_c.NODE_KEY, _c.DEFAULT_KEY, 'color'}

    assert result.node_attrs.index.is_monotonic_increasing
    assert (result.node_attrs.index == result.node_attrs[_c.NODE_KEY]).all()
    assert result.edge_attrs.index.is_monotonic_increasing
    assert (result.edge_attrs.index == result.edge_attrs[_c.EDGE_ID]).all()


def test_df():

    edges = pd.DataFrame(
        {
            'source': ['a', 'b', 'a'],
            'target': ['b', 'c', 'c'],
        },
    )
    nodes = pd.DataFrame(
        {
            _c.DEFAULT_KEY: ['a', 'b', 'c'],
            'color': ['blue', 'red', 'green'],
        },
    )
    result = _network.NetworkBase(edges = edges, nodes = nodes)

    # Ensure the edges are stored correctly
    assert len(result.edges) == len(edges)
    assert result.nodes['b'] == ({1}, {0})
    assert set(result.edge_attrs.columns) == {_c.EDGE_ID}
    assert set(result.edge_attrs[_c.EDGE_ID]) == set(range(len(edges)))
    assert set(result.edges.keys()) == set(range(len(edges)))
    assert (
        sorted(result.edges.values()) ==
        sorted(({s}, {t}) for s, t in zip(edges.source, edges.target))
    )

    # Ensure the nodes are extracted from the edges
    nodes = set(itertools.chain(edges.source, edges.target))
    assert set(result.nodes.keys()) == nodes
    assert result.node_attrs.shape[0] == len(nodes)
    assert result.node_attrs[_c.NODE_KEY].isin(nodes).all()
    assert set(result.node_attrs.columns) == {_c.NODE_KEY, _c.DEFAULT_KEY, 'color'}

    assert result.node_attrs.index.is_monotonic_increasing
    assert (result.node_attrs.index == result.node_attrs[_c.NODE_KEY]).all()
    assert result.edge_attrs.index.is_monotonic_increasing
    assert (result.edge_attrs.index == result.edge_attrs[_c.EDGE_ID]).all()


def test_copy():

    edges = [('a', 'b'), ('b', 'c'), ('a', 'c')]
    nodes = {
        'a': {'color': 'blue'},
        'b': {'color': 'red'},
        'c': {'color': 'green'},
    }
    parent = _network.NetworkBase(edges = edges, nodes = nodes)
    result = _network.NetworkBase(parent)

    # Ensure the edges are stored correctly
    assert len(result.edges) == len(edges)
    assert result.nodes['b'] == ({1}, {0})
    assert set(result.edge_attrs.columns) == {_c.EDGE_ID}
    assert set(result.edge_attrs[_c.EDGE_ID]) == set(range(len(edges)))
    assert set(result.edges.keys()) == set(range(len(edges)))
    assert (
        sorted(result.edges.values()) ==
        sorted(({s}, {t}) for s, t in edges)
    )

    # Ensure the nodes are extracted from the edges
    nodes = set(itertools.chain(*edges))
    assert set(result.nodes.keys()) == nodes
    assert result.node_attrs.shape[0] == len(nodes)
    assert result.node_attrs[_c.NODE_KEY].isin(nodes).all()
    assert set(result.node_attrs.columns) == {_c.NODE_KEY, _c.DEFAULT_KEY, 'color'}

    assert result.node_attrs.index.is_monotonic_increasing
    assert (result.node_attrs.index == result.node_attrs[_c.NODE_KEY]).all()
    assert result.edge_attrs.index.is_monotonic_increasing
    assert (result.edge_attrs.index == result.edge_attrs[_c.EDGE_ID]).all()
