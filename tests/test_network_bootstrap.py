import itertools
import copy

import numpy as np

from networkcommons.network import _bootstrap
from networkcommons.network import _constants as _c


def test_nodes_list_str():

    nodes = ['a', 'b', 'c']

    result = _bootstrap.Bootstrap(nodes = nodes)

    assert set(result._nodes.keys()) == set(nodes)
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert set(result._node_attrs.columns) == {_c.DEFAULT_KEY, _c.NODE_KEY}
    assert set(result._node_attrs[_c.DEFAULT_KEY]) == set(nodes)


def test_node_list_dict(nodes_list_dict):

    nodes = nodes_list_dict
    nodes_original = copy.deepcopy(nodes)
    node_key = 'name'

    result = _bootstrap.Bootstrap(nodes = nodes, node_key = node_key)

    node_ids = {n['name'] for n in nodes}

    assert set(result._nodes.keys()) == node_ids
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert (
        set(result._node_attrs.columns) ==
        {'name', 'color', 'size', _c.NODE_KEY}
    )
    assert set(result._node_attrs['name']) == node_ids
    assert set(result._node_attrs['color']) == {n['color'] for n in nodes}
    assert nodes == nodes_original


def test_node_tuple_ids(nodes_list_dict_2):

    nodes = nodes_list_dict_2
    nodes_original = copy.deepcopy(nodes)
    node_key = ('name', 'color')

    result = _bootstrap.Bootstrap(nodes = nodes, node_key = node_key)

    node_ids = {(n['name'], n['color']) for n in nodes}

    assert np.isnan(
        result._node_attrs[
            (result._node_attrs['name'] == 'c') &
            (result._node_attrs['color'] == 'red')
        ]['size'].iloc[0]
    )
    assert (
        set(result._node_attrs.columns) ==
        {'name', 'color', 'size', _c.NODE_KEY}
    )
    assert set(result._node_attrs[_c.NODE_KEY]) == node_ids
    assert set(result._node_attrs['color']) == {n['color'] for n in nodes}
    assert nodes == nodes_original


def test_node_dict_dict():

    nodes = {'a': {'color': 'blue'}, 'b': {'color': 'red'}, 'c': {'color': 'green'}}
    nodes_original = copy.deepcopy(nodes)
    result = _bootstrap.Bootstrap(nodes = nodes)

    assert set(result._nodes.keys()) == set(nodes.keys())
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert set(result._node_attrs.columns) == {_c.DEFAULT_KEY, 'color', _c.NODE_KEY}
    assert set(result._node_attrs[_c.DEFAULT_KEY]) == set(nodes.keys())
    assert set(result._node_attrs['color']) == set(nodes[n]['color'] for n in nodes.keys())
    assert nodes == nodes_original


def test_nodes_empty_list():

    nodes = []
    result = _bootstrap.Bootstrap(nodes=nodes)

    # Test that no nodes are in the _nodes dictionary
    assert len(result._nodes) == 0

    # Test that _edges is still an empty dictionary
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0

    # Test that _node_attrs DataFrame is empty
    assert result._node_attrs.empty


def test_duplicate_nodes():

    nodes = ['a', 'b', 'a']
    result = _bootstrap.Bootstrap(nodes=nodes)

    # Ensure duplicates are handled and only unique nodes are stored
    assert len(result._nodes) == len(set(nodes))
    assert set(result._nodes.keys()) == set(nodes)

    # Ensure the node attributes DataFrame has only unique nodes
    assert len(result._node_attrs) == len(set(nodes))


def test_invalid_node_key():

    nodes = [
        {'name': 'a', 'color': 'blue'},
        {'name': 'b', 'color': 'red'}
    ]
    node_key = 'invalid_key'

    try:
        _bootstrap.Bootstrap(nodes=nodes, node_key=node_key)
    except ValueError as e:
        # Ensure the proper exception is raised when an invalid node_key is provided
        assert str(e) == 'Node with empty key: `{}`.'.format(nodes[0])


def test_mixed_node_types():

    nodes = [
        'a',
        {'name': 'b', 'color': 'red'},
        'c'
    ]
    node_key = 'name'

    result = _bootstrap.Bootstrap(nodes=nodes, node_key=node_key)

    # Mixed types should raise an exception or be processed correctly
    node_ids = {'b', 'a', 'c'}
    assert set(result._nodes.keys()) == node_ids
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert 'name' in result._node_attrs.columns
    assert 'color' in result._node_attrs.columns


def test_node_dict_with_partial_attrs():

    nodes = [
        {'name': 'a', 'color': 'blue', 'size': 10.0},
        {'name': 'b', 'color': 'red'},
        {'name': 'c'},
        {'name': 'd', 'size': 5.0}
    ]
    nodes_original = copy.deepcopy(nodes)
    node_key = 'name'

    result = _bootstrap.Bootstrap(nodes=nodes, node_key=node_key)

    node_ids = {n['name'] for n in nodes}
    assert set(result._nodes.keys()) == node_ids
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert (
            set(result._node_attrs.columns) ==
            {'name', 'color', 'size', _c.NODE_KEY}
    )
    assert set(result._node_attrs['name']) == node_ids
    assert set(result._node_attrs['color']) == {n.get('color', np.nan) for n in nodes}
    assert (
        {i for i in result._node_attrs['size'] if not np.isnan(i)} ==
        {n['size'] for n in nodes if 'size' in n}
    )
    assert nodes == nodes_original


def test_edges_list_of_tuples():

    edges = [('a', 'b'), ('b', 'c'), ('a', 'c')]
    result = _bootstrap.Bootstrap(edges = edges)

    # Ensure the edges are stored correctly
    assert len(result._edges) == len(edges)
    assert result._nodes['b'] == ({1}, {0})
    assert set(result._edge_attrs.columns) == {_c.EDGE_ID}
    assert set(result._edge_attrs[_c.EDGE_ID]) == set(range(len(edges)))
    assert set(result._edges.keys()) == set(range(len(edges)))
    assert (
        sorted(result._edges.values()) ==
        sorted(({s}, {t}) for s, t in edges)
    )

    # Ensure the nodes are extracted from the edges
    nodes = set(itertools.chain(*edges))
    assert set(result._nodes.keys()) == nodes
    assert result._node_attrs.shape[0] == len(nodes)
    assert result._node_attrs[_c.NODE_KEY].isin(nodes).all()
    assert set(result._node_attrs.columns) == {_c.DEFAULT_KEY, _c.NODE_KEY}


def test_edges_list_of_dicts(edges_list_dict):

    edges = edges_list_dict
    edges_original = copy.deepcopy(edges)
    result = _bootstrap.Bootstrap(edges = edges)

    # Ensure the edges are stored correctly with attributes
    assert len(result._edges) == len(edges)
    assert result._nodes['b'] == ({1}, {0})
    assert set(result._edge_attrs.columns) == {_c.EDGE_ID, 'weight', 'style'}
    assert set(result._edge_attrs[_c.EDGE_ID]) == set(range(len(edges)))
    assert set(result._edges.keys()) == set(range(len(edges)))
    assert (
        sorted(result._edges.values()) ==
        sorted(({e['source']}, {e['target']}) for e in edges)
    )

    # Check the edge attributes
    assert 'weight' in result._edge_attrs.columns
    assert set(result._edge_attrs['weight']) == {e['weight'] for e in edges}

    # Ensure the nodes are extracted from the edges
    nodes = {e[side] for e in edges for side in ('source', 'target')}
    assert set(result._nodes.keys()) == nodes
    assert result._node_attrs.shape[0] == len(nodes)
    assert result._node_attrs[_c.NODE_KEY].isin(nodes).all()
    assert set(result._node_attrs.columns) == {_c.DEFAULT_KEY, _c.NODE_KEY}
    assert edges == edges_original


def test_edgenode_attributes_1():

    _edges = (
        [
            {
                'source': 'a',
                'target': 'b',
                'weight': 1.5,
                'source_attrs': {'color': 'blue'},
                'target_attrs': {'color': 'red'},
            },
            {
                'source': 'b',
                'target': 'c',
                'weight': 2.3,
                'source_attrs': {'color': 'green'},
                'target_attrs': {'color': 'purple'},
            },
        ],
        [
            {
                'source': 'a',
                'target': 'b',
                'weight': 1.5,
                'source_attrs': {'a': {'color': 'blue'}},
                'target_attrs': {'b': {'color': 'red'}},
            },
            {
                'source': 'b',
                'target': 'c',
                'weight': 2.3,
                'source_attrs': {'b': {'color': 'green'}},
                'target_attrs': {'c': {'color': 'purple'}},
            },
        ],
    )

    for edges in _edges:

        edges_original = copy.deepcopy(edges)
        result = _bootstrap.Bootstrap(edges=edges)
        neattrs = result._edge_node_attrs

        # Ensure nodes and edges are stored
        assert len(result._nodes) == 3
        assert len(result._edges) == 2

        # Check that node-edge attributes are stored correctly
        assert set(neattrs.columns) == {_c.NODE_KEY, _c.EDGE_ID, _c.SIDE, 'color'}
        assert len(neattrs) == len(edges) * 2  # source and target for each edge
        assert (
            neattrs[
                (neattrs[_c.NODE_KEY] == 'b') &
                (neattrs[_c.SIDE] == 'source')
            ]['color'].iloc[0] == 'green'
        )
        assert edges == edges_original


def test_edgenode_attributes_2():

    _edges = (
        [
            {
                'source': {'a', 'x'},
                'target': {'b', 'y'},
                'weight': 1.5,
                'source_attrs': {
                    'a': {'color': 'blue'},
                    'x': {'color': 'green'},
                },
                'target_attrs': {
                    'b': {'color': 'red'},
                    'y': {'color': 'purple'},
                },
            },
            {
                'source': {'b', 'y'},
                'target': {'c', 'x'},
                'weight': 2.3,
                'source_attrs': {
                    'b': {'color': 'blue'},
                    'y': {'color': 'green'},
                },
                'target_attrs': {
                    'c': {'color': 'red'},
                    'x': {'color': 'purple'},
                },
            },
        ],
        [
            {
                'source': [
                    {'name': 'a', 'color': 'blue'},
                    {'name': 'x', 'color': 'green'},
                ],
                'target': [
                    {'name': 'b', 'color': 'red'},
                    {'name': 'y', 'color': 'purple'},
                ],
                'weight': 1.5,
            },
            {
                'source': [
                    {'name': 'b', 'color': 'blue'},
                    {'name': 'y', 'color': 'green'},
                ],
                'target': [
                    {'name': 'c', 'color': 'red'},
                    {'name': 'x', 'color': 'purple'},
                ],
                'weight': 2.3,
            },
        ],
    )

    for edges in _edges:

        edges_original = copy.deepcopy(edges)
        result = _bootstrap.Bootstrap(edges = edges, node_key = 'name')
        neattrs = result._edge_node_attrs

        # Ensure nodes and edges are stored
        assert len(result._nodes) == 5
        assert len(result._edges) == 2
        assert set(result._node_attrs.columns) == {_c.NODE_KEY, 'name'}
        assert set(result._edge_attrs.columns) == {_c.EDGE_ID, 'weight'}

        # Check that node-edge attributes are stored correctly
        assert set(neattrs.columns) == {_c.NODE_KEY, _c.EDGE_ID, _c.SIDE, 'color'}
        assert len(neattrs) == len(edges) * 2 * 2
        assert (
            neattrs[
                (neattrs[_c.NODE_KEY] == 'b') &
                (neattrs[_c.SIDE] == 'source')
            ]['color'].iloc[0] == 'blue'
        )
        assert edges == edges_original


def test_undirected_edges():

    edges = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    result = _bootstrap.Bootstrap(edges = edges, directed = False)

    # Ensure all nodes are on the source side
    assert len(result._edges) == len(edges)
    assert not any(e[1] for e in result._edges.values())
    assert all(len(e[0]) == 2 for e in result._edges.values())
    assert result._nodes['b'] == ({0, 1}, set())
    assert set(result._edge_node_attrs[_c.SIDE]) == {'source'}


def test_edges_missing_keys():

    edges = [{'source': 'a'}, {'source': 'b', 'target': 'c'}]

    try:
        _bootstrap.Bootstrap(edges=edges)
    except ValueError as e:
        # Ensure a ValueError is raised for missing mandatory keys
        assert str(e).startswith('Edge `{')
