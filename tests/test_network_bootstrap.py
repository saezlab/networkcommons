import numpy as np

from networkcommons.network import _bootstrap
from networkcommons.network import _constants as _c


def test_nodes_list_str():

    nodes = ['a', 'b', 'c']

    result = _bootstrap.Bootstrap(nodes = nodes)

    assert all((n,) in result._nodes.keys() for n in nodes)
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert set(result._node_attrs.columns) == {_c.DEFAULT_KEY, _c.NODE_KEY}
    assert set(result._node_attrs[_c.DEFAULT_KEY]) == set(nodes)


def test_node_list_dict():

    nodes = [
        {'name': 'a', 'color': 'blue', 'size': 10},
        {'name': 'b', 'color': 'red', 'size': 17},
        {'name': 'c', 'color': 'green', 'size': 9},
    ]
    node_key = 'name'

    result = _bootstrap.Bootstrap(nodes = nodes, node_key = node_key)

    node_ids = {n['name'] for n in nodes}

    assert all((n,) in result._nodes.keys() for n in node_ids)
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert (
        set(result._node_attrs.columns) ==
        {'name', 'color', 'size', _c.NODE_KEY}
    )
    assert set(result._node_attrs['name']) == node_ids
    assert set(result._node_attrs['color']) == {n['color'] for n in nodes}


def test_node_tuple_ids():

    nodes = [
        {'name': 'a', 'color': 'blue', 'size': 10},
        {'name': 'b', 'color': 'red', 'size': 17},
        {'name': 'c', 'color': 'green', 'size': 9},
        {'name': 'c', 'color': 'red'},
    ]
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


def test_node_dict_dict():
    nodes = {'a': {'color': 'blue'}, 'b': {'color': 'red'}, 'c': {'color': 'green'}}
    result = _bootstrap.Bootstrap(nodes = nodes)

    assert all((n,) in result._nodes.keys() for n in nodes.keys())
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert set(result._node_attrs.columns) == {_c.DEFAULT_KEY, 'color', _c.NODE_KEY}
    assert set(result._node_attrs[_c.DEFAULT_KEY]) == set(nodes.keys())
    assert set(result._node_attrs['color']) == set(nodes[n]['color'] for n in nodes.keys())


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
    assert all((n,) in result._nodes.keys() for n in set(nodes))

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
    node_ids = {('b',), ('a',), ('c',)}
    assert all((n,) in result._nodes.keys() for n in node_ids)
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
    node_key = 'name'

    result = _bootstrap.Bootstrap(nodes=nodes, node_key=node_key)

    node_ids = {n['name'] for n in nodes}
    assert all((n,) in result._nodes.keys() for n in node_ids)
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

def test_edges_list_of_tuples():
    edges = [('a', 'b'), ('b', 'c'), ('a', 'c')]
    result = _bootstrap.Bootstrap(edges=edges)

    # Ensure the edges are stored correctly
    assert len(result._edges) == len(edges)
    assert all(
        (source, target) in result._edges.values()
        for source, target in edges
    )

    # Ensure the nodes are extracted from the edges
    nodes = set(itertools.chain(*edges))
    assert all((n,) in result._nodes.keys() for n in nodes)


def test_edges_list_of_dicts():
    edges = [
        {'source': 'a', 'target': 'b', 'weight': 1.5},
        {'source': 'b', 'target': 'c', 'weight': 2.3},
        {'source': 'a', 'target': 'c', 'weight': 0.9}
    ]
    result = _bootstrap.Bootstrap(edges=edges)

    # Ensure the edges are stored correctly with attributes
    assert len(result._edges) == len(edges)
    assert all(
        (source['source'], source['target']) in result._edges.values()
        for source in edges
    )

    # Check the edge attributes
    assert 'weight' in result._edge_attrs.columns
    assert set(result._edge_attrs['weight']) == {e['weight'] for e in edges}


def test_edges_with_node_attributes():
    edges = [
        {'source': 'a', 'target': 'b', 'weight': 1.5, 'source_attrs': {'color': 'blue'},
         'target_attrs': {'color': 'red'}},
        {'source': 'b', 'target': 'c', 'weight': 2.3, 'source_attrs': {'color': 'red'},
         'target_attrs': {'color': 'green'}}
    ]
    result = _bootstrap.Bootstrap(edges=edges)

    # Ensure nodes from edges are stored with attributes
    assert len(result._nodes) == 3  # a, b, c

    # Check that node-edge attributes are stored correctly
    assert 'color' in result._node_edge_attrs.columns
    assert len(result._node_edge_attrs) == len(edges) * 2  # source and target for each edge


def test_directed_edges():
    edges = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    result = _bootstrap.Bootstrap(edges=edges, directed=True)

    # Ensure edges are stored correctly and directed
    assert len(result._edges) == len(edges)
    assert all(
        (source, target) in result._edges.values()
        for source, target in edges
    )

    # Ensure directionality is respected
    for edge in edges:
        assert edge in result._edges.values()


def test_undirected_edges():
    edges = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    result = _bootstrap.Bootstrap(edges=edges, directed=False)

    # Ensure edges are stored correctly and undirected
    assert len(result._edges) == len(edges)
    assert all(
        {(source, target)} == {result._edges[i]}
        for i, (source, target) in enumerate(edges)
    )

    # Ensure no directionality is enforced
    for edge in edges:
        source, target = edge
        assert any(
            {source, target} == set(e)
            for e in result._edges.values()
        )


def test_edges_missing_keys():
    edges = [{'source': 'a'}, {'source': 'b', 'target': 'c'}]

    try:
        _bootstrap.Bootstrap(edges=edges)
    except ValueError as e:
        # Ensure a ValueError is raised for missing mandatory keys
        assert str(e).startswith('Edge `{')
