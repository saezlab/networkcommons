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

# def test_node_dict_dict():
#     nodes = {'a': {'color': 'blue'}, 'b': {'color': 'red'}, 'c': {'color': 'green'}}
#     node_attrs = {'color': 'red'}
#     result = _bootstrap.Bootstrap(nodes = nodes, node_attrs = node_attrs)
#
#     assert all((n,) in result._nodes.keys() for n in nodes.keys())
#     assert all(result._nodes[(n,)]['color'] == nodes[n]['color'] for n in nodes.keys())
#     assert isinstance(result._edges, dict)
#     assert len(result._edges) == 0
#     assert set(result._node_attrs.columns) == {'_id', 'color'}
#     assert set(result._node_attrs['_id']) == set(nodes.keys())
#     assert set(result._node_attrs['color']) == set(node_attrs['color'] for n in nodes.keys())
