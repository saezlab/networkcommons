from networkcommons.network import _bootstrap


def test_nodes_list_str():

    nodes = ['a', 'b', 'c']
    result = _bootstrap.Bootstrap(nodes = nodes)

    assert all((n,) in result._nodes.keys() for n in nodes)
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert set(result._node_attrs.columns) == {'id'}
    assert set(result._node_attrs['id']) == set(nodes)


def test_node_dict():
    nodes = {'a': {'color': 'blue'}, 'b': {'color': 'red'}, 'c': {'color': 'green'}}
    result = _bootstrap.Bootstrap(nodes = nodes)

    assert all((n,) in result._nodes.keys() for n in nodes.keys())
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert set(result._node_attrs.columns) == {'id', 'color'}
    assert set(result._node_attrs['id']) == set(nodes.keys())
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
#     assert set(result._node_attrs.columns) == {'id', 'color'}
#     assert set(result._node_attrs['id']) == set(nodes.keys())
#     assert set(result._node_attrs['color']) == set(node_attrs['color'] for n in nodes.keys())