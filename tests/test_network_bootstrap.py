from networkcommons.network import _bootstrap


def test_nodes_list_str():

    nodes = ['a', 'b', 'c']
    result = _bootstrap.Bootstrap(nodes = nodes)

    assert all((n,) in result._nodes.keys() for n in nodes)
    assert isinstance(result._edges, dict)
    assert len(result._edges) == 0
    assert set(result._node_attrs.columns) == {'id'}
    assert set(result._node_attrs['id']) == set(nodes)
