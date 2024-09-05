import pytest
import networkx as nx
from networkcommons.network._formats._networkx import to_networkx
from networkcommons.network._network import NetworkBase

@pytest.fixture
def small_network():
    nodes = ['A', 'B', 'C']
    edges = [('A', 'B'), ('B', 'C')]
    network = NetworkBase(edges = edges, nodes = nodes)
    return network


def test_to_networkx(small_network):
    #create a networkbase object
    #convert it to networkx
    netx = to_networkx(small_network)
    assert isinstance(netx, nx.Graph)
