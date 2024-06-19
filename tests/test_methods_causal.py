import networkx as nx
import corneto as cn

from networkcommons._methods import causal


def test_run_corneto_carnival():

    # From CORNETO docs:
    network = cn.Graph.from_sif_tuples([
        ('I1', 1, 'N1'),  # I1 activates N1
        ('N1', 1, 'M1'),  # N1 activates M1
        ('N1', 1, 'M2'),  # N1 activates M2
        ('I2', -1, 'N2'),  # I2 inhibits N2
        ('N2', -1, 'M2'),  # N2 inhibits M2
        ('N2', -1, 'M1'),  # N2 inhibits M1
    ])

    source_dict = {'I1': 1}
    target_dict = {'M1': 1, 'M2': 1}

    result_network = causal.run_corneto_carnival(
        network,
        source_dict,
        target_dict,
        betaWeight = .1,
        solver = 'scipy',
    )

    test_network = nx.DiGraph()
    test_network.add_edge('I1', 'N1')
    test_network.add_edge('N1', 'M1')
    test_network.add_edge('N1', 'M2')

    assert nx.is_isomorphic(result_network, test_network)
    assert isinstance(result_network, nx.Graph)
    assert list(result_network.nodes) == ['I1', 'N1', 'M1', 'M2']
    assert '_s' not in result_network.nodes
    assert '_pert_c0' not in result_network.nodes
    assert '_meas_c0' not in result_network.nodes
