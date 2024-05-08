import networkx as nx
from networkcommons.methods import (
    run_shortest_paths,
    run_reachability_filter,
    run_all_paths,
    run_corneto_carnival,
    run_sign_consistency,
    add_pagerank_scores,
    compute_ppr_overlap)
import corneto as cn


def test_run_shortest_paths():
    # Crea un grafo de prueba
    network = nx.DiGraph()
    network.add_edge('A', 'B', weight=1)
    network.add_edge('B', 'C', weight=2)
    network.add_edge('C', 'D', weight=3)
    network.add_edge('A', 'D', weight=6)
    network.add_edge('A', 'E', weight=4)
    network.add_edge('E', 'F', weight=5)

    source_dict = {'A': 1}
    target_dict = {'D': 1}

    subnetwork, shortest_paths_res = run_shortest_paths(network,
                                                        source_dict,
                                                        target_dict)

    assert list(subnetwork.edges) == [('A', 'D'),
                                      ('A', 'B'),
                                      ('B', 'C'),
                                      ('C', 'D')]
    assert shortest_paths_res == [['A', 'D'], ['A', 'B', 'C', 'D']]


def test_run_sign_consistency():
    network = nx.DiGraph()
    network.add_edge('A', 'B', weight=1, sign=1)
    network.add_edge('B', 'C', weight=2, sign=1)
    network.add_edge('C', 'D', weight=3, sign=-1)
    network.add_edge('A', 'D', weight=6, sign=1)
    network.add_edge('A', 'E', weight=4, sign=1)
    network.add_edge('E', 'F', weight=5, sign=-1)

    source_dict = {'A': 1}
    target_dict = {'D': -1}

    paths = [['A', 'B', 'C', 'D']]

    subnetwork, sign_consistency_res = run_sign_consistency(network,
                                                            paths,
                                                            source_dict,
                                                            target_dict)

    assert list(subnetwork.edges) == [('A', 'B'), ('B', 'C'), ('C', 'D')]
    assert sign_consistency_res == [['A', 'B', 'C', 'D']]


def test_run_reachability_filter():
    network = nx.DiGraph()
    network.add_edge('A', 'B')
    network.add_edge('B', 'C')
    network.add_edge('C', 'D')
    network.add_edge('A', 'D')
    network.add_edge('A', 'E')
    network.add_edge('E', 'F')

    source_dict = {'B': 1}

    subnetwork = run_reachability_filter(network, source_dict)

    assert list(subnetwork.edges) == [('B', 'C'), ('C', 'D')]


def test_run_all_paths():
    network = nx.DiGraph()

    network.add_edge('A', 'B', weight=1)
    network.add_edge('B', 'C', weight=2)
    network.add_edge('C', 'D', weight=3)
    network.add_edge('A', 'D', weight=10)
    network.add_edge('D', 'E', weight=4)
    network.add_edge('E', 'F', weight=5)
    network.add_edge('X', 'A', weight=7)
    network.add_edge('Y', 'A', weight=8)

    source_dict = {'A': 1}
    target_dict = {'D': 1}

    subnetwork, all_paths_res = run_all_paths(network,
                                              source_dict,
                                              target_dict)

    assert list(subnetwork.edges) == [('A', 'B'),
                                      ('A', 'D'),
                                      ('B', 'C'),
                                      ('C', 'D')]
    assert all_paths_res == [['A', 'B', 'C', 'D'], ['A', 'D']]


def test_add_pagerank_scores():
    network = nx.DiGraph()
    network.add_edge('A', 'B', weight=1)
    network.add_edge('B', 'C', weight=2)
    network.add_edge('C', 'D', weight=3)
    network.add_edge('A', 'D', weight=10)
    network.add_edge('D', 'E', weight=4)
    network.add_edge('E', 'F', weight=5)
    network.add_edge('X', 'A', weight=7)
    network.add_edge('Y', 'A', weight=8)

    source_dict = {'A': 1}
    target_dict = {'D': 1}

    network_with_pagerank = add_pagerank_scores(network,
                                                source_dict,
                                                target_dict,
                                                personalize_for='source')
    network_with_pagerank = add_pagerank_scores(network,
                                                source_dict,
                                                target_dict,
                                                personalize_for='target')

    network.nodes['A']['pagerank_from_sources'] = 0.3053985948347749
    network.nodes['A']['pagerank_from_targets'] = 0.2806554600002494

    network.nodes['B']['pagerank_from_sources'] = 0.023598856445643426
    network.nodes['B']['pagerank_from_targets'] = 0.05881811234250767

    network.nodes['C']['pagerank_from_sources'] = 0.020059163994325776
    network.nodes['C']['pagerank_from_targets'] = 0.06919815804909274

    network.nodes['D']['pagerank_from_sources'] = 0.25303871980865306
    network.nodes['D']['pagerank_from_targets'] = 0.3527711939246324

    network.nodes['E']['pagerank_from_sources'] = 0.21508439476242489
    network.nodes['E']['pagerank_from_targets'] = 0.0

    network.nodes['F']['pagerank_from_sources'] = 0.18282027015417815
    network.nodes['F']['pagerank_from_targets'] = 0.0

    network.nodes['X']['pagerank_from_sources'] = 0.0
    network.nodes['X']['pagerank_from_targets'] = 0.111326635318975

    network.nodes['Y']['pagerank_from_sources'] = 0.0
    network.nodes['Y']['pagerank_from_targets'] = 0.1272304403645429

    assert nx.is_isomorphic(network, network_with_pagerank)

    for node in network.nodes:
        assert network.nodes[node] == network_with_pagerank.nodes[node]

    for edge in network.edges:
        assert network.edges[edge] == network_with_pagerank.edges[edge]


def test_compute_ppr_overlap():
    network = nx.DiGraph()
    network.add_edge('A', 'B', weight=1)
    network.add_edge('B', 'C', weight=2)
    network.add_edge('C', 'D', weight=3)
    network.add_edge('A', 'D', weight=10)
    network.add_edge('D', 'E', weight=4)
    network.add_edge('E', 'F', weight=5)
    network.add_edge('X', 'A', weight=7)
    network.add_edge('Y', 'A', weight=8)

    network.nodes['A']['pagerank_from_sources'] = 0.3053985948347749
    network.nodes['A']['pagerank_from_targets'] = 0.2806554600002494

    network.nodes['B']['pagerank_from_sources'] = 0.023598856445643426
    network.nodes['B']['pagerank_from_targets'] = 0.05881811234250767

    network.nodes['C']['pagerank_from_sources'] = 0.020059163994325776
    network.nodes['C']['pagerank_from_targets'] = 0.06919815804909274

    network.nodes['D']['pagerank_from_sources'] = 0.25303871980865306
    network.nodes['D']['pagerank_from_targets'] = 0.3527711939246324

    network.nodes['E']['pagerank_from_sources'] = 0.21508439476242489
    network.nodes['E']['pagerank_from_targets'] = 0.0

    network.nodes['F']['pagerank_from_sources'] = 0.18282027015417815
    network.nodes['F']['pagerank_from_targets'] = 0.0

    network.nodes['X']['pagerank_from_sources'] = 0.0
    network.nodes['X']['pagerank_from_targets'] = 0.111326635318975

    network.nodes['Y']['pagerank_from_sources'] = 0.0
    network.nodes['Y']['pagerank_from_targets'] = 0.1272304403645429

    subnetwork = compute_ppr_overlap(network)

    test_network = nx.DiGraph()
    test_network.add_edge('A', 'D', weight=10)

    test_network.nodes['A']['pagerank_from_sources'] = 0.3053985948347749
    test_network.nodes['A']['pagerank_from_targets'] = 0.2806554600002494

    test_network.nodes['D']['pagerank_from_sources'] = 0.25303871980865306
    test_network.nodes['D']['pagerank_from_targets'] = 0.3527711939246324

    assert nx.is_isomorphic(test_network, subnetwork)

    for node in test_network.nodes:
        assert test_network.nodes[node] == subnetwork.nodes[node]

    for edge in test_network.edges:
        assert test_network.edges[edge] == subnetwork.edges[edge]


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

    result_network = run_corneto_carnival(network,
                                          source_dict,
                                          target_dict,
                                          betaWeight=0.1,
                                          solver='scipy')

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
