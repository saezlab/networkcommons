import pytest

import pandas as pd
import networkx as nx

from networkcommons.methods import _graph


def _network(weights: bool = False, signs: bool = False) -> nx.DiGraph:

    edges = pd.DataFrame(
        {
            'source': ['A', 'B', 'C', 'A', 'A', 'E'],
            'target': ['B', 'C', 'D', 'D', 'E', 'F'],
            'weight': [1, 2, 3, 6, 4, 5],
            'sign': [1, 1, -1, 1, 1, -1],
        }
    )

    if not weights:
        edges = edges.drop(columns=['weight'])

    if not signs:
        edges = edges.drop(columns=['sign'])

    network = nx.from_pandas_edgelist(
        edges,
        'source',
        'target',
        edge_attr = weights or signs or None,
        create_using = nx.DiGraph,
    )

    return network


@pytest.fixture
def net():

    return _network()


@pytest.fixture
def net_weighted():

    return _network(weights = True)


@pytest.fixture
def net_signed():

    return _network(weights = True, signs = True)


@pytest.fixture
def net2():

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

    return network, source_dict, target_dict


def test_run_shortest_paths(net_weighted):
    # Crea un grafo de prueba
    source_dict = {'A': 1}
    target_dict = {'D': 1}

    subnetwork, shortest_paths_res = _graph.run_shortest_paths(
        net_weighted,
        source_dict,
        target_dict,
    )

    assert (
        list(subnetwork.edges) ==
        [
            ('A', 'D'),
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'D'),
        ]
    )
    assert shortest_paths_res == [['A', 'D'], ['A', 'B', 'C', 'D']]


def test_run_sign_consistency(net_signed):

    source_dict = {'A': 1}
    target_dict = {'D': -1}

    paths = [['A', 'B', 'C', 'D']]

    subnetwork, sign_consistency_res = _graph.run_sign_consistency(
        net_signed,
        paths,
        source_dict,
        target_dict,
    )

    assert list(subnetwork.edges) == [('A', 'B'), ('B', 'C'), ('C', 'D')]
    assert sign_consistency_res == [['A', 'B', 'C', 'D']]


def test_run_reachability_filter(net):

    source_dict = {'B': 1}

    subnetwork = _graph.run_reachability_filter(net, source_dict)

    assert list(subnetwork.edges) == [('B', 'C'), ('C', 'D')]


def test_run_all_paths(net_weighted):

    source_dict = {'A': 1}
    target_dict = {'D': 1}

    subnetwork, all_paths_res = _graph.run_all_paths(
        net_weighted,
        source_dict,
        target_dict,
    )

    assert list(subnetwork.edges) == [('A', 'B'),
                                      ('A', 'D'),
                                      ('B', 'C'),
                                      ('C', 'D')]
    assert all_paths_res == [['A', 'B', 'C', 'D'], ['A', 'D']]


def test_add_pagerank_scores(net2):

    network, source_dict, target_dict = net2

    network_with_pagerank = _graph.add_pagerank_scores(
        network,
        source_dict,
        target_dict,
        personalize_for='source',
    )
    network_with_pagerank = _graph.add_pagerank_scores(
        network,
        source_dict,
        target_dict,
        personalize_for='target',
    )

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

        assert (
            network.nodes[node] ==
            pytest.approx(network_with_pagerank.nodes[node])
        )

    for edge in network.edges:

        assert (
            network.edges[edge] ==
            pytest.approx(network_with_pagerank.edges[edge])
        )


def test_compute_ppr_overlap(net2):

    network, source_dict, target_dict = net2

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

    subnetwork = _graph.compute_ppr_overlap(network)

    test_network = nx.DiGraph()
    test_network.add_edge('A', 'D', weight=10)

    test_network.nodes['A']['pagerank_from_sources'] = 0.3053985948347749
    test_network.nodes['A']['pagerank_from_targets'] = 0.2806554600002494

    test_network.nodes['D']['pagerank_from_sources'] = 0.25303871980865306
    test_network.nodes['D']['pagerank_from_targets'] = 0.3527711939246324

    assert nx.is_isomorphic(test_network, subnetwork)

    for node in test_network.nodes:
        assert (
            test_network.nodes[node] ==
            pytest.approx(subnetwork.nodes[node])
        )

    for edge in test_network.edges:

        assert (
            test_network.edges[edge] ==
            pytest.approx(subnetwork.edges[edge])
        )
