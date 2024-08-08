import pytest

import pandas as pd
import networkx as nx

from networkcommons.methods import _graph

from unittest.mock import patch


def _network(weights: bool = False, signs: bool = False) -> nx.DiGraph:

    edges = pd.DataFrame(
        {
            'source': ['A', 'B', 'C', 'A', 'A', 'E', 'D'],
            'target': ['B', 'C', 'D', 'D', 'E', 'F', 'E'],
            'weight': [1, 2, 3, 6, 4, 5, 2],
            'sign': [1, 1, -1, 1, 1, -1, 1],
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


def test_run_shortest_paths_no_path_or_node_not_found():
    network = nx.DiGraph()
    network.add_edge('A', 'B', weight=1)
    network.add_edge('B', 'C', weight=2)

    # Source node exists, but target node does not exist
    source_dict = {'A': 1}
    target_dict = {'D': 1}

    subnetwork, shortest_paths_res = _graph.run_shortest_paths(
        network,
        source_dict,
        target_dict,
    )

    assert list(subnetwork.edges) == []
    assert shortest_paths_res == []

    # No path between source and target
    target_dict = {'C': 1}
    network.remove_edge('B', 'C')  # Remove the edge to create no path scenario

    subnetwork, shortest_paths_res = _graph.run_shortest_paths(
        network,
        source_dict,
        target_dict,
    )

    assert list(subnetwork.edges) == []
    assert shortest_paths_res == []


def test_run_sign_consistency_branch_false(net_signed):
    source_dict = {'A': 1}
    target_dict = {'D': 1}  # Ensure the target sign will cause the branch to be false
    paths = [['A', 'B', 'C', 'D']]  # Path that should not be added to sign_consistency_res

    subnetwork, sign_consistency_res = _graph.run_sign_consistency(
        net_signed,
        paths,
        source_dict,
        target_dict,
    )

    assert list(subnetwork.edges) == []
    assert sign_consistency_res == []


def test_run_sign_consistency_inferred_signs(net_signed):
    source_dict = {'A': 1}
    target_dict = None  # Set target_dict to None to trigger the else block lien 122
    paths = [['A', 'B', 'C', 'D']]  # Path for testing

    subnetwork, sign_consistency_res, inferred_target_sign = _graph.run_sign_consistency(
        net_signed,
        paths,
        source_dict,
        target_dict
    )

    assert list(subnetwork.edges) == [('A', 'B'), ('B', 'C'), ('C', 'D')]
    assert sign_consistency_res == [['A', 'B', 'C', 'D']]
    assert inferred_target_sign == {'D': -1}  # Check inferred sign based on path


@patch('random.choice')
def test_run_sign_consistency_ambiguous_signs(mock_random_choice, net_signed):
    source_dict = {'A': 1, 'B': 1}
    target_dict = None  # Set target_dict to None to trigger the else block
    paths = [
        ['B', 'C', 'D', 'E'],
        ['A', 'E']
    ]  # Paths for testing

    mock_random_choice.return_value = 1  # Mock random.choice to return 1
    subnetwork, sign_consistency_res, inferred_target_sign = _graph.run_sign_consistency(
        net_signed,
        paths,
        source_dict,
        target_dict,
    )

    assert list(subnetwork.edges) == [('A', 'E')]
    assert sign_consistency_res == [['A', 'E']]
    assert inferred_target_sign == {'E': 1}
    mock_random_choice.assert_called_once_with([-1, 1])


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
    print(subnetwork.edges)

    assert list(subnetwork.edges) == [('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')]


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


def test_run_all_paths_exceptions():
    # Create a network that will trigger the exceptions
    network = nx.DiGraph()
    network.add_edge('A', 'B')
    network.add_edge('B', 'C')

    # Define source and target dictionaries that will cause exceptions
    source_dict = {'A': 1}
    target_dict = {'D': 1}  # 'D' is not connected to 'A', causing NetworkXNoPath
    source_dict_not_in_graph = {'X': 1}  # 'X' is not in the graph, causing NodeNotFound

    # Check if the function handles NetworkXNoPath exception without raising it
    try:
        subnetwork, all_paths_res = _graph.run_all_paths(
            network,
            source_dict,
            target_dict,
        )
    except nx.NetworkXNoPath:
        pytest.fail("NetworkXNoPath was raised")
    except nx.NodeNotFound:
        pytest.fail("NodeNotFound was raised")

    assert all_paths_res == []
    assert list(subnetwork.edges) == []

    # Check if the function handles NodeNotFound exception without raising it
    try:
        subnetwork, all_paths_res = _graph.run_all_paths(
            network,
            source_dict_not_in_graph,
            target_dict,
        )
    except nx.NetworkXNoPath:
        pytest.fail("NetworkXNoPath was raised")
    except nx.NodeNotFound:
        pytest.fail("NodeNotFound was raised")

    assert all_paths_res == []
    assert list(subnetwork.edges) == []


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

# TODO: i don't know why in the codecov it says it's missing this branch,
        # targeted by this test (personalization is None)


def test_add_pagerank_scores_no_personalization():
    # Create a test network
    network = nx.DiGraph()
    network.add_edge('A', 'B', weight=1)
    network.add_edge('B', 'C', weight=2)
    network.add_edge('C', 'D', weight=3)
    network.add_edge('A', 'D', weight=10)
    network.add_edge('D', 'E', weight=4)
    network.add_edge('E', 'F', weight=5)

    source_dict = {'A': 1}
    target_dict = {'D': 1}

    # Run the add_pagerank_scores function without personalization
    network_with_pagerank = _graph.add_pagerank_scores(
        network,
        source_dict,
        target_dict,
        personalize_for=None,
    )

    # Check that the PageRank scores are added to the nodes
    for node in network_with_pagerank.nodes:
        assert 'pagerank' in network_with_pagerank.nodes[node]

    # Verify that the PageRank scores are correct
    expected_pagerank = nx.pagerank(network,
                                    alpha=0.85,
                                    max_iter=100,
                                    tol=1.0e-6,
                                    weight='weight')
    for node, pr_value in expected_pagerank.items():
        assert network_with_pagerank.nodes[node]['pagerank'] == pr_value


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


def test_compute_ppr_overlap_keyerror():
    # Create a test network without PageRank attributes
    network1 = nx.DiGraph()
    network1.add_edge('A', 'B', weight=1)
    network1.add_edge('B', 'C', weight=2)
    network1.add_edge('C', 'D', weight=3)
    network1.add_edge('A', 'D', weight=10)
    network1.add_edge('D', 'E', weight=4)
    network1.add_edge('E', 'F', weight=5)

    # Ensure no PageRank attributes are added
    for node in network1.nodes:
        assert 'pagerank_from_sources' not in network1.nodes[node]
        assert 'pagerank_from_targets' not in network1.nodes[node]

    # Attempt to compute PPR overlap and expect KeyError
    with pytest.raises(KeyError, match="Please run the add_pagerank_scores method first with \
                        personalization options."):
        _graph.compute_ppr_overlap(network1)

