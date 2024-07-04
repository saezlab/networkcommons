import networkx as nx
import pandas as pd
from networkcommons.methods import _moon


def test_meta_network_cleanup():

    graph = nx.DiGraph()
    graph.add_nodes_from(['A', 'B', 'C', 'D'])
    graph.add_edge('A', 'A', sign=1)  # Self-loop
    graph.add_edge('A', 'B', sign=1)
    graph.add_edge('B', 'C', sign=0)
    graph.add_edge('C', 'D', sign=1)
    graph.add_edge('D', 'C', sign=-1)

    cleaned_graph = _moon.meta_network_cleanup(graph)

    assert list(nx.nodes_with_selfloops(cleaned_graph)) == [], \
        "Self-loops are removed"
    for u, v, d in cleaned_graph.edges(data=True):
        assert d['sign'] in [1, -1], f"Unexpected sign value: {d['sign']}"
    assert len(cleaned_graph.edges) == 3, "Unexpected number of edges"


def test_prepare_metab_inputs():

    metab_input = {'glucose': 1.0, 'fructose': 2.0}
    compartment_codes = ['c', 'm', 'invalid']

    prepared_input = _moon.prepare_metab_inputs(metab_input, compartment_codes)

    assert 'Metab__glucose_c' in prepared_input
    assert 'Metab__fructose_m' in prepared_input
    assert 'Metab__glucose_invalid' not in prepared_input
    assert len(prepared_input) == 4, "Unexpected number of metabolite inputs"


def test_is_expressed():

    expressed_genes_entrez = ["GENE1", "GENE2", "GENE3"]

    # Test case for matching "Gene[0-9]+__[A-Z0-9_]+$"
    assert _moon.is_expressed("Gene123__GENE1", expressed_genes_entrez) == "Gene123__GENE1", "Test case 1 failed"
    assert _moon.is_expressed("Gene123__GENE4", expressed_genes_entrez) is None, "Test case 2 failed"

    # Test case for matching "Gene[0-9]+__[^_][a-z]"
    assert _moon.is_expressed("Gene456__Aa", expressed_genes_entrez) == "Gene456__Aa", "Test case 3 failed"
    assert _moon.is_expressed("Gene789__Bb", expressed_genes_entrez) == "Gene789__Bb", "Test case 4 failed"
    assert _moon.is_expressed("Gene456__Ab", expressed_genes_entrez) == "Gene456__Ab", "Test case 5 failed"
    assert _moon.is_expressed("Gene789__Cz", expressed_genes_entrez) == "Gene789__Cz", "Test case 6 failed"

    # Test case for matching "Gene[0-9]+__[A-Z0-9_]+reverse"
    expressed_genes_entrez = ["GENE1", "GENE2"]
    assert _moon.is_expressed("Gene111__GENE1_GENE2_reverse", expressed_genes_entrez) == "Gene111__GENE1_GENE2_reverse", "Test case 7 failed"
    assert _moon.is_expressed("Gene222__GENE1_GENE4_reverse", expressed_genes_entrez) is None, "Test case 8 failed"

    # Test case for matching exact gene names in expressed_genes_entrez
    expressed_genes_entrez = ["Gene1", "Gene2", "Gene3"]
    assert _moon.is_expressed("Gene1", expressed_genes_entrez) == "Gene1", "Test case 9 failed"
    assert _moon.is_expressed("Gene4", expressed_genes_entrez) is None, "Test case 10 failed"

    # Test case for non-Metab and non-orphanReac strings that don't match any regex
    assert _moon.is_expressed("RandomGene", expressed_genes_entrez) is None, "Test case 11 failed"

    # Test case for "Metab" and "orphanReac" strings which should return the input itself
    assert _moon.is_expressed("Metab1", expressed_genes_entrez) == "Metab1", "Test case 12 failed"
    assert _moon.is_expressed("orphanReac1", expressed_genes_entrez) == "orphanReac1", "Test case 13 failed"


def test_filter_pkn_expressed_genes():

    expressed_genes = ['Gene1', 'Gene2', 'Gene3']
    graph = nx.DiGraph()
    graph.add_nodes_from(['Gene1', 'Gene2', 'Gene4'])
    graph.add_edge('Gene1', 'Gene2')
    graph.add_edge('Gene2', 'Gene4')

    filtered_graph = _moon.filter_pkn_expressed_genes(expressed_genes, graph)

    assert 'Gene4' not in filtered_graph.nodes, "Unexpressed node not removed"
    assert len(filtered_graph.nodes) == 2, "Unexpected number of nodes"


def test_filter_input_nodes_not_in_pkn():

    data = {'Gene1': 1, 'Gene2': 2, 'Gene3': 3}
    graph = nx.DiGraph()
    graph.add_nodes_from(['Gene1', 'Gene2'])

    filtered_data = _moon.filter_input_nodes_not_in_pkn(data, graph)

    assert 'Gene3' not in filtered_data, "Node not in PKN not removed"
    assert len(filtered_data) == 2, "Unexpected number of input nodes"


def test_keep_controllable_neighbours():

    source_dict = {'Gene1': 1, 'Gene2': 1}
    graph = nx.DiGraph()
    graph.add_edges_from([('Gene1', 'Gene3'),
                          ('Gene2', 'Gene4'),
                          ('Gene0', 'Gene1')])

    filtered_sources = _moon.keep_controllable_neighbours(source_dict, graph)

    assert 'Gene1' in filtered_sources
    assert 'Gene2' in filtered_sources
    assert 'Gene3' in filtered_sources
    assert 'Gene4' in filtered_sources
    assert 'Gene0' not in filtered_sources


def test_keep_observable_neighbours():

    target_dict = {'Gene3': 1, 'Gene4': 1}
    graph = nx.DiGraph()
    graph.add_edges_from([('Gene1', 'Gene3'),
                          ('Gene2', 'Gene4'),
                          ('Gene0', 'Gene1'),
                          ('Gene4', 'Gene5')])

    filtered_targets = _moon.keep_observable_neighbours(target_dict, graph)

    assert 'Gene2' in filtered_targets
    assert 'Gene1' in filtered_targets
    assert 'Gene0' in filtered_targets
    assert 'Gene4' in filtered_targets
    assert 'Gene5' not in filtered_targets


def test_compress_same_children():

    graph = nx.DiGraph()
    graph.add_edges_from([
        ('A', 'B', {'sign': 1}),
        ('A', 'C', {'sign': 1}),
        ('B', 'D', {'sign': 1}),
        ('C', 'D', {'sign': 1}),
    ])
    sig_input = []
    metab_input = []

    (
        subnetwork,
        node_signatures,
        duplicated_parents,
    ) = _moon.compress_same_children(graph, sig_input, metab_input)

    assert len(subnetwork.nodes) == 3, "Unexpected number of nodes in subnetwork" # noqa E501
    assert len(node_signatures) == 3, "Unexpected number of node signatures" # noqa E501
    assert len(duplicated_parents) == 2, "Unexpected number of duplicated parents" # noqa E501
    assert duplicated_parents['B'] == node_signatures['B'], "Duplicated parents mismatch" # noqa E501

    graph = nx.DiGraph()
    graph.add_edges_from([
        ('A', 'B', {'sign': 1}),
        ('A', 'C', {'sign': -1}),
        ('B', 'D', {'sign': 1}),
        ('C', 'D', {'sign': -1}),
    ])
    sig_input = []
    metab_input = []
    (
        subnetwork,
        node_signatures,
        duplicated_parents
    ) = _moon.compress_same_children(
        graph, sig_input, metab_input
    )

    assert len(subnetwork.nodes) == 4, "Unexpected number of nodes in subnetwork" # noqa E501
    assert len(node_signatures) == 3, "Unexpected number of node signatures" # noqa E501
    assert len(duplicated_parents) == 0, "Duplicated parents mismatch" # noqa E501


def test_run_moon_core():

    graph = nx.DiGraph()
    graph.add_edges_from([
        ('A', 'B', {'sign': 1}),
        ('B', 'C', {'sign': 1}),
        ('B', 'D', {'sign': 1}),
        ('D', 'E', {'sign': 1}),
        ('E', 'F', {'sign': -1}),
    ])
    upstream_input = {'A': 1}
    downstream_input = {'E': 0.5, 'F': -2}

    result = _moon.run_moon_core(
        upstream_input=upstream_input,
        downstream_input=downstream_input,
        graph=graph,
        n_layers=5,
        statistic='wmean'
    )

    assert 'score' in result.columns, "Score column missing in result"
    assert 'source' in result.columns, "Source column missing in result"
    assert len(result.index) == 3, "Unexpected number of rows in result"
    assert result.empty is False, "Empty result"


def test_filter_incoherent_TF_target():

    moon_res = pd.DataFrame({'source': ['TF1'], 'score': [1]})
    TF_reg_net = pd.DataFrame({'source': ['TF1', 'TF1'],
                               'target': ['Gene1', 'Gene2'],
                               'weight': [1, -1]})
    meta_network = nx.DiGraph()
    meta_network.add_edge('TF1', 'Gene1')
    meta_network.add_edge('TF1', 'Gene2')
    RNA_input = {'Gene1': -1, 'Gene2': -1}

    filtered_network = _moon.filter_incoherent_TF_target(
        moon_res, TF_reg_net, meta_network, RNA_input
    )

    assert len(filtered_network.edges) == 1, "Incoherent edge not removed"


def test_decompress_moon_result():

    moon_res = pd.DataFrame({'source': ['A'], 'score': [1]})
    node_signatures = {'A': 'A'}
    duplicated_parents = {}
    meta_network_graph = nx.DiGraph()
    meta_network_graph.add_edge('A', 'B')

    decompressed_result = _moon.decompress_moon_result(
        moon_res, node_signatures, duplicated_parents, meta_network_graph
    )

    assert 'source_original' in decompressed_result.columns, \
        "Missing 'source_original' in result"


def test_reduce_solution_network():

    moon_res = pd.DataFrame({'source_original': ['A', 'B', 'C'],
                             'source': ['Parent_of_C', 'Parent_of_C', 'C'],
                             'score': [1.4783, -1.54, 0.5]})
    meta_network = nx.DiGraph()
    meta_network.add_edges_from([
        ('A', 'B', {'sign': -1}),
        ('B', 'C', {'sign': -1}),
    ])
    sig_input = {'A': 1}

    res_network, att = _moon.reduce_solution_network(
        moon_res, meta_network, cutoff=1, sig_input=sig_input
    )

    assert len(nx.get_node_attributes(res_network, 'moon_score')) != 0, \
        "Missing moon_score attribute in network"
    assert 'score' in att.columns, "Missing score column in attributes"
    print(res_network.nodes)
    assert len(res_network.nodes) == 2, "Unexpected number of nodes"


def test_get_ego_graph():

    # Create a sample directed graph
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "G")])

    # Define the sources and depth limit
    sources = ["A"]
    depth_limit = 2

    # Call the function
    ego_graph = _moon.get_ego_graph(G, sources, depth_limit)

    # Define expected nodes and edges in the ego graph
    expected_nodes = {"A", "B", "C", "D", "E"}
    expected_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")]

    # Perform assertions
    assert set(ego_graph.nodes()) == expected_nodes, "Ego graph nodes are incorrect"
    assert set(ego_graph.edges()) == set(expected_edges), "Ego graph edges are incorrect"

    # Test with a different source and depth limit
    sources = ["B"]
    depth_limit = 1
    ego_graph = _moon.get_ego_graph(G, sources, depth_limit)

    # Define expected nodes and edges in the ego graph for new source and depth
    expected_nodes = {"B", "D"}
    expected_edges = [("B", "D")]

    # Perform assertions
    assert set(ego_graph.nodes()) == expected_nodes, "Ego graph nodes are incorrect for source 'B'"
    assert set(ego_graph.edges()) == set(expected_edges), "Ego graph edges are incorrect for source 'B'"


def test_translate_res():

    G = nx.DiGraph()
    G.add_edges_from([
        ("Metab__HMDB1_a", "Metab__HMDB2_b"),
        ("Metab__HMDB3_b", "GeneC_c")
    ])

    att_data = {
        "nodes": [
            "Metab__HMDB1_a",
            "Metab__HMDB2_b",
            "Metab__HMDB3_b",
            "GeneC_c"
        ],
        "score": [1, 2, 3, 4]
    }
    att_df = pd.DataFrame(att_data)

    mapping_dict = {
        "HMDB1": "Alpha",
        "HMDB2": "Beta",
        "HMDB3": "Gamma"
    }

    translated_network, translated_att = _moon.translate_res(
        G, att_df, mapping_dict
    )
    expected_edges = [
        ("Metab__Alpha_a", "Metab__Beta_b"),
        ("Metab__Gamma_b", "EnzymeC")
    ]
    expected_att_nodes = [
        "Metab__Alpha_a",
        "Metab__Beta_b",
        "Metab__Gamma_b",
        "EnzymeC_c"
    ]

    assert set(translated_network.edges()) == set(expected_edges), "Translated network edges are incorrect"
    assert translated_att['nodes'].tolist() == expected_att_nodes, "Translated attribute table nodes are incorrect"
    assert 'Metab__Alpha_a' in translated_network.nodes, "Translation failed"
    assert 'Metab__Alpha_a' in translated_att['nodes'].values, \
        "Translation failed in attributes"
