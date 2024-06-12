from typing import Counter
import networkx as nx
import re
from networkcommons.methods import run_reachability_filter
import pandas as pd
import decoupler as dc
import numpy as np


def meta_network_cleanup(graph):
    '''
    This function cleans up a meta network graph by removing self-interactions,
    calculating the mean interaction values for duplicated source-target pairs,
    and keeping only interactions with values of 1 or -1.

    Parameters:
    - graph: A NetworkX graph.

    Returns:
    - A cleaned up meta network graph.
    '''
    # Clean up the meta network
    # Remove self-interactions
    pre_graph = graph.copy()
    pre_graph.remove_edges_from(nx.selfloop_edges(pre_graph))

    # Keep only interactions with values of 1 or -1
    post_graph = nx.DiGraph(
        [
            (u, v, d)
            for u, v, d in pre_graph.edges(data=True)
            if d['sign'] in [1, -1]
        ]
    )

    return post_graph


def prepare_metab_inputs(metab_input, compartment_codes):
    """
    Prepares the metabolite inputs by adding compartment codes.

    Args:
        metab_input (dict): A dictionary containing the metabolite names and
        their corresponding values.
        compartment_codes (list): A list of compartment codes to be added to
        the metabolite names.

    Returns:
        dict: A dictionary containing the updated metabolite names with
        compartment codes.

    """
    comps = ["r", "c", "e", "x", "m", "l", "n", "g"]

    ignored = [code for code in compartment_codes if code not in comps]
    if ignored:
        print("The following compartment codes are not found in the PKN and "
              "will be ignored:")
        print(ignored)

    compartment_codes = [code for code in compartment_codes if code in comps]

    if not compartment_codes:
        print("There are no valid compartments left. No compartment codes "
              "will be added.")
        metab_input = {
            f"Metab__{name}": value for name, value in metab_input.items()
        }

        return metab_input

    else:
        print("Adding compartment codes.")

        metab_input_list = []

        for compartment_code in compartment_codes:
            curr_metab_input = metab_input.copy()
            curr_metab_input = {
                f"{name}_{compartment_code}": value
                for name, value in curr_metab_input.items()
            }
            curr_metab_input = {
                f"Metab__{name}": value
                for name, value in curr_metab_input.items()
            }
            metab_input_list.append(curr_metab_input)

        metab_input = {
            name: value
            for curr_metab_input in metab_input_list
            for name, value in curr_metab_input.items()
        }

        return metab_input


def is_expressed(x, expressed_genes_entrez):
    """
    Determines if a gene is expressed based on the given criteria.

    Args:
        x (str): The gene name.
        expressed_genes_entrez (list): List of expressed genes.

    Returns:
        str or None: The gene name if it is expressed, otherwise None.
    """
    if not re.search("Metab", x) and not re.search("orphanReac", x):
        if x in expressed_genes_entrez:
            return x
        if re.search("Gene[0-9]+__[A-Z0-9_]+$", x):
            genes = re.sub("Gene[0-9]+__", "", x).split("_")
            if sum(gene in expressed_genes_entrez for gene in genes) != len(genes): # noqa E501
                return None
            return x
        if re.search("Gene[0-9]+__[^_][a-z]", x):
            print(x)
            return x
        if re.search("Gene[0-9]+__[A-Z0-9_]+reverse", x):
            genes = re.sub("_reverse", "", re.sub("Gene[0-9]+__", "", x)).split("_") # noqa E501
            if sum(gene in expressed_genes_entrez for gene in genes) != len(genes): # noqa E501
                return None
            return x
        return None
    return x


def filter_pkn_expressed_genes(expressed_genes_entrez, unfiltered_graph):
    """
    Filters out unexpressed nodes from the prior knowledge network (PKN).

    Args:
        expressed_genes_entrez (list): List of expressed genes in Entrez ID
        format.
        meta_pkn (nx.DiGraph): prior knowledge network (PKN) graph.

    Returns:
        nx.DiGraph: Filtered PKN graph with unexpressed nodes removed.
    """
    print("MOON: removing unexpressed nodes from PKN...")

    graph = unfiltered_graph.copy()

    nodes_to_remove = [
        node
        for node in graph.nodes
        if is_expressed(node, expressed_genes_entrez) is None
    ]

    graph.remove_nodes_from(nodes_to_remove)

    print(f"MOON: {len(nodes_to_remove)} nodes removed")

    return graph


def filter_input_nodes_not_in_pkn(data, pkn):
    """
    Filters the input nodes in the 'data' dictionary that are not present in
    the PKN.

    Args:
        data (dict): A dictionary containing the input nodes.
        pkn (nx.DiGraph): The network object representing the PKN.

    Returns:
        dict: A new dictionary containing only the input nodes that are
        present in the PKN.
    """
    new_data = {key: value for key, value in data.items() if key in pkn.nodes}

    if len(data) != len(new_data):
        removed_nodes = [
            node for node in data.keys() if node not in new_data.keys()
        ]

        print(f"COSMOS: {len(removed_nodes)} input/measured nodes are not in"
              f"PKN anymore: {removed_nodes}")

    return new_data


def keep_controllable_neighbours(source_dict, graph):
    '''
    This function filters out nodes from a dictionary of source nodes that are
    not controllable from the graph.

    Parameters:
    - source_dict: A dictionary of source nodes.
    - graph: A NetworkX graph.

    Returns:
    - A dictionary of source nodes that are observable from the graph.
    '''

    return run_reachability_filter(graph, source_dict)


def keep_observable_neighbours(target_dict, graph):
    '''
    This function filters out nodes from a dictionary of target nodes that are
    not observable from the graph.

    Parameters:
    - target_dict: A dictionary of target nodes.
    - graph: A NetworkX graph.

    Returns:
    - A dictionary of target nodes that are observable from the graph.
    '''

    subnetwork = run_reachability_filter(graph.reverse(), target_dict)

    return subnetwork.reverse()


def compress_same_children(uncompressed_graph, sig_input, metab_input):
    """
    Compresses nodes in the graph that have the same children by relabeling
    them with a common signature.

    Parameters:
    - graph (networkx.Graph): The input graph.
    - sig_input (list): List of signatures to exclude from compression.
    - metab_input (list): List of metadata signatures to exclude from
    compression.

    Returns:
    - tuple: A tuple containing the compressed subnetwork, node signatures,
    and duplicated parents.
    """
    graph = uncompressed_graph.copy()

    parents = [node for node in graph.nodes if graph.out_degree(node) > 0]
    parents.sort()

    df_signature = nx.to_pandas_edgelist(graph)
    df_signature = df_signature.sort_values(by=['source', 'target'])

    df_signature['target'] = df_signature['target'] + df_signature['sign'].astype(str) # noqa E501

    # Create a dictionary to map each parent to its targets
    parent_to_targets = df_signature.groupby('source')['target'].apply(lambda targets: '_____'.join(targets))

    # Generate the node signatures
    node_signatures = {parent: 'parent_of_' + parent_to_targets[parent] for parent in parents}

    # Count the occurrences of each signature
    node_signatures = {parent: signature for parent, signature in node_signatures.items()
                       if parent not in sig_input and
                       parent not in metab_input}

    signature_counts = Counter(node_signatures.values())

    # Identify duplicated signatures that are not in metab_input or sig_input
    duplicated_parents = {
        node: signature for node, signature in node_signatures.items()
        if signature_counts[signature] > 1
    }

    # Check for edges with different signs and exclude them from compression
    records = []
    for original_node, signature in duplicated_parents.items():
        for parent in graph.predecessors(original_node):
            sign = graph[parent][original_node]['sign']
            records.append((signature, original_node, parent, sign))

    df_records = pd.DataFrame(records, columns=['signature', 'original_node', 'parent', 'sign'])
    # Identify rows with the same signature but different signs

    signature_parent_signs = df_records.groupby(['signature', 'parent'])['sign'].nunique() > 1

    # Filter out the rows where signature-parent pairs have conflicting signs
    conflicting_pairs = signature_parent_signs[signature_parent_signs].index

    excluded_nodes = []
    for signature, parent in conflicting_pairs:
        excluded_nodes.append(df_records[(df_records['signature'] == signature) & (df_records['parent'] == parent)].original_node.values[0])
        df_records = df_records[~((df_records['signature'] == signature) & (df_records['parent'] == parent))]

    df_records = df_records[~df_records['original_node'].isin(excluded_nodes)]

    # Build new duplicated_signatures_dict
    new_duplicated_parents = {row['original_node']: row['signature'] for _, row in df_records.iterrows()}

    # Relabel the nodes in the graph based on the new duplicated signatures
    subnetwork = nx.relabel_nodes(graph, new_duplicated_parents, copy=False).copy()

    return subnetwork, node_signatures, duplicated_parents


def run_moon_core(
        upstream_input=None,
        downstream_input=None,
        graph=None,
        n_layers=None,
        n_perm=1000,
        downstream_cutoff=0,
        statistic="ulm"
):
    """
    Runs the MOON algorithm to iteratively infer MOON scores from downstream
    nodes.

    Args:
        upstream_input (dict, optional): Dictionary containing upstream input
        data. Defaults to None.
        downstream_input (dict): Dictionary containing downstream input data.
        meta_network (networkx.DiGraph): Graph representing the regulatory
        network.
        n_layers (int): Number of layers to run the MOON algorithm.
        n_perm (int): Number of permutations for statistical testing. Defaults
        to 1000.
        downstream_cutoff (float): Cutoff value for downstream input scores.
        Defaults to 0.
        statistic (str): Statistic to use for scoring. Can be "ulm"
        (univariate linear model) or "wmean" (weighted mean). Defaults to ulm.

    Returns:
        pandas.DataFrame: DataFrame containing the decoupled regulatory
        network.
    """
    regulons = nx.to_pandas_edgelist(graph)
    regulons = regulons[~regulons["source"].isin(downstream_input.keys())]

    decoupler_mat = pd.DataFrame(
        list(downstream_input.values()), index=downstream_input.keys()
    ).T

    if "wmean" in statistic:
        estimate, norm, corr, pvals = dc.run_wmean(
            mat=decoupler_mat, net=regulons, times=n_perm, weight='sign', min_n=1
        )
        if statistic == "norm_wmean":
            estimate = norm
    elif statistic == "ulm":
        estimate, pvals = dc.run_ulm(
            mat=decoupler_mat, net=regulons, weight='sign', min_n=1
        )

    n_plus_one = estimate.T
    n_plus_one.columns = ["score"]
    n_plus_one["level"] = 1

    res_list = [n_plus_one]
    i = 1
    while len(regulons) > 1 and \
            regulons["target"].isin(res_list[i - 1].index.values).sum() > 1 \
            and i < n_layers:
        print(f"Iteration count: {i}")
        regulons = regulons[~regulons["source"].isin(res_list[i - 1].index.values)] # noqa E501
        previous_n_plus_one = res_list[i - 1].drop(columns="level").T

        if "wmean" in statistic:
            estimate, norm, corr, pvals = dc.run_wmean(
                mat=previous_n_plus_one,
                net=regulons,
                times=n_perm,
                weight='sign',
                min_n=1
            )
            if statistic == "norm_wmean":
                estimate = norm
        elif statistic == "ulm":
            estimate, pvals = dc.run_ulm(
                mat=previous_n_plus_one,
                net=regulons,
                weight='sign',
                min_n=1
            )

        n_plus_one = estimate.T
        regulons = regulons[~regulons["source"].isin(n_plus_one.index.values)]
        n_plus_one["level"] = i + 1
        res_list.append(n_plus_one)
        i += 1

    recursive_decoupleRnival_res = pd.concat(res_list)

    downstream_names = pd.DataFrame.from_dict(
        downstream_input, orient="index", columns=["score"]
    )
    downstream_names = downstream_names[
        abs(downstream_names["score"]) > downstream_cutoff
    ]
    downstream_names["level"] = 0

    recursive_decoupleRnival_res = pd.concat(
        [recursive_decoupleRnival_res, downstream_names]
    )

    if upstream_input is not None:
        upstream_input_df = pd.DataFrame.from_dict(
            upstream_input, orient="index", columns=["real_score"]
        )
        upstream_input_df = upstream_input_df.join(
            recursive_decoupleRnival_res, how='right'
        )
        upstream_input_df = upstream_input_df[
            (np.sign(upstream_input_df["real_score"]) ==
             np.sign(upstream_input_df["score"])) |
            (np.isnan(upstream_input_df["real_score"]))
        ]
        recursive_decoupleRnival_res = upstream_input_df.drop(
            columns="real_score"
        )

    recursive_decoupleRnival_res.reset_index(inplace=True)
    recursive_decoupleRnival_res.rename(columns={"index": "source"}, inplace=True) # noqa E501

    return recursive_decoupleRnival_res


def filter_incoherent_TF_target(
        moon_res, TF_reg_net, meta_network, RNA_input
):
    """
    Filters incoherent TF-target interactions from the meta_network based on
    the given inputs.

    Parameters:
    moon_res (pd.DataFrame): DataFrame
    TF_reg_net (pd.DataFrame): DataFrame containing TF regulatory network.
    meta_network (networkx.Graph): Graph representing the meta network.
    RNA_input (dict): Dictionary containing RNA input values.

    Returns:
    networkx.Graph: Filtered meta network with incoherent TF-target
    interactions removed.
    """
    filtered_meta_network = meta_network.copy()

    RNA_df = pd.DataFrame.from_dict(
        RNA_input, orient='index', columns=['RNA_input']
    )
    reg_meta = pd.merge(
        moon_res, TF_reg_net, left_on='source', right_on='source', how='inner'
    )
    reg_meta.rename(columns={'score': 'TF_score'}, inplace=True)

    reg_meta = pd.merge(
        reg_meta, RNA_df, left_on='target', right_index=True, how='inner'
    )
    reg_meta['incoherent'] = np.sign(
        reg_meta['TF_score'] * reg_meta['RNA_input'] * reg_meta['weight']
    ) < 0

    reg_meta = reg_meta[reg_meta["incoherent"]][['source', 'target']]

    tuple_list = list(reg_meta.itertuples(index=False, name=None))

    filtered_meta_network.remove_edges_from(tuple_list)

    return filtered_meta_network


def decompress_moon_result(
        moon_res, node_signatures, duplicated_parents, meta_network_graph
):
    """
    Decompresses the moon_res dataframe by mapping the compressed nodes to
    their corresponding original source using the provided
    meta_network_compressed_list and the filtered meta_network.

    Args:
        moon_res (pandas.DataFrame): The compressed moon_res dataframe.
        node_signatures (dict): The node signatures dictionary.
        duplicated_parents (dict): The duplicated parents dictionary.
        meta_network_graph (nx.DiGraph): The compressed meta_network.

    Returns:
        pandas.DataFrame: The decompressed moon_res dataframe with the source
        column mapped to its corresponding original source.
    """
    compressed_meta_network = nx.to_pandas_edgelist(meta_network_graph)

    # Create a dataframe for duplicated parents
    duplicated_parents_df = pd.DataFrame.from_dict(
        duplicated_parents, orient='index', columns=['source']
    )
    duplicated_parents_df['source_original'] = duplicated_parents_df.index

    # Create a dataframe for addons
    addons = pd.DataFrame(
        list(node_signatures.keys() - duplicated_parents_df['source_original'])
    )
    addons.columns = ['source']
    addons['source_original'] = addons['source']
    addons.sort_values(by='source', inplace=True)

    # Get final leaves (nodes with no outgoing edges)
    final_leaves = compressed_meta_network[
        ~compressed_meta_network['target'].isin(
            compressed_meta_network['source']
        )
    ]['target']
    final_leaves = pd.DataFrame(
        {'source': final_leaves, 'source_original': final_leaves}
    )

    # Combine addons and final leaves
    addons = pd.concat([addons, final_leaves])

    # Create mapping table by combining duplicated parents and addons
    mapping_table = pd.concat([duplicated_parents_df, addons])
    mapping_table = mapping_table.drop_duplicates()

    # Merge the moon_res dataframe with the mapping table
    moon_res = pd.merge(moon_res, mapping_table, on='source', how='inner')

    # Return the merged dataframe
    return moon_res


def reduce_solution_network(
        moon_res, meta_network, cutoff, sig_input, rna_input=None
):
    """
    Reduces the solution network based on MOON score cutoffs and returns the
    reduced network and attribute table.

    Args:
        moon_res (pandas.DataFrame): The solution MOON df.
        meta_network (networkx.DiGraph): The original network.
        cutoff (float): The cutoff value for filtering edges.
        sig_input (dict): Dictionary containing the significant input scores.
        rna_input (dict, optional): Dictionary containing the RNA input scores.
        Defaults to None.

    Returns:
        res_network (networkx.DiGraph): The reduced network.
        att (pandas.DataFrame): The attribute table containing the relevant
        attributes of the nodes in the reduced network.
    """
    recursive_moon_res = moon_res.copy()

    recursive_moon_res = recursive_moon_res[
        abs(recursive_moon_res['score']) > cutoff
    ]
    consistency_vec = recursive_moon_res.set_index(
        'source_original')['score'].to_dict()

    res_network = meta_network.subgraph(
        [
            node for node in meta_network.nodes if node in
            recursive_moon_res.source_original.values
        ]
    )
    res_network_edges = res_network.edges(data=True)
    res_network = nx.DiGraph(res_network)
    for source, target, data in res_network_edges:
        if data['sign'] != np.sign(consistency_vec[source] * consistency_vec[target]): # noqa E501
            res_network.remove_edge(source, target)

    recursive_moon_res.rename(columns={'source_original': 'nodes'},
                              inplace=True)
    recursive_moon_res.drop(columns=['source'], inplace=True)

    sig_input_df = pd.DataFrame.from_dict(
        sig_input, orient='index', columns=['real_score']
    )
    merged_df = pd.merge(
        sig_input_df, recursive_moon_res, how='inner',
        left_index=True, right_on='nodes'
    )
    merged_df['filterout'] = np.sign(
        merged_df['real_score']) != np.sign(
            merged_df['score'])
    merged_df = merged_df[~ merged_df['filterout']]
    upstream_nodes = merged_df.nodes.values
    upstream_nodes = {
        node: 1 for node in upstream_nodes if node in res_network.nodes
    }

    res_network = keep_controllable_neighbours(upstream_nodes, res_network)

    moon_scores = recursive_moon_res.set_index(
        'nodes')['score'].to_dict()
    nx.set_node_attributes(G=res_network, values=moon_scores, name='moon_score')

    att = recursive_moon_res[
        recursive_moon_res['nodes'].isin(res_network.nodes)
    ]

    if rna_input is not None:
        rna_input_df = pd.DataFrame.from_dict(
            rna_input, orient='index', columns=['real_score']
        ).reset_index().rename(columns={'index': 'nodes'})
        att = pd.merge(att, rna_input_df, how='left', on='nodes')
    else:
        att['RNA_input'] = np.nan

    return res_network, att


def translate_res(network, att, mapping_dict):
    """
    Translates the network and attribute table based on the given mapping
    dataframe.

    Args:
        network (networkx.DiGraph): The network to be translated.
        att (pandas.DataFrame): The attribute table to be translated.
        mapping_df (pandas.DataFrame): The mapping dataframe containing the
        translation information.

    Returns:
        network (networkx.DiGraph): The translated network.
        att (pandas.DataFrame): The translated attribute table.
    """
    to_rename = att.nodes.values
    renamed = {}

    for name in to_rename:
        name_changed = re.sub("Metab__", "", name)
        name_changed = re.sub("^Gene", "Enzyme", name_changed)
        suffix = re.search("_[a-z]$", name_changed)
        name_changed = re.sub("_[a-z]$", "", name_changed)
        if name_changed in mapping_dict:
            name_changed = mapping_dict[name_changed]
            name_changed = "Metab__" + name_changed + suffix.group() \
                if suffix else "Metab__" + name_changed
        renamed[name] = name_changed

    att['nodes'] = att['nodes'].map(renamed)

    network = nx.relabel_nodes(network, renamed, copy=True)

    return network, att
