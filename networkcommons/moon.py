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
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Keep only interactions with values of 1 or -1
    graph = nx.DiGraph(
        [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if d['sign'] in [1, -1]
        ]
    )

    return graph


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
        else:
            if re.search("Gene[0-9]+__[A-Z0-9_]+$", x):
                genes = re.sub("Gene[0-9]+__", "", x).split("_")
                if sum(gene in expressed_genes_entrez for gene in genes) != len(genes):
                    return None
                else:
                    return x
            else:
                if re.search("Gene[0-9]+__[^_][a-z]", x):
                    print(x)
                    return x
                else:
                    if re.search("Gene[0-9]+__[A-Z0-9_]+reverse", x):
                        genes = re.sub("_reverse", "", re.sub("Gene[0-9]+__", "", x)).split("_")
                        if sum(gene in expressed_genes_entrez for gene in genes) != len(genes):
                            return None
                        else:
                            return x
                    else:
                        return None
    else:
        return x


def filter_pkn_expressed_genes(expressed_genes_entrez, graph):
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
              "PKN anymore: {removed_nodes}")

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


def compress_same_children(graph, sig_input, metab_input):
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

    parents = [node for node in graph.nodes if graph.out_degree(node) > 0]

    df_signature = pd.DataFrame(graph.edges, columns=['source', 'target'])
    sign_dict = nx.get_edge_attributes(graph, 'sign')
    df_signature['sign'] = df_signature.apply(lambda row: sign_dict.get((row['source'], row['target'])), axis=1)

    df_signature['target'] = df_signature['target'] + df_signature['sign'].astype(str)

    node_signatures = {}
    for parent in parents:
        node_signatures[parent] = 'parent_of_' + '_____'.join(df_signature[df_signature['source'] == parent]['target'])

    dubs = [signature for signature, count in Counter(node_signatures.values()).items() if count > 1 and signature not in metab_input and signature not in sig_input]
    duplicated_parents = {node: signature for node, signature in node_signatures.items() if signature in dubs}

    subnetwork = nx.relabel_nodes(graph, duplicated_parents, copy=False)

    return subnetwork, node_signatures, duplicated_parents


def run_moon_core(upstream_input=None,
                  downstream_input=None,
                  meta_network=None,
                  n_layers=None,
                  n_perm=1000,
                  downstream_cutoff=0,
                  statistic="ulm"):
    """
    Runs the MOON algorithm to iteratively infer MOON scores from downstream
    nodes.

    Args:
        upstream_input (dict, optional): Dictionary containing upstream input
        data. Defaults to None.
        downstream_input (dict): Dictionary containing downstream input data.
        meta_network (networkx.Graph): Graph representing the regulatory
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

    regulons = nx.to_pandas_edgelist(meta_network)
    regulons.rename(columns={"sign": "mor"}, inplace=True)
    regulons = regulons[~regulons["source"].isin(downstream_input.keys())]

    decoupler_mat = pd.DataFrame(
        list(downstream_input.values()),
        index=downstream_input.keys()).T

    if "wmean" in statistic:
        estimate, norm, corr, pvals = dc.run_wmean(
            mat=decoupler_mat,
            net=regulons,
            times=n_perm,
            weight=None,
            min_n=1
        )
        if statistic == "norm_wmean":
            estimate = norm
    elif statistic == "ulm":
        print(decoupler_mat)
        estimate, pvals = dc.run_ulm(
            mat=decoupler_mat,
            net=regulons,
            weight=None,
            min_n=1
        )

    n_plus_one = estimate.T
    n_plus_one.columns = ["score"]
    n_plus_one["level"] = 1

    res_list = [n_plus_one]
    i = 1
    while len(regulons) > 1 and \
            regulons["target"].isin(res_list[i - 1].index.values).sum() > 1 and \
            i <= n_layers:

        regulons = regulons[~regulons["source"].isin(res_list[i - 1].index.values)]
        previous_n_plus_one = res_list[i - 1].drop(columns="level").T

        if "wmean" in statistic:
            estimate, norm, corr, pvals = dc.run_wmean(
                mat=previous_n_plus_one,
                net=regulons,
                times=n_perm,
                weight=None,
                min_n=1
            )
            if statistic == "norm_wmean":
                estimate = norm
        elif statistic == "ulm":
            estimate, pvals = dc.run_ulm(
                mat=previous_n_plus_one,
                net=regulons,
                weight=None,
                min_n=1
            )

        n_plus_one = estimate.T
        regulons = regulons[~regulons["source"].isin(n_plus_one.index.values)]
        n_plus_one["level"] = i + 1
        res_list.append(n_plus_one)
        i += 1

    recursive_decoupleRnival_res = pd.concat(res_list)

    downstream_names = pd.DataFrame.from_dict(downstream_input, orient="index", columns=["score"])
    downstream_names = downstream_names[abs(downstream_names["score"]) > downstream_cutoff]
    downstream_names["level"] = 0

    recursive_decoupleRnival_res = pd.concat([recursive_decoupleRnival_res, downstream_names])

    if upstream_input is not None:
        upstream_input_df = pd.DataFrame.from_dict(upstream_input, orient="index", columns=["real_score"])
        upstream_input_df = upstream_input_df.join(recursive_decoupleRnival_res, how='right')
        upstream_input_df = upstream_input_df[(np.sign(upstream_input_df["real_score"]) == np.sign(upstream_input_df["score"])) | (np.isnan(upstream_input_df["real_score"]))]
        recursive_decoupleRnival_res = upstream_input_df.drop(columns="real_score")


    return recursive_decoupleRnival_res


def filter_incoherent_TF_target(decoupleRnival_res,
                               TF_reg_net,
                               meta_network,
                               RNA_input):
    """
    Filters incoherent TF-target interactions from the meta_network based on
    the given inputs.

    Parameters:
    decoupleRnival_res (pd.DataFrame): DataFrame
    TF_reg_net (pd.DataFrame): DataFrame containing TF regulatory network.
    meta_network (networkx.Graph): Graph representing the meta network.
    RNA_input (dict): Dictionary containing RNA input values.

    Returns:
    networkx.Graph: Filtered meta network with incoherent TF-target
    interactions removed.
    """

    TF_reg_net.set_index('source', inplace=True, drop=True)
    RNA_df = pd.DataFrame.from_dict(RNA_input, orient='index', columns=['RNA_input'])

    reg_meta = decoupleRnival_res[decoupleRnival_res.index.isin(TF_reg_net.index)]
    reg_meta = reg_meta.join(TF_reg_net)
    reg_meta.rename(columns={'score': 'TF_score'}, inplace=True)

    reg_meta = pd.merge(reg_meta, RNA_df, left_on='target', right_index=True)
    reg_meta['incoherent'] = np.sign(reg_meta['TF_score'] * reg_meta['RNA_input'] * reg_meta['weight']) < 0

    reg_meta = reg_meta[reg_meta["incoherent"]==True][['target']]

    to_tuple_list = reg_meta.rename_axis("source").reset_index()
    tuple_list = list(to_tuple_list.itertuples(index=False, name=None))

    meta_network.remove_edges_from(tuple_list)

    return meta_network


def decompress_moon_result(moon_res, meta_network_compressed_list, meta_network_graph):
    """
    Decompresses the moon_res dataframe by mapping the compressed nodes to
    their corresponding 
    original source using the provided meta_network_compressed_list and
    meta_network.

    Args:
        moon_res (pandas.DataFrame): The compressed moon_res dataframe.
        meta_network_compressed_list (dict): The compressed
        meta_network_compressed_list containing node_signatures and
        duplicated_parents.
        meta_network (pandas.DataFrame): The original meta_network dataframe.

    Returns:
        pandas.DataFrame: The decompressed moon_res dataframe with the source
        column mapped to its corresponding original source.
    """
    meta_network = nx.to_pandas_edgelist(meta_network_graph)


    # Extract node_signatures and duplicated_parents from the list
    node_signatures = meta_network_compressed_list['node_signatures']
    duplicated_parents = meta_network_compressed_list['duplicated_signatures']

    # Create a dataframe for duplicated parents
    duplicated_parents_df = pd.DataFrame.from_dict(duplicated_parents, orient='index', columns=['source'])
    duplicated_parents_df['source_original'] = duplicated_parents_df.index

    # Create a dataframe for addons
    addons = pd.DataFrame(list(node_signatures.keys() - duplicated_parents_df['source_original']))
    addons.columns = ['source']
    addons['source_original'] = addons['source']

    # Get final leaves
    final_leaves = meta_network[~meta_network['target'].isin(meta_network['source'])]['target']
    final_leaves = pd.DataFrame({'source': final_leaves, 'source_original': final_leaves})

    # Combine addons and final leaves
    addons = pd.concat([addons, final_leaves])

    # Create mapping table by combining duplicated parents and addons
    mapping_table = pd.concat([duplicated_parents_df, addons])
    mapping_table = mapping_table.drop_duplicates()

    # Merge the moon_res dataframe with the mapping table
    moon_res = pd.merge(moon_res, mapping_table, on='source')

    # Add node score to the graph
    for node in meta_network.nodes:
        if node in moon_res.index:
            meta_network.nodes[node]["score"] = moon_res.loc[node, "score"]

    # Return the merged dataframe
    return moon_res


def reduce_solution_network(decoupleRnival_res, meta_network, cutoff, sig_input, RNA_input=None):
    recursive_decoupleRnival_res = decoupleRnival_res.copy()
    
    recursive_decoupleRnival_res = recursive_decoupleRnival_res[abs(recursive_decoupleRnival_res['score']) > cutoff]
    consistency_vec = recursive_decoupleRnival_res['score']
    
    res_network = meta_network.subgraph([node for node in meta_network.nodes if node in recursive_decoupleRnival_res.source.values])
    res_network_edges = res_network.edges(data=True)
    res_network = nx.DiGraph()
    for source, target, data in res_network_edges:
        if data['sign'] == np.sign(consistency_vec[source] * consistency_vec[target]):
            res_network.add_edge(source, target, interaction=data['sign'])
    
    recursive_decoupleRnival_res.columns = ['nodes'] + list(recursive_decoupleRnival_res.columns)[1:]
    
    res_network_edges = res_network.edges(data=True)
    res_network = nx.DiGraph()
    for source, target, data in res_network_edges:
        res_network.add_edge(source, target, interaction=data['interaction'])

    sig_input_df = pd.DataFrame.from_dict(sig_input, orient='index', columns=['real_score'])
    merged_df = sig_input_df.join(recursive_decoupleRnival_res, how='left')
    merged_df['filterout'] = np.sign(merged_df['real_score']) != np.sign(merged_df['score'])
    merged_df = merged_df[merged_df['filterout'] == False]
    upstream_nodes = merged_df.index.values
    upstream_nodes = {node: 1 for node in upstream_nodes if node in res_network.nodes}

    res_network = keep_controllable_neighbours(upstream_nodes, res_network)

    return res_network

### formats the network to be human-readable. test whole framework with the data to understand 
#   SIF <- res_network
#   ATT <- recursive_decoupleRnival_res[recursive_decoupleRnival_res$nodes %in% SIF$source | recursive_decoupleRnival_res$nodes %in% SIF$target,]
  
#   if(!is.null(RNA_input))
#   {
#     RNA_df <- data.frame(RNA_input)
#     RNA_df$nodes <- row.names(RNA_df)
    
#     ATT <- merge(ATT, RNA_df, all.x = T)
#   } else
#   {
#     ATT$RNA_input <- NA
#   }
#   return(list("SIF" = SIF, "ATT" = ATT))
# }
###







upstream_input = {"A": 1, "B": -1, "C": 0.5}
downstream_input = {"D": 2, "E": -1.5}
meta_network_edges = [
    ("A", "B", 1),
    ("A", "C", -1),
    ("B", "D", -1),
    ("C", "E", 1),
    ("C", "D", -1),
    ("D", "B", -1),
    ("E", "A", 1)
]

graph = nx.DiGraph()
graph.add_weighted_edges_from(meta_network_edges)

RNA_input = {
    "A": 1,
    "B": -1,
    "C": 5,
    "D": -0.7,
    "E": -0.3
}

RNA_input = {
    "A": 1,
    "B": -1,
    "C": 5,
    "D": -0.7,
    "E": -0.3
}

TF_reg_net = pd.DataFrame({
    "source": ["B"],
    "target": ["D"],
    "mor": [-1]
})

