import networkx as nx
from networkcommons.utils import get_subnetwork
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np


def shortest_paths(network, source_df, target_df, verbose = False):
    """
    Calculate the shortest paths between sources and targets.

    Args:
        network (nx.Graph): The network.
        source_df (pd.DataFrame): A pandas DataFrame containing the sources.
        target_df (pd.DataFrame): A pandas DataFrame containing the targets. Must contain three columns: source, target and sign
        verbose (bool): If True, print warnings when no path is found to a given target.

    Returns:
        nx.Graph: The subnetwork containing the shortest paths.
        list: A list containing the shortest paths.
    """

    shortest_paths_res = []

    sources = set(source_df['source'].values)

    for source_node in sources:
        targets = set(target_df[target_df['source'] == source_node]['target'].values)
        for target_node in targets:
            try:
                shortest_paths_res.extend([p for p in nx.all_shortest_paths(network, 
                                                                            source=source_node, 
                                                                            target=target_node, 
                                                                            weight='weight'
                                                                            )])
            except nx.NetworkXNoPath as e:
                if verbose:
                    print(f"Warning: {e}")
            except nx.NodeNotFound as e:
                if verbose:
                    print(f"Warning: {e}")

    subnetwork = get_subnetwork(network, shortest_paths_res)

    return subnetwork, shortest_paths_res


def sign_consistency(network, paths, source_df, target_df):
    """
    Calculate the sign consistency between sources and targets.

    Args:
        network (nx.Graph): The network.
        paths (list): A list containing the shortest paths.
        source_df (pd.DataFrame): A pandas DataFrame containing the sources.
        target_df (pd.DataFrame): A pandas DataFrame containing the targets. Must contain three columns: source, target and sign

    Returns:
        nx.Graph: The subnetwork containing the sign consistent paths.
        list: A list containing the sign consistent paths.
    """
    directed = nx.is_directed(network)
    subnetwork = nx.DiGraph() if directed else nx.Graph()
    sign_consistency_res = []

    for path in paths:
        source = path[0]
        product_sign = 1
        target = path[-1]

        source_sign = source_df[source_df['source'] == source]['sign'].values[0]
        target_sign = target_df[(target_df['source'] == source) & (target_df['target'] == target)]['sign'].values[0]

        for i in range(len(path) - 1):
            edge_sign = np.sign(network.get_edge_data(path[i], path[i + 1])['weight'])
            product_sign *= edge_sign

        if np.sign(source_sign * product_sign) == np.sign(target_sign):
            sign_consistency_res.append(path)
    
    subnetwork = get_subnetwork(network, sign_consistency_res)

    return subnetwork, sign_consistency_res


def reachability_filter(network, source_df):
    """
    Filters out all nodes from the graph which cannot be reached from source(s).

    Args:
        network (nx.Graph): The network.
        source_df (pd.DataFrame): A pandas DataFrame containing the source nodes.

    Returns:
        None
    """
    source_nodes = set(source_df['source'].values)
    reachable_nodes = source_nodes
    for source in source_nodes:
        reachable_nodes.update(nx.descendants(network, source))
    
    subnetwork = network.subgraph(reachable_nodes)

    return subnetwork


def all_paths(network, source_df, target_df, depth_cutoff=None, verbose=False, num_processes=None):
    """
    Calculate all paths between sources and targets.

    Args:
        cutoff (int, optional): Cutoff for path length. If None, there's no cutoff.
        verbose (bool): If True, print warnings when no path is found to a given target.

    Returns:
        list: A list containing all paths.
    """
    all_paths_res = []
    connected_all_path_targets = {}
    sources = set(source_df['source'].values)
    results = {}

    for source in sources:
        targets = set(target_df[target_df['source'] == source]['target'].values)
        try:
            results[source] = compute_all_paths(network, source, targets, depth_cutoff)
        except nx.NetworkXNoPath as e:
            if verbose:
                print(f"Warning: {e}")
        except nx.NodeNotFound as e:
            if verbose:
                print(f"Warning: {e}")

    for i, source in enumerate(sources):
        paths_for_source = results[source]
        all_paths_res.extend(paths_for_source)

    subnetwork = get_subnetwork(network, all_paths_res)

    return subnetwork, all_paths_res

def compute_all_paths(network, source, targets, cutoff):
    """
    Compute all paths between source and targets.

    Args:
        args (tuple): Tuple containing the network, source, targets and cutoff.

    Returns:
        list: A list containing all paths.
    """
    paths_for_source = []
    for target in targets:
        paths = list(nx.all_simple_paths(network, 
                                         source=source, 
                                         target=target, 
                                         cutoff=cutoff))
        paths_for_source.extend(paths)

    return paths_for_source

