import networkx as nx
from networkcommons.utils import get_subnetwork
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import corneto as cn
from corneto.contrib.networkx import networkx_to_corneto_graph, corneto_graph_to_networkx
from corneto.methods.carnival import get_result, get_selected_edges

def run_shortest_paths(network, source_dict, target_dict, verbose = False):
    """
    Calculate the shortest paths between sources and targets.

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign of perturbation.
        target_dict (dict): A dictionary containing the targets and sign of measurements.
            Must contain three columns: source, target and sign
        verbose (bool): If True, print warnings when no path is found to a given target.

    Returns:
        nx.Graph: The subnetwork containing the shortest paths.
        list: A list containing the shortest paths.
    """

    shortest_paths_res = []

    sources = source_dict.keys()

    for source_node in sources:
        targets = target_dict[source_node].keys()
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


def run_sign_consistency(network, paths, source_dict, target_dict):
    """
    Calculate the sign consistency between sources and targets.

    Args:
        network (nx.Graph): The network.
        paths (list): A list containing the shortest paths.
        source_dict (dict): A dictionary containing the sources and sign of perturbation.
        target_dict (dict): A dictionary containing the targets and sign of measurements.

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

        source_sign = source_dict[source]
        target_sign = target_dict[source][target]

        for i in range(len(path) - 1):
            edge_sign = network.get_edge_data(path[i], path[i + 1])['sign']
            product_sign *= edge_sign

        if np.sign(source_sign * product_sign) == np.sign(target_sign):
            sign_consistency_res.append(path)
    
    subnetwork = get_subnetwork(network, sign_consistency_res)

    return subnetwork, sign_consistency_res


def run_reachability_filter(network, source_dict):
    """
    Filters out all nodes from the graph which cannot be reached from source(s).

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign of perturbation.

    Returns:
        None
    """
    source_nodes = list(source_dict.keys())
    reachable_nodes = source_nodes
    for source in source_nodes:
        reachable_nodes.update(nx.descendants(network, source))
    
    subnetwork = network.subgraph(reachable_nodes)

    return subnetwork


def run_all_paths(network, source_dict, target_dict, depth_cutoff=None, verbose=False):
    """
    Calculate all paths between sources and targets.

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign of perturbation.
        target_dict (dict): A dictionary containing the targets and sign of measurements.
        depth_cutoff (int, optional): Cutoff for path length. If None, there's no cutoff.
        verbose (bool): If True, print warnings when no path is found to a given target.

    Returns:
        list: A list containing all paths.
    """
    all_paths_res = []
    sources = list(source_dict.keys())

    for source in sources:
        targets = list(target_dict[source].keys())
        try:
            all_paths_res.extend(compute_all_paths(network, source, targets, depth_cutoff))
        except nx.NetworkXNoPath as e:
            if verbose:
                print(f"Warning: {e}")
        except nx.NodeNotFound as e:
            if verbose:
                print(f"Warning: {e}")

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


def add_pagerank_scores(network, 
                        source_dict, 
                        target_dict, 
                        alpha=0.85, 
                        max_iter=100, 
                        tol=1.0e-6, 
                        nstart=None, 
                        weight='weight', 
                        personalize_for=None):
    """
    Add PageRank scores to the nodes of the network.

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign of perturbation.
        target_dict (dict): A dictionary containing the targets and sign of measurements.
        percentage (int): Percentage of nodes to keep.
        alpha (float): Damping factor for the PageRank algorithm.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance to determine convergence.
        nstart (dict): Starting value of PageRank iteration for all nodes.
        weight (str): Edge data key to use as weight.
        personalize_for (str): Personalize the PageRank by setting initial probabilities for either sources or targets.

    Returns:
        tuple: Contains nodes above threshold from sources, nodes above threshold from targets, and overlapping nodes.
    """
    sources = source_dict.keys()
    targets = set()
    for key, value in target_dict.items():
        for sub_key, sub_value in value.items():
            targets.add(sub_key)

    if personalize_for == "source":
        personalized_prob = {n: 1/len(sources) for n in sources}
    elif personalize_for == "target":
        personalized_prob = {n: 1/len(targets) for n in targets}
        network = network.reverse()
    else:
        personalized_prob = None

    pagerank = nx.pagerank(network, 
                           alpha=alpha, 
                           max_iter=max_iter, 
                           personalization=personalized_prob, 
                           tol=tol, nstart=nstart, 
                           weight=weight, 
                           dangling=personalized_prob)
    
    if personalize_for == "target":
        network = network.reverse()

    for node, pr_value in pagerank.items():
        if personalize_for == "target":
            attribute_name = 'pagerank_from_targets'
        elif personalize_for == "source":
            attribute_name = 'pagerank_from_sources'
        elif personalize_for is None:
            attribute_name = 'pagerank'
        network.nodes[node][attribute_name] = pr_value

    return network


def compute_ppr_overlap(network, percentage=20):
    """
    Compute the overlap of nodes that exceed the personalized PageRank percentage threshold from sources and targets.

    Args:
        network (nx.Graph): The network.
        percentage (int): Percentage of top nodes to keep.

    Returns:
        tuple: Contains nodes above threshold from sources, nodes above threshold from targets, and overlapping nodes.
    """
    # Sorting nodes by PageRank score from sources and targets
    try:
        sorted_nodes_sources = sorted(network.nodes(data=True), 
                                      key=lambda x: x[1].get('pagerank_from_sources'), 
                                      reverse=True)
        sorted_nodes_targets = sorted(network.nodes(data=True), 
                                      key=lambda x: x[1].get('pagerank_from_targets'), 
                                      reverse=True)
    except KeyError:
        raise KeyError("Please run the add_pagerank_scores method first with personalization options.")

    # Calculating the number of nodes to keep
    num_nodes_to_keep_sources = int(len(sorted_nodes_sources) * (percentage / 100))
    num_nodes_to_keep_targets = int(len(sorted_nodes_targets) * (percentage / 100))

    # Selecting the top nodes
    nodes_above_threshold_from_sources = {node[0] for node in sorted_nodes_sources[:num_nodes_to_keep_sources]}
    nodes_above_threshold_from_targets = {node[0] for node in sorted_nodes_targets[:num_nodes_to_keep_targets]}

    overlap = nodes_above_threshold_from_sources.intersection(nodes_above_threshold_from_targets)
    nodes_to_include = nodes_above_threshold_from_sources.union(nodes_above_threshold_from_targets)

    subnetwork = network.subgraph(nodes_to_include)

    return subnetwork

def convert_cornetograph(graph):
    """
    Convert a networkx graph to a corneto graph, if needed.

    Args:
        graph (nx.Graph or nx.DiGraph): The corneto graph.

    Returns:
        cn.Graph: The corneto graph.
    """
    if type(graph) == cn._graph.Graph:
        corneto_graph = graph
    elif type(graph) == nx.Graph or type(graph) == nx.DiGraph:
        corneto_graph = networkx_to_corneto_graph(graph)
    
    return corneto_graph


def run_corneto_carnival(network, source_dict, target_dict, betaWeight=0.2, solver=None, verbose=True):
    """
    Run the Vanilla Carnival algorithm via CORNETO.

    Args:
        network (nx.Graph): The network.
        source_df (pd.DataFrame): A pandas DataFrame containing the sources.
        target_df (pd.DataFrame): A pandas DataFrame containing the targets. 
            Must contain three columns: source, target and sign

    Returns:
        nx.Graph: The subnetwork containing the paths found by CARNIVAL.
        list: A list containing the paths found by CARNIVAL.
    """
    corneto_net = convert_cornetograph(network)

    for source in list(source_dict.keys()):
        corneto_measurements = target_dict[source]
        problem, graph = cn.methods.runVanillaCarnival(perturbations=source_dict, 
                                                       measurements=corneto_measurements, 
                                                       priorKnowledgeNetwork=corneto_net, 
                                                       betaWeight=betaWeight,
                                                       solver=solver,
                                                       verbose=verbose)
    
    network_sol = graph.edge_subgraph(get_selected_edges(problem, graph))

    network_nx = corneto_graph_to_networkx(network_sol, skip_unsupported_edges=True)

    network_nx.remove_nodes_from(['_s', '_pert_c0', '_meas_c0'])

    return network_nx




