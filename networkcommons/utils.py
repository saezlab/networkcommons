import pandas as pd
import networkx as nx
import numpy as np

def read_network_from_file(file_path, source_col='source', target_col='target', directed=True, sep='\t'):
    """
    Read network from a file.
    
    Parameters
    ----------
    file_path : str
        Path to the csv file.
    source_col : str
        Column name for the source nodes.
    target_col : str
        Column name for the target nodes.
    directed : bool
        Whether the network is directed or not.
    sep : str
        Delimiter for the file.
        
    Returns
    -------
    network : nx.Graph or nx.DiGraph
        The network.
    """

    network_type = nx.DiGraph if directed else nx.Graph

    network_df = pd.read_csv(file_path, sep=sep)

    if list(network_df.columns) == list([source_col, target_col]):
        network = nx.from_pandas_edgelist(network_df, 
                                          source=source_col, 
                                          target=target_col, 
                                          create_using=network_type)
    else:
        network = nx.from_pandas_edgelist(network_df, 
                                          source=source_col, 
                                          target=target_col, 
                                          edge_attr=True, 
                                          create_using=network_type)

    return network

def get_subnetwork(network, paths):
    """
    Creates a subnetwork from a list of paths.

    Args:
        paths (list): A list of lists containing paths.
        network: original graph from which the paths were extracted.
    
    Returns:
        Tuple: Contains the subgraph, list of connected targets per source
    """
    directed = nx.is_directed(network)
    subnetwork = nx.DiGraph() if directed else nx.Graph()
    connected_targets = {}
    for path in paths:
        source = path[0]
        target = path[-1]

        if source not in connected_targets:
            connected_targets[source] = []

        connected_targets[source].append(target) if target not in connected_targets[source] else None
        for i in range(len(path) - 1):
            edge_data = network.get_edge_data(path[i], path[i + 1])
            subnetwork.add_edge(path[i], path[i + 1], **edge_data)
    
    return subnetwork, connected_targets