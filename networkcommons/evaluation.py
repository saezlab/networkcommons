import pandas as pd
import networkx as nx
import numpy as np

def get_number_nodes(network):
    """
    Get the number of nodes in the network.

    Args:
        network (nx.Graph): The network to get the number of nodes from.

    Returns:
        int: The number of nodes in the network.
    """

    return network.number_of_nodes()



def get_number_edges(network):
    """
    Get the number of edges in the network.

    Args:
        network (nx.Graph): The network to get the number of edges from.

    Returns:
        int: The number of edges in the network.
    """

    return network.number_of_edges()



def get_mean_degree(network):
    """
    Get the mean degree centrality of the network.

    Args:
        network (nx.Graph): The network to get the mean degree from.

    Returns:
        float: The mean degree of the network.
    """

    return np.mean(list(dict(nx.degree(network)).values()))



def get_mean_betweenness(network):
    """
    Get the mean betweenness centrality of the network.

    Args:
        network (nx.Graph): The network to get the mean betweenness from.

    Returns:
        float: The mean betweenness of the network.
    """

    return np.mean(list(nx.betweenness_centrality(network).values()))



def get_mean_closeness(network):
    """
    Get the mean closeness centrality of the network.

    Args:
        network (nx.Graph): The network to get the mean closeness from.

    Returns:
        float: The mean closeness of the network.
    """

    return np.mean(list(nx.closeness_centrality(network).values()))
