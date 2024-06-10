import pandas as pd
import networkx as nx


def read_network_from_file(file_path,
                           source_col='source',
                           target_col='target',
                           directed=True,
                           sep='\t'):
    """
    Read network from a file.

    Args:
        file_path(str): Path to the file.
        source_col(str): Column name for the source nodes.
        target_col(str): Column name for the target nodes.
        directed(bool): Whether the network is directed or not.
        sep(str): Delimiter for the file.

    Returns:
        nx.Graph or nx.DiGraph: The network.
    """

    network_df = pd.read_csv(file_path, sep=sep)

    network = network_from_df(network_df,
                              source_col=source_col,
                              target_col=target_col,
                              directed=directed)

    return network


def network_from_df(network_df,
                    source_col='source',
                    target_col='target',
                    directed=True):
    """
    Create a network from a DataFrame.

    Args:
        df(DataFrame): DataFrame containing the network data.
        source_col(str): Column name for the source nodes.
        target_col(str): Column name for the target nodes.
        directed(bool): Whether the network is directed or not.

    Returns:
        nx.Graph or nx.DiGraph: The network.
    """
    network_type = nx.DiGraph if directed else nx.Graph

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
        if ('weight' in network_df.columns and
                (network_df['weight'] < 0).any()):
            for u, v, data in network.edges(data=True):
                weight = data['weight']
                data['sign'] = 1 if weight >= 0 else -1
                data['weight'] = abs(weight)

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
    for path in paths:
        for i in range(len(path) - 1):
            edge_data = network.get_edge_data(path[i], path[i + 1])
            subnetwork.add_edge(path[i], path[i + 1], **edge_data)

    return subnetwork


def decoupler_formatter(df,
                        columns: list):
    """
    Format dataframe to be used by decoupler.

    Parameters:
        df (DataFrame): A pandas DataFrame.
        column (str): The column to be used as index.

    Returns:
        A formatted DataFrame.
    """
    if not isinstance(columns, list):
        columns = [columns]
    df_f = df[columns].dropna().T
    return df_f


def targetlayer_formatter(df, source, n_elements=25):
    """
    Format dataframe to be used by the network methods.
    It converts the df values to 1, -1 and 0, and outputs a dictionary.

    Parameters:
        df (DataFrame): A pandas DataFrame.
        source (str): The source node of the perturbation.

    Returns:
        A dictionary of dictionaries {source: {target: sign}}
    """
    df.columns = ['sign']
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'target'}, inplace=True)
    df.insert(0, 'source', source)

    # Sort the DataFrame by the absolute value of the
    # 'sign' column and get top n elements
    df = df.sort_values(by='sign', key=lambda x: abs(x))

    df = df.head(n_elements)

    df['sign'] = df['sign'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # Pivot wider
    df = df.pivot(index='target', columns='source', values='sign')
    dict_df = df.to_dict()
    return dict_df
