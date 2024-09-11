import pandas as pd
import networkx as nx
import numpy as np
import corneto as cn
import corneto.contrib.networkx as cn_nx
from networkcommons._session import _log


def node_attrs_from_corneto(graph: cn.Graph) -> pd.DataFrame:
    """
    Extract node attributes from a corneto graph to a pandas dataframe.

    Args:
        graph:
            A corneto graph.

    Returns:
        A pandas dataframe of node attributes.
    """

    return pd.DataFrame.from_dict(graph.get_attr_vertices())


def edge_attrs_from_corneto(graph: cn.Graph) -> pd.DataFrame:
    """
    Extract edge attributes from a corneto graph to a pandas dataframe.

    Args:
        graph:
            A corneto graph.

    Returns:
        A pandas dataframe of edge attributes.
    """

    edge_df = pd.DataFrame.from_dict(graph.get_attr_edges())
    concat_df = pd.concat([edge_df['__source_attr'], edge_df['__target_attr']]).reset_index()
    concat_df.rename(columns={0: 'node'}, inplace=True)


def to_cornetograph(graph):
    """
    Convert a networkx graph to a corneto graph, if needed.

    Args:
        graph (nx.DiGraph): The corneto graph.

    Returns:
        cn.Graph: The corneto graph.
    """
    if isinstance(graph, nx.MultiDiGraph):
        raise NotImplementedError("Only nx.DiGraph graphs and corneto graphs are supported.")
    elif isinstance(graph, cn.Graph):
        corneto_graph = graph
    elif isinstance(graph, nx.DiGraph):
        # substitute 'sign' for 'interaction' in the graph
        nx_graph = graph.copy()
        for u, v, data in nx_graph.edges(data=True):
            data['interaction'] = data.pop('sign')

        corneto_graph = cn_nx.networkx_to_corneto_graph(nx_graph)
    elif isinstance(graph, nx.Graph):
        raise NotImplementedError("Only nx.DiGraph graphs and corneto graphs are supported.")
    else:
        raise NotImplementedError("Only nx.DiGraph graphs and corneto graphs are supported.")

    return corneto_graph


def to_networkx(graph, skip_unsupported_edges=True):
    """
    Convert a corneto graph to a networkx graph, if needed.

    Args:
        graph (cn.Graph): The corneto graph.

    Returns:
        nx.Graph: The networkx graph.
    """
    if isinstance(graph, nx.MultiDiGraph):
        raise NotImplementedError("Only nx.DiGraph graphs and corneto graphs are supported.")
    elif isinstance(graph, nx.DiGraph):
        networkx_graph = graph
    elif isinstance(graph, cn.Graph):
        networkx_graph = cn_nx.corneto_graph_to_networkx(
            graph,
            skip_unsupported_edges=skip_unsupported_edges)
        # rename interaction for sign
        for u, v, data in networkx_graph.edges(data=True):
            data['sign'] = data.pop('interaction')
    elif isinstance(graph, nx.Graph):
        raise NotImplementedError("Only nx.DiGraph graphs and corneto graphs are supported.")
    else:
        raise NotImplementedError("Only nx.DiGraph graphs and corneto graphs are supported.")

    return networkx_graph


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
        df (DataFrame): A pandas DataFrame. Index should be populated
        column (str): The columns to be subsetted.

    Returns:
        A formatted DataFrame.
    """
    if not isinstance(columns, list):
        columns = [columns]
    df_f = df[columns].dropna().T
    return df_f


def targetlayer_formatter(df, n_elements=25, act_col='stat'):
    """
    Format dataframe to be used by the network methods.
    It converts the df values to 1, -1 and 0, and outputs a dictionary.

    Parameters:
        df (DataFrame): A pandas DataFrame.
        n_elements (int): The number of elements to be selected.
        act_col (str): The column containing the activity scores
    Returns:
        A dictionary {target: sign}
    """

    # Sort the DataFrame by the absolute value of the
    # 'sign' column and get top n elements
    df = df.sort_values(by=act_col, key=lambda x: abs(x), ascending=False)

    df = df[act_col].head(n_elements)

    df = df.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # Pivot wider
    dict_df = df.to_dict()

    return dict_df


def handle_missing_values(df, threshold=0.1, fill=np.mean):
    """
    Handles missing values in a DataFrame by filling them with a specified function or value, or dropping the rows.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - threshold (float): The threshold for the share (0<n<1) of missing values in a row. Rows with a share
                         of missing values greater than or equal to the threshold will be dropped.
    - fill (callable, int, float, or None): If callable, the function is applied to each row to fill missing values.
                                            If an integer or float, it is used to fill missing values.
                                            If None, no filling is done.

    Returns:
    - df (pandas.DataFrame): The DataFrame with missing values handled.

    Raises:
    - ValueError: If more than one non-numeric column is found in the DataFrame.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [3, 2, np.nan], 'C': [np.nan, 7, 8]})
    >>> handle_missing_values(df, 0.5, fill=np.mean)
    Number of genes filled: 1
    Number of genes removed: 1
    """
    df = df.copy()

    # Check non-numeric columns. If only one, set it as index. If more, abort.
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_columns) == 1:
        df = df.set_index(non_numeric_columns[0])
    elif len(non_numeric_columns) > 1:
        raise ValueError(f"More than one non-numeric column found: {non_numeric_columns}")

    # Replace -inf values with nan
    df = df.replace(-np.inf, np.nan)

    na_percentage = df.isna().mean(axis=1)

    # Determine which rows to fill and which to drop
    to_fill = na_percentage < threshold
    to_drop = na_percentage >= threshold

    filled_count = (df[to_fill].isna().sum(axis=1) > 0).sum()

    # Replace NAs based on the fill argument
    if callable(fill):
        # If fill is a function (like np.mean, np.median), apply it row-wise
        df.loc[to_fill] = df.loc[to_fill].apply(lambda row: row.fillna(fill(row)), axis=1)
        _log(f"Number of genes filled using function {fill.__name__}: {filled_count}")
    elif isinstance(fill, (int, float)):
        # If fill is a constant (like 0), use it directly
        df.loc[to_fill] = df.loc[to_fill].fillna(fill)
        _log(f"Number of genes filled with value {fill}: {filled_count}")
    elif fill is not None:
        raise ValueError("fill parameter must be a callable, a numeric value, or None")

    # Drop rows with NA percentage greater than or equal to threshold
    df = df[~to_drop]

    # Return index to column if necessary
    df = df.reset_index()

    removed_count = to_drop.sum()

    print(f"Number of genes removed: {removed_count}")

    return df


def subset_df_with_nodes(network, dataframe):
    """
    Subsets a dataframe using the nodes of a network as the index.

    Parameters:
    network (networkx.Graph): The network from which to extract nodes.
    dataframe (pd.DataFrame): The dataframe to subset.

    Returns:
    pd.DataFrame: A subset of the dataframe with rows that have indices matching the nodes of the network.
    """
    # Extract the nodes from the network
    nodes = list(network.nodes)

    # Subset the dataframe using the nodes as index
    subset_df = dataframe[dataframe.index.isin(nodes)]

    return subset_df
