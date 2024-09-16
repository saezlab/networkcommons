#!/usr/bin/env python

#
# This file is part of the `networkcommons` Python module
#
# Copyright 2024
# Heidelberg University Hospital
#
# File author(s): Saez Lab (omnipathdb@gmail.com)
#
# Distributed under the GPLv3 license
# See the file `LICENSE` or read a copy at
# https://www.gnu.org/licenses/gpl-3.0.txt
#

"""
Plot network (graph) metrics.

This module provides several functions to visualize network metrics and generate
plots for data analysis. Functions include plotting the number of nodes and edges,
creating heatmaps, generating scatter plots, and more.
"""

from __future__ import annotations
from networkcommons._session import _log

__all__ = [
    'plot_n_nodes_edges',
    'plot_n_nodes_edges_from_df',
    'build_heatmap_with_tree',
    'lollipop_plot',
    'create_heatmap',
    'plot_scatter',
    'plot_rank'
]

from typing import List, Dict, Union
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def plot_rank(df: pd.DataFrame,
              bio_ids: List[str] = None,
              figsize: tuple = (12, 6),
              x_label: str = 'Proteins',
              y_label: str = 'Average Intensity',
              title: str = 'Protein abundance Rank Plot',
              legend_labels: dict = None,
              id_column: str = 'idx',
              average_color: str = 'blue',
              stdev_color: str = 'gray',
              stdev_alpha: float = 0.2,
              highlight_color: str = 'red',
              highlight_size: int = 5,
              highlight_zorder: int = 5,
              filepath: str = None,
              render: bool = False) -> plt.Figure:
    """
    Plot a protein abundance rank plot.

    This function generates a plot showing protein abundance ranked by their average intensity
    with an option to highlight specific proteins.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing gene or protein data.
    bio_ids : list of str, optional
        List of specific genes or proteins to highlight. Defaults to None.
    figsize : tuple, optional
        Size of the figure. Defaults to (12, 6).
    x_label : str, optional
        Label for the x-axis. Defaults to 'Proteins'.
    y_label : str, optional
        Label for the y-axis. Defaults to 'Average Intensity'.
    title : str, optional
        Title of the plot. Defaults to 'Protein abundance Rank Plot'.
    legend_labels : dict, optional
        Dictionary with legend labels for 'average', 'stdev', and 'highlight'. Defaults to None.
    id_column : str, optional
        Name of the column containing the IDs (e.g., gene symbols or proteins). Defaults to 'idx'.
    average_color : str, optional
        Color of the average intensity line. Defaults to 'blue'.
    stdev_color : str, optional
        Color of the standard deviation shaded area. Defaults to 'gray'.
    stdev_alpha : float, optional
        Transparency for the standard deviation shaded area. Defaults to 0.2.
    highlight_color : str, optional
        Color for highlighting specific genes or proteins. Defaults to 'red'.
    highlight_size : int, optional
        Size of the highlighted points. Defaults to 5.
    highlight_zorder : int, optional
        Z-order for the highlighted points. Defaults to 5.
    filepath : str, optional
        Path to save the plot. If None, the plot will not be saved. Defaults to None.
    render : bool, optional
        Whether to display the plot. Defaults to False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for the plot.
    """
    if id_column not in df.columns:
        _log(f"Column '{id_column}' not found in the DataFrame. Using the index as the ID column.")
        df = df.reset_index()  # Reset the index, moving it to a new column
        id_column = 'index'  # Set id_column to 'index' as it's now a column
        df[id_column] = df[id_column].astype(str)

    df = df.copy()
    # Compute average and standard deviation across columns, ignoring the non-numeric columns
    df['average'] = df.select_dtypes(include=[np.number]).mean(axis=1)
    df['stdev'] = df.select_dtypes(include=[np.number]).std(axis=1)

    # Sort by average value
    df = df.sort_values('average').reset_index(drop=True)

    n_proteins = len(df)

    # Ensure legend_labels is a dictionary
    if legend_labels is None:
        legend_labels = {}

    # Plotting
    plt.figure(figsize=figsize)

    # Plot the average intensity
    plt.plot(df['average'], label=legend_labels.get('average', 'Average Intensity'), color=average_color)

    # Plot the shaded area for standard deviation
    plt.fill_between(df.index, df['average'] - df['stdev'], df['average'] + df['stdev'], color=stdev_color,
                     alpha=stdev_alpha, label=legend_labels.get('stdev', 'Standard Deviation'))

    # Highlight the specific genes or proteins if bio_ids is provided
    if bio_ids:
        for bio_id in bio_ids:
            specific_gene = df[df[id_column].str.contains(bio_id, na=False)]
            if not specific_gene.empty:
                for _, gene in specific_gene.iterrows():
                    plt.scatter(gene.name, gene['average'], color=highlight_color,
                                zorder=highlight_zorder, s=highlight_size)
                    plt.annotate(
                        bio_id,
                        xy=(gene.name, gene['average']),
                        xytext=(gene.name - n_proteins * 0.02, gene['average'] + gene['stdev']),
                        arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=0, shrinkB=0, lw=1),
                        fontsize=9,
                        color=highlight_color
                    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()

    if filepath:
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)

    if render:
        plt.show()

    if not filepath and not render:
        _log("No output specified. Returning the plot object.")

    return plt.gcf()

def plot_scatter(df: pd.DataFrame,
                 summarise_df: bool = True,
                 x_col: str = 'diff_dysregulation',
                 y_col: str = 'coverage',
                 size_col: str = 'nodes_with_phosphoinfo',
                 hue_col: str = 'method',
                 style_col: str = 'type',
                 numeric_cols: List[str] = None,
                 xlabel: str = 'Difference in Activity scores',
                 ylabel: str = 'Coverage',
                 title: str = 'Coverage vs Difference in Activity scores',
                 figsize: tuple = (10, 6),
                 filepath: str = "scatter_plot.png",
                 render: bool = False) -> plt.Figure:
    """
    Plot a scatter plot with customizable column labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to plot.
    summarise_df : bool, optional
        Whether to summarize the random networks if a random control has been performed. Defaults to True.
    x_col : str, optional
        Column name for the x-axis. Defaults to 'diff_dysregulation'.
    y_col : str, optional
        Column name for the y-axis. Defaults to 'coverage'.
    size_col : str, optional
        Column name for the size of the points. Defaults to 'nodes_with_phosphoinfo'.
    hue_col : str, optional
        Column name for the hue (color) of the points. Defaults to 'method'.
    style_col : str, optional
        Column name for the style of the points. Defaults to 'type'.
    numeric_cols : list of str, optional
        List of numeric columns to summarize. Defaults to all numeric columns in the DataFrame.
    xlabel : str, optional
        Label for the x-axis. Defaults to 'Difference in Activity scores'.
    ylabel : str, optional
        Label for the y-axis. Defaults to 'Coverage'.
    title : str, optional
        Title of the plot. Defaults to 'Coverage vs Difference in Activity scores'.
    figsize : tuple, optional
        Figure size of the plot. Defaults to (10, 6).
    filepath : str, optional
        Path to save the plot. If None, the plot will be saved as "scatter_plot.png". Defaults to "scatter_plot.png".
    render : bool, optional
        Whether to display the plot. Defaults to False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for the plot.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns

    if summarise_df and style_col is not None:
        summary_df = df.groupby([hue_col, style_col])[numeric_cols].mean().reset_index()
    else:
        summary_df = df

    # Plot
    plt.figure(figsize=figsize)
    sns.scatterplot(data=summary_df, x=x_col, y=y_col, size=size_col,
                    hue=hue_col, style=style_col, sizes=(50, 200))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if filepath:
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)

    if render:
        plt.show()

    if not filepath and not render:
        _log("No output specified. Returning the plot object.")

    return plt.gcf()


def lollipop_plot(
    df,
    label_col,
    value_col,
    orientation='vertical',
    color_palette='tab10',
    size=10,
    linewidth=2,
    label_fontsize=10,
    label_gap=0.1,
    rounding_decimals=2,
    marker='o',
    title='',
    plt_size=(12, 8),
    filepath=None,
    render=False) -> plt.Figure:
    """
    Function to plot metrics using a lollipop plot from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        label_col (str): Column name for labels.
        value_col (str): Column name for values.
        orientation (str): 'vertical' or 'horizontal'. Default is 'vertical'.
        color_palette (str): Matplotlib color palette. Default is 'tab10'.
        size (int): Size of the markers. Default is 10.
        linewidth (int): Line width of the lollipops. Default is 2.
        label_fontsize (int): Font size for the labels. Default is 10.
        label_gap (float): Gap between the value and the label. Default is 0.1.
        marker (str): Marker style for the lollipops. Default is 'o'.
        rounding_decimals (int): Number of decimal places to round the values to. Default is 2.
        title (str): Title of the plot. Default is ''.
        plt_size (tuple): Size of the plot. Default is (12, 8).
        filepath (str): Path to save the plot. Default is None.
        render (bool): Whether to display the plot. Default is False.
    """
    # Determine color palette
    palette = plt.get_cmap(color_palette)
    colors = palette.colors if hasattr(palette, 'colors') else palette(np.linspace(0, 1, len(df[label_col].unique())))

    # Dynamically adjust figure size based on number of labels
    num_labels = len(df[label_col].unique())
    fig, ax = plt.subplots(figsize=(max(12, num_labels * 0.6), 8))

    if "Category" not in df.columns:
        df['Category'] = 'All'  # Default category if not provided

    labels = df[label_col].unique()
    categories = df['Category'].unique()
    offset = 0.15  # Offset for separating categories side by side

    # Find the min and max values in the dataset and apply padding
    borders_padding = 0.20
    max_value = df[value_col].max()
    min_value = df[value_col].min()
    value_limit_max = max_value + abs(max_value) * borders_padding
    value_limit_min = min_value - abs(min_value) * borders_padding

    for i, category in enumerate(categories):
        subset = df[df['Category'] == category]
        subset_labels = subset[label_col].values  # Only use the labels available in the subset
        if orientation == 'vertical':
            positions = np.arange(len(subset_labels)) + (i - len(categories) / 2) * offset
        else:
            positions = np.arange(len(subset_labels)) + (i - len(categories) / 2) * offset

        values = subset[value_col].round(rounding_decimals)  # Round the values
        values = values.fillna(0)  # Fill NaN values with 0
        color = colors[i % len(colors)]

        if orientation == 'vertical':
            # Handle both positive and negative values
            for j, value in enumerate(values):
                ax.vlines(x=positions[j], ymin=0 if value >= 0 else value, ymax=value if value >= 0 else 0,
                          color=color, linewidth=linewidth, label=category if j == 0 else "")  # Label only once
                ax.scatter(positions[j], value, color=color, s=size ** 2, marker=marker, zorder=3)

                # Determine the format based on whether it's an integer or float
                if value == int(value):
                    value_str = f'{int(value)}'
                else:
                    value_str = f'{value:.{rounding_decimals}f}'

                # Adjust the label gap depending on positive or negative values
                ax.text(positions[j], value + size * label_gap * np.sign(value), value_str,
                        ha='center', va='bottom' if value >= 0 else 'top', fontsize=label_fontsize - 2)
        else:
            # Handle both positive and negative values in horizontal orientation
            for j, value in enumerate(values):
                ax.hlines(y=positions[j], xmin=0 if value >= 0 else value, xmax=value if value >= 0 else 0,
                          color=color, linewidth=linewidth, label=category if j == 0 else "")  # Label only once
                ax.scatter(value, positions[j], color=color, s=size ** 2, marker=marker, zorder=3)

                # Determine the format based on whether it's an integer or float
                if  value == int(value):
                    value_str = f'{int(value)}'
                else:
                    value_str = f'{value:.{rounding_decimals}f}'

                ax.text(value + size * label_gap * np.sign(value), positions[j], value_str,
                        va='center', ha='left' if value >= 0 else 'right', fontsize=label_fontsize - 2)

    # Adjust axis limits by applying padding to the min and max values
    if orientation == 'vertical':
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel(label_col)
        ax.set_ylabel(value_col)
        ax.set_ylim(value_limit_min, value_limit_max)  # Apply padding to y-axis
    else:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_ylabel(label_col)
        ax.set_xlabel(value_col)
        ax.set_xlim(value_limit_min, value_limit_max)  # Apply padding to x-axis

    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title="Category", loc='best')

    # Adjust layout
    fig.tight_layout()
    fig.set_size_inches(plt_size)

    # Save the figure if a filepath is provided
    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')

    # Render the plot if specified
    if render:
         plt.show()

    return fig


def plot_n_nodes_edges(networks: Dict[str, nx.DiGraph],
                       filepath: str = None,
                       render: bool = False,
                       orientation: str = 'vertical',
                       color_palette: str = 'Set2',
                       size: int = 10,
                       linewidth: int = 2,
                       marker: str = 'o',
                       show_nodes: bool = True,
                       show_edges: bool = True) -> plt.Figure:
    """
    Plot the number of nodes and edges in the networks using a lollipop plot.

    Parameters
    ----------
    networks : dict of nx.DiGraph
        A dictionary of network names and their corresponding graphs.
    filepath : str, optional
        Path to save the plot. Defaults to None.
    render : bool, optional
        Whether to display the plot. Defaults to False.
    orientation : str, optional
        Orientation of the plot ('vertical' or 'horizontal'). Defaults to 'vertical'.
    color_palette : str, optional
        Matplotlib color palette to use. Defaults to 'Set2'.
    size : int, optional
        Size of the markers. Defaults to 10.
    linewidth : int, optional
        Line width of the lollipops. Defaults to 2.
    marker : str, optional
        Marker style for the lollipops. Defaults to 'o'.
    show_nodes : bool, optional
        Whether to show the number of nodes. Defaults to True.
    show_edges : bool, optional
        Whether to show the number of edges. Defaults to True.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for the plot.
    """
    if not show_nodes and not show_edges:
        _log("Both 'show_nodes' and 'show_edges' are False. Using show nodes as default.")
        show_nodes = True

    labels = []
    categories = []
    values = []

    for network_name, network in networks.items():
        if show_nodes:
            labels.append(network_name)
            categories.append('Nodes')
            values.append(len(network.nodes))

        if show_edges:
            labels.append(network_name)
            categories.append('Edges')
            values.append(len(network.edges))

    title = "Number of Nodes and Edges"
    if show_nodes and not show_edges:
        title = "Number of Nodes"
    elif show_edges and not show_nodes:
        title = "Number of Edges"

    # Construct the DataFrame
    df = pd.DataFrame({
        'Network': labels,
        'Category': categories,
        'Values': values
    })

    # Plot using the updated lollipop_plot function
    lolli_plot = lollipop_plot(
        df,
        label_col="Network",
        value_col="Values",
        orientation=orientation,
        color_palette=color_palette,
        size=size,
        linewidth=linewidth,
        marker=marker,
        title=title,
        filepath=filepath,
        render=render
    )

    return lolli_plot


def plot_n_nodes_edges_from_df(metrics_df: pd.DataFrame,
                               metrics: List[str],
                               filepath: str = None,
                               render: bool = False,
                               orientation: str = 'vertical',
                               color_palette: str = 'Set2',
                               size: int = 10,
                               linewidth: int = 2,
                               marker: str = 'o') -> plt.Figure:
    """
    Plot the specified metrics from a DataFrame using a lollipop plot.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing metrics with networks as rows and specified metrics in columns.
    metrics : list of str
        List of column names in the DataFrame to plot.
    filepath : str, optional
        Path to save the plot. Defaults to None.
    render : bool, optional
        Whether to display the plot. Defaults to False.
    orientation : str, optional
        Orientation of the plot ('vertical' or 'horizontal'). Defaults to 'vertical'.
    color_palette : str, optional
        Matplotlib color palette to use. Defaults to 'Set2'.
    size : int, optional
        Size of the markers. Defaults to 10.
    linewidth : int, optional
        Line width of the lollipops. Defaults to 2.
    marker : str, optional
        Marker style for the lollipops. Defaults to 'o'.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for the plot.

    Raises
    ------
    ValueError
        If no metrics are provided or if the DataFrame is empty.
    TypeError
        If elements in 'metrics' are not strings.
    """
    if not metrics:
        raise ValueError("At least one metric must be specified.")

    if not all(isinstance(metric, str) for metric in metrics):
        raise TypeError("All elements in 'metrics' must be strings corresponding to column names in the DataFrame.")

    missing_columns = [metric for metric in metrics if metric not in metrics_df.columns]
    if missing_columns:
        raise ValueError(f"The following metrics are not columns in the DataFrame: {', '.join(missing_columns)}")

    if metrics_df.empty:
        raise ValueError("The metrics DataFrame is empty.")

    # Flatten the data: Each row represents one network and one metric
    rows = []
    for network_name, row in metrics_df.iterrows():
        for metric in metrics:
            rows.append({'Network': network_name, 'Category': metric, 'Values': row[metric]})

    # Create a DataFrame from the flattened data
    df = pd.DataFrame(rows)

    title = "Metrics"
    if len(metrics) == 1:
        title = f"Number of {metrics[0]}"
    else:
        title = f"Number of {' and '.join(metrics)}"

    # Plot using the updated lollipop_plot function
    lolli_plot = lollipop_plot(
        df,
        label_col="Network",
        value_col="Values",
        orientation=orientation,
        color_palette=color_palette,
        size=size,
        linewidth=linewidth,
        marker=marker,
        title=title,
        filepath=filepath,
        render=render
    )

    return lolli_plot

def build_heatmap_with_tree(distance_df: pd.DataFrame,
                            distance: str = 'jaccard',
                            title: str = "Heatmap Distance Matrix",
                            palette: str = "viridis",
                            save: bool = False,
                            output_dir: str = ".",
                            render=False):
    """
    Build a heatmap with hierarchical clustering based on a Jaccard distance matrix.

    Args:
        distance_df (pd.DataFrame): DataFrame containing the distance matrix.
        distance (str, optional): Type of distance metric used. Defaults to 'jaccard'.
        title (str, optional): Title of the plot. Defaults to "Heatmap Distance Matrix".
        palette (str, optional): Color palette for the heatmap. Defaults to "viridis".
        save (bool, optional): Whether to save the plot. Defaults to False.
        output_dir (str, optional): Directory to save the plot. Defaults to ".".
        render (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    # Convert the square distance matrix to a condensed distance matrix
    condensed_dist_matrix = squareform(distance_df)

    # Perform hierarchical clustering
    linked = linkage(condensed_dist_matrix, method='average')

    # Create the clustermap
    g = sns.clustermap(
        distance_df,
        row_linkage=linked,
        col_linkage=linked,
        cmap=palette,
        figsize=(12, 10),
        cbar_kws={'label': f'{distance} Distance'}
    )

    # Adjust the position of the title to make sure it appears
    g.fig.suptitle(title, fontsize=14, x=0.5, y=1.00)

    if save:
        g.fig.savefig(f"{output_dir}/heatmap_with_tree.png", bbox_inches='tight')

    if render:
        g.fig.show()

    if not save and not render:
        _log("No output specified. Returning the plot")


    return g.fig


def create_heatmap(results: pd.DataFrame,
                   terms=None,
                   y_axis_col: str = 'ora_Term',
                   x_axis_col: str = 'network',
                   value_col: str = 'ora_rank',
                   title: str = "Heatmap of ORA Term Ranks by Network",
                   x_label: str = 'Network',
                   y_label: str = 'ORA Term',
                   cmap="coolwarm_r",
                   filepath="rank_heatmap.png",
                   render=False):
    """
    Create a heatmap. By default, creates a heatmap with rows as ora_terms and columns as networks,
    displaying the rank number in the cells.

    Args:
        results (pd.DataFrame): DataFrame containing the ORA results with ranks.
        terms (list): List of ORA terms to include in the heatmap.
        y_axis_col (str): Column name for the y-axis. Default is 'ora_Term'.
        x_axis_col (str): Column name for the x-axis. Default is 'network'.
        value_col (str): Column name for the values. Default is 'ora_rank'.
        title (str): Title of the plot. Default is "Heatmap of ORA Term Ranks by Network".
        x_label (str): Label for the x-axis. Default is 'Network'.
        y_label (str): Label for the y-axis. Default is 'ORA Term'.
        cmap (str): Color map for the heatmap. Default is "coolwarm_r".
        filepath (str): Path to save the plot. Default is "rank_heatmap.png".
        render (bool): Whether to display the plot. Default is False.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    if terms is not None:
        # Filter the results to include only the specified terms
        results = results[results[y_axis_col].isin(terms)]

    # Pivot the DataFrame to have ora_Term as rows and network as columns
    heatmap_data = results.pivot(index=y_axis_col, columns=x_axis_col, values=value_col)

    xsize = len(heatmap_data.columns)
    ysize = len(heatmap_data.index)

    # Create the heatmap
    plt.figure(figsize=(xsize, ysize))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap=cmap, linewidths=.5, cbar=False)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if filepath:
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)

    if render:
        plt.show()

    if not filepath and not render:
        _log("No output specified. Returning the plot object.")

    return plt.gcf()