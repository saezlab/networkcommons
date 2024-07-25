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
"""

from __future__ import annotations

__all__ = ['plot_n_nodes_edges',
           'plot_n_nodes_edges_from_df',
           'build_heatmap_with_tree',
           'lollipop_plot',
           'create_rank_heatmap',
           'plot_scatter',
           'plot_rank'
           ]

from typing import List, Dict, Union

import lazy_import
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def plot_rank(df, 
                   bio_ids=None,
                   figsize=(12, 6),
                   x_label='Proteins',
                   y_label='Average Intensity',
                   title='Protein abundance Rank Plot',
                   legend_labels=None,
                   id_column='idx',
                   average_color='blue',
                   stdev_color='gray',
                   stdev_alpha=0.2,
                   highlight_color='red',
                   highlight_size=5,
                   highlight_zorder=5):
    """
    Plots gene data with customizable attributes.

    Args:
        df (pd.DataFrame): Input DataFrame containing gene data.
        bio_ids (list of str): List of specific genes to highlight.
        figsize (tuple): Size of the figure.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        legend_labels (dict): Dictionary with legend labels for 'average', 'stdev', and 'highlight'.
        id_column (str): Name of the column containing the IDs (e.g gene_symbols or proteins).
        average_color (str): Color of the average intensity line.
        stdev_color (str): Color of the standard deviation shaded area.
        stdev_alpha (float): Alpha (transparency) for the standard deviation shaded area.
        highlight_color (str): Color for highlighting the specific gene.
        highlight_size (int): Size of the highlighted points.
        highlight_zorder (int): Z-order for the highlighted points.

    Returns:
        None
    """
    df = df.copy()
    # Compute average and standard error across columns, ignoring the non-numeric columns
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

    # Plot the shaded area for stderr
    plt.fill_between(df.index, df['average'] - df['stdev'], df['average'] + df['stdev'], color=stdev_color, alpha=stdev_alpha, label=legend_labels.get('stdev', 'Standard Deviation'))

    # Highlight the specific genes if bio_ids is provided
    if bio_ids:
        for bio_id in bio_ids:
            specific_gene = df[df[id_column].str.contains(bio_id, na=False)]
            if not specific_gene.empty:
                for _, gene in specific_gene.iterrows():
                    plt.scatter(gene.name, gene['average'], color=highlight_color, zorder=highlight_zorder, s=highlight_size)
                    plt.annotate(
                        bio_id,
                        xy=(gene.name, gene['average']),
                        xytext=(gene.name - n_proteins*0.02, gene['average'] + gene['stdev']),
                        arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=0, shrinkB=0, lw=1),
                        fontsize=9,
                        color=highlight_color
                    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scatter(df,
                              summarise_df=True,
                              x_col='diff_dysregulation',
                              y_col='coverage',
                              size_col='nodes_with_phosphoinfo',
                              hue_col='method',
                              style_col='type',
                              numeric_cols=None,
                              xlabel='Difference in Average Abundance',
                              ylabel='Coverage',
                              title='Coverage vs Difference in Average Abundance',
                              figsize=(10, 6)):
    """
    Plots a scatter plot with customizable column labels. It is prepared to be used
    by default with a dataframe from get_phosphorylation_status, as shown in Vignette 4.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to plot.
        summarise_df (bool): In case a random control has been performed, whether to summarise or not 
        the random networks.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        size_col (str): Column name for the size of the points.
        hue_col (str): Column name for the hue (color) of the points.
        style_col (str): Column name for the style of the points.
        numeric_cols (list of str): List of numeric columns to summarise.
        Defaults to all numeric columns in the DataFrame.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        figsize (tuple): Figure size of the plot.

    Returns:
        None
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns

    if summarise_df:
        summary_df = df.groupby([hue_col, style_col])[numeric_cols].mean().reset_index()
    else:
        summary_df = df

    # Plot
    plt.figure(figsize=figsize)
    sns.scatterplot(data=summary_df, x=x_col, y=y_col, size=size_col, hue=hue_col, style=style_col, sizes=(50, 200))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def lollipop_plot(
    df,
    label_col,
    value_col,
    orientation='vertical',
    color_palette='tab10',
    size=10, linewidth=2,
    marker='o',
    title='',
    filepath=None,
    render=False):
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
        marker (str): Marker style for the lollipops. Default is 'o'.
        title (str): Title of the plot. Default is ''.
        filepath (str): Path to save the plot. Default is None.
        render (bool): Whether to display the plot. Default is False.
    """
    palette = plt.get_cmap(color_palette)
    colors = palette.colors if hasattr(palette, 'colors') else palette(range(len(df[label_col].unique())))

    fig, ax = plt.subplots(figsize=(12, 8))

    labels = df[label_col]
    values = df[value_col]

    color = colors[0 % len(colors)]
    positions = labels

    if orientation == 'vertical':
        ax.vlines(x=positions, ymin=0, ymax=values, color=color, linewidth=linewidth, label=label_col)
        ax.scatter(positions, values, color=color, s=size ** 2, marker=marker, zorder=3)

        for i, value in enumerate(values):
            offset = size * 0.1 if value < 10 else size * 0.2
            ax.text(positions[i], value + offset, str(value), ha='center', va='bottom', fontsize=size)
    else:
        ax.hlines(y=positions, xmin=0, xmax=values, color=color, linewidth=linewidth, label=label_col)
        ax.scatter(values, positions, color=color, s=size ** 2, marker=marker, zorder=3)

        for i, value in enumerate(values):
            offset = size * 0.1 if value < 10 else size * 0.2
            ax.text(value + offset, positions[i], str(value), va='center', ha='left', fontsize=size)

    if orientation == 'vertical':
        ax.set_xlabel("Network")
        ax.set_ylabel(value_col)
    else:
        ax.set_ylabel("Network")
        ax.set_xlabel(value_col)

    ax.set_title(title)
    ax.legend()

    if filepath:
        plt.savefig(filepath)

    if render:
        plt.show()


def plot_n_nodes_edges(
        networks: Dict[str, nx.DiGraph],
        filepath=None,
        render=False,
        orientation='vertical',
        color_palette='Set2',
        size=10,
        linewidth=2,
        marker='o',
        show_nodes=True,
        show_edges=True
):
    """
    Plot the number of nodes and edges in the networks using a lollipop plot.

    Args:
        networks (Dict[str, nx.DiGraph]): A dictionary of network names and their corresponding graphs.
        filepath (str): Path to save the plot. Default is None.
        render (bool): Whether to display the plot. Default is False.
        orientation (str): 'vertical' or 'horizontal'. Default is 'vertical'.
        color_palette (str): Matplotlib color palette. Default is 'Set2'.
        size (int): Size of the markers. Default is 10.
        linewidth (int): Line width of the lollipops. Default is 2.
        marker (str): Marker style for the lollipops. Default is 'o'.
        show_nodes (bool): Whether to show nodes count. Default is True.
        show_edges (bool): Whether to show edges count. Default is True.
    """
    if not show_nodes and not show_edges:
        raise ValueError("At least one of 'show_nodes' or 'show_edges' must be True.")

    labels = []
    values = []
    categories = []

    for network_name, network in networks.items():
        n_nodes = len(network.nodes)
        n_edges = len(network.edges)
        value_set = []
        category_set = []

        if show_nodes:
            category_set.append('Nodes')
            value_set.append(n_nodes)
        if show_edges:
            category_set.append('Edges')
            value_set.append(n_edges)

        labels.append(network_name)
        values.append(value_set)
        categories.append(category_set)

    title = "Number of Nodes and Edges"
    if show_nodes and not show_edges:
        title = "Number of Nodes"
    elif show_edges and not show_nodes:
        title = "Number of Edges"

    lollipop_plot(labels, values, categories, orientation, color_palette, size, linewidth, marker, title, filepath,
                  render)


def plot_n_nodes_edges_from_df(
        metrics_df: pd.DataFrame,
        metrics: List[str],
        filepath=None,
        render=False,
        orientation='vertical',
        color_palette='Set2',
        size=10,
        linewidth=2,
        marker='o'
):
    """
    Plot the specified metrics from a DataFrame using a lollipop plot.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics with networks as rows and specified metrics in columns.
        metrics (List[str]): List of column names in the DataFrame to plot.
        filepath (str): Path to save the plot. Default is None.
        render (bool): Whether to display the plot. Default is False.
        orientation (str): 'vertical' or 'horizontal'. Default is 'vertical'.
        color_palette (str): Matplotlib color palette. Default is 'Set2'.
        size (int): Size of the markers. Default is 10.
        linewidth (int): Line width of the lollipops. Default is 2.
        marker (str): Marker style for the lollipops. Default is 'o'.
    """
    if not metrics:
        raise ValueError("At least one metric must be specified.")

    labels = []
    values = []
    categories = []

    for network_name, row in metrics_df.iterrows():
        value_set = []
        category_set = []

        for metric in metrics:
            category_set.append(metric)
            value_set.append(row[metric])

        labels.append(network_name)
        values.append(value_set)
        categories.append(category_set)

    title = "Metrics"
    if len(metrics) == 1:
        title = f"Number of {metrics[0]}"
    else:
        title = f"Number of {' and '.join(metrics)}"

    lollipop_plot(labels, values, categories, orientation, color_palette, size, linewidth, marker, title, filepath,
                  render)


def build_heatmap_with_tree(
        jaccard_df: pd.DataFrame,
        title: str = "Heatmap (Jaccard Distance)",
        palette: str = "viridis",
        save: bool = False,
        output_dir: str = "."
):
    """
    Build a heatmap with hierarchical clustering based on a Jaccard distance matrix.

    Args:
        jaccard_df (pd.DataFrame): DataFrame containing Jaccard distance matrix.
        title (str): Title of the plot.
        palette (str): Color palette for the heatmap. Default is "viridis".
        save (bool): Whether to save the plot. Default is False.
        output_dir (str): Directory to save the plot. Default is ".".
    """
    # Convert the square distance matrix to a condensed distance matrix
    condensed_dist_matrix = squareform(jaccard_df)

    # Perform hierarchical clustering
    linked = linkage(condensed_dist_matrix, method='average')

    # Create the clustermap
    g = sns.clustermap(
        jaccard_df,
        row_linkage=linked,
        col_linkage=linked,
        cmap=palette,
        figsize=(12, 10),
        cbar_kws={'label': 'Jaccard Distance'}
    )

    # Adjust the position of the title to make sure it appears
    # (depends on specific data/plot size)
    g.fig.suptitle(title, fontsize=14, x=0.5, y=1.00)

    if save:
        plt.savefig(f"{output_dir}/heatmap_with_tree.png", bbox_inches='tight')

    plt.show()


def create_rank_heatmap(ora_results, ora_terms):
    """
    Create a heatmap with rows as ora_terms and columns as networks,
    displaying the rank number in the cells.

    Args:
        ora_results (pd.DataFrame): DataFrame containing the ORA results with ranks.
        ora_terms (list): List of ora_Term to include in the heatmap.

    Returns:
        None
    """
    # Filter the ORA results to include only the specified ora_Term
    filtered_data = ora_results[ora_results['ora_Term'].isin(ora_terms)]

    # Pivot the dataframe to have ora_Term as rows and network as columns
    heatmap_data = filtered_data.pivot(index='ora_Term', columns='network', values='ora_rank')

    xsize = len(heatmap_data.columns)
    ysize = len(heatmap_data.index)*2

    # Create the heatmap
    plt.figure(figsize=(xsize, ysize))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm_r", linewidths=.5, cbar=False)
    plt.title("Heatmap of ORA Term Ranks by Network")
    plt.ylabel('ORA Term')
    plt.xlabel('Network')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()