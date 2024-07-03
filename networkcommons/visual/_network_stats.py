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

__all__ = ['plot_n_nodes_edges', 'plot_n_nodes_edges_from_df', 'build_heatmap_with_tree']

from typing import List, Dict, Union

import lazy_import
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def lollipop_plot(
        labels: List[str],
        values: List[List[int]],
        categories: List[List[str]],
        orientation: str,
        color_palette: str,
        size: int,
        linewidth: int,
        marker: str,
        title: str,
        filepath: str = None,
        render: bool = False
):
    """
    Common function to plot metrics using a lollipop plot.

    Args:
        labels (List[str]): Labels for the x or y axis.
        values (List[List[int]]): List of values to plot.
        categories (List[List[str]]): List of categories (e.g., 'Nodes', 'Edges') for each label.
        orientation (str): 'vertical' or 'horizontal'.
        color_palette (str): Matplotlib color palette.
        size (int): Size of the markers.
        linewidth (int): Line width of the lollipops.
        marker (str): Marker style for the lollipops.
        title (str): Title of the plot.
        filepath (str): Path to save the plot. Default is None.
        render (bool): Whether to display the plot. Default is False.
    """
    palette = plt.get_cmap(color_palette)
    colors = palette.colors if hasattr(palette, 'colors') else palette(range(len(labels)))

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, (label, value_set, category_set) in enumerate(zip(labels, values, categories)):
        color = colors[idx % len(colors)]

        if orientation == 'vertical':
            positions = [f"{label} {cat}" for cat in category_set]
            ax.vlines(x=positions, ymin=0, ymax=value_set, color=color, linewidth=linewidth, label=label)
            ax.scatter(positions, value_set, color=color, s=size ** 2, marker=marker, zorder=3)

            for i, value in enumerate(value_set):
                offset = size * 0.1 if value < 10 else size * 0.2
                ax.text(positions[i], value + offset, str(value), ha='center', va='bottom', fontsize=size)
        else:
            positions = [f"{label} {cat}" for cat in category_set]
            ax.hlines(y=positions, xmin=0, xmax=value_set, color=color, linewidth=linewidth, label=label)
            ax.scatter(value_set, positions, color=color, s=size ** 2, marker=marker, zorder=3)

            for i, value in enumerate(value_set):
                offset = size * 0.1 if value < 10 else size * 0.2
                ax.text(value + offset, positions[i], str(value), va='center', ha='left', fontsize=size)

    if orientation == 'vertical':
        ax.set_xlabel("Network and Type")
        ax.set_ylabel("Count")
    else:
        ax.set_ylabel("Network and Type")
        ax.set_xlabel("Count")

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
