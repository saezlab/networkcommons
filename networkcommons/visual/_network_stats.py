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

__all__  = ['plot_n_nodes_edges']

from typing import List, Dict

import lazy_import
import pandas as pd
plt = lazy_import.lazy_module('matplotlib.pyplot')
import networkx as nx


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

    # Set the color palette
    palette = plt.get_cmap(color_palette)
    colors = palette.colors if hasattr(palette, 'colors') else palette(range(len(networks)))

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, (network_name, network) in enumerate(networks.items()):
        # Get the number of nodes and edges
        n_nodes = len(network.nodes)
        n_edges = len(network.edges)
        categories = []
        values = []

        if show_nodes:
            categories.append('Nodes')
            values.append(n_nodes)
        if show_edges:
            categories.append('Edges')
            values.append(n_edges)

        color = colors[idx % len(colors)]

        if orientation == 'vertical':
            positions = [f"{network_name} {cat}" for cat in categories]
            ax.vlines(x=positions, ymin=0, ymax=values, color=color, linewidth=linewidth, label=network_name)
            ax.scatter(positions, values, color=color, s=size ** 2, marker=marker, zorder=3)

            # Annotate the values
            for i, value in enumerate(values):
                offset = size * 0.1 if value < 10 else size * 0.2
                ax.text(positions[i], value + offset, str(value), ha='center', va='bottom', fontsize=size)
        else:
            positions = [f"{network_name} {cat}" for cat in categories]
            ax.hlines(y=positions, xmin=0, xmax=values, color=color, linewidth=linewidth, label=network_name)
            ax.scatter(values, positions, color=color, s=size ** 2, marker=marker, zorder=3)

            # Annotate the values
            for i, value in enumerate(values):
                offset = size * 0.1 if value < 10 else size * 0.2
                ax.text(value + offset, positions[i], str(value), va='center', ha='left', fontsize=size)

    # Set the axis labels
    if orientation == 'vertical':
        ax.set_xlabel("Network and Type")
        ax.set_ylabel("Count")
    else:
        ax.set_ylabel("Network and Type")
        ax.set_xlabel("Count")

    # Set the title depending on the categories
    title = "Number of Nodes and Edges"
    if show_nodes and not show_edges:
        title = "Number of Nodes"
    elif show_edges and not show_nodes:
        title = "Number of Edges"
    ax.set_title(title)
    ax.legend()

    # Save the plot
    if filepath is not None:
        plt.savefig(filepath)

    if render:
        plt.show()
