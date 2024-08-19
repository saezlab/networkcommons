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
Module for Visualizing Networks with Customizable Styles.

This module provides functionality to visualize networks using various styles.

Classes:
- `NetworkVisualizerBase`: A basic visualizer for networks with minimal functionality.
- `NetworkXVisualizer`: An advanced visualizer that extends `NetworkVisualizerBase` with additional features for network visualization.

Functions:
- `get_styles()`: Returns the default styles used for visualizing nodes and edges.
- `set_style_attributes()`: Sets style attributes for nodes and edges based on the provided styles.
- `visualize_network_default()`: Visualizes the network using default styles.
- `visualize_network_sign_consistent()`: Visualizes the network considering sign consistency.
- `visualize_network()`: Main function for network visualization based on the network type.

"""

from __future__ import annotations

__all__ = ['NetworkVisualizerBase', 'NetworkXVisualizer']

import lazy_import
import networkx as nx
import matplotlib.pyplot as plt

from . import _aux
from . import _styles
from networkcommons._session import _log


class NetworkVisualizerBase:
    """
    A base class for visualizing networks with basic styling options.

    Attributes:
        network (nx.Graph): The network graph to visualize.
        style (dict): The style dictionary for visualizing the graph.

    Methods:
        __init__(network): Initializes the visualizer with a network and default style.
        set_custom_style(custom_style): Sets a custom style for visualization.
        visualize_graph_simple(source_dict, target_dict, prog, is_sign_consistent, max_nodes): Visualizes the network with basic settings and styling.
    """

    def __init__(self, network):
        """
        Initializes the visualizer with a network and default style.

        Args:
            network (nx.Graph): The network graph to visualize.
        """
        self.network = network
        self.style = _styles.get_styles()['default']  # Basic style applied by default

    def set_custom_style(self, custom_style=None):
        """
        Sets a custom style for the visualizer, merging it with the default style.

        Args:
            custom_style (dict, optional): Custom style dictionary to merge with the default style.
        """
        if custom_style:
            self.style = _styles.merge_styles(self.style, custom_style)

    def visualize_graph_simple(self, source_dict, target_dict, prog='dot', is_sign_consistent=True, max_nodes=75):
        """
        Visualizes the graph using basic settings and styling.

        Args:
            source_dict (dict): A dictionary of sources and perturbation signs.
            target_dict (dict): A dictionary of targets and measurement signs.
            prog (str, optional): Layout program to use. Defaults to 'dot'.
            is_sign_consistent (bool, optional): Whether to visualize only sign-consistent paths. Defaults to True.
            max_nodes (int, optional): Maximum number of nodes for visualization. Defaults to 75.

        Returns:
            A (pygraphviz.AGraph): The visualized network graph.
        """
        if len(self.network.nodes) > max_nodes:
            _log.error("The network is too large to visualize.")
            return

        A = nx.nx_agraph.to_agraph(self.network)
        A.graph_attr['ratio'] = '1.2'

        sources = list(source_dict.keys())
        targets = set(target_dict.keys())

        for node in A.nodes():
            n = node.get_name()
            if n in sources:
                base_style = self.style['nodes']['sources']
            elif n in targets:
                base_style = self.style['nodes']['targets']
            else:
                base_style = self.style['nodes']['default']

            _styles.set_style_attributes(node, base_style)

        for edge in A.edges():
            u, v = edge
            edge_data = self.network.get_edge_data(u, v)
            if 'interaction' in edge_data:
                edge_data['sign'] = edge_data.pop('interaction')

            if edge_data['sign'] == 1 and is_sign_consistent:
                edge_style = self.style['edges']['positive']
            elif edge_data['sign'] == -1 and is_sign_consistent:
                edge_style = self.style['edges']['negative']
            else:
                edge_style = self.style['edges']['default']

            _styles.set_style_attributes(edge, edge_style)

        A.layout(prog=prog)
        return A


class NetworkXVisualizer(NetworkVisualizerBase):
    """
    An advanced visualizer that extends `NetworkVisualizerBase` with additional features.

    Attributes:
        color_by (str): Attribute to color edges by.
        edge_colors (dict): Edge colors used for visualization.

    Methods:
        __init__(network, color_by): Initializes the visualizer with a network and color attribute.
        set_custom_edge_colors(custom_edge_colors): Sets custom colors for edges.
        color_nodes(): Colors nodes based on their type.
        color_edges(): Colors edges based on a specified attribute.
        visualize_network_default(network, source_dict, target_dict, prog, custom_style): Visualizes the network using default styles.
        visualize_network_sign_consistent(network, source_dict, target_dict, prog, custom_style): Visualizes the network with sign consistency.
        visualize_network(network, source_dict, target_dict, prog, network_type, custom_style): Main function for network visualization based on network type.
        visualize(output_file, render, highlight_nodes, style): Visualizes the network and saves or shows the plot.
    """

    def __init__(self, network, color_by="effect"):
        """
        Initializes the visualizer with a network and color attribute.

        Args:
            network (nx.Graph): The network graph to visualize.
            color_by (str, optional): Attribute to color edges by. Defaults to "effect".
        """
        super().__init__(network)
        self.color_by = color_by
        self.edge_colors = _styles.get_styles()['default']['edges']

    def set_custom_edge_colors(self, custom_edge_colors):
        """
        Sets custom colors for edges.

        Args:
            custom_edge_colors (dict): Dictionary of custom edge colors.
        """
        self.edge_colors.update(custom_edge_colors)

    def color_nodes(self):
        """
        Colors nodes based on their type. Retrieves colors from the style dictionary and assigns them to nodes.
        """
        styles = _styles.get_styles()
        default_node_styles = styles['default']['nodes']
        source_node_color = default_node_styles['sources']['fillcolor']
        target_node_color = default_node_styles['targets']['fillcolor']
        default_color = default_node_styles['default']['fillcolor']

        for node in self.network.nodes:
            node_data = self.network.nodes[node]
            node_data['color'] = default_color

            if node_data.get("type") == "source":
                node_data['color'] = source_node_color
            elif node_data.get("type") == "target":
                node_data['color'] = target_node_color

    def color_edges(self):
        """
        Colors edges based on a specified attribute. Uses colors from the style dictionary.
        """
        edge_colors = _styles.get_styles()['default']['edges']
        for edge in self.network.edges():
            u, v = edge
            edge_data = self.network.get_edge_data(u, v)
            if self.color_by in edge_data:
                color = edge_colors.get(edge_data[self.color_by], edge_colors['default'])
            else:
                color = edge_colors['default']
            edge_data['color'] = color

    def visualize_network_default(self,
                                  network,
                                  source_dict,
                                  target_dict,
                                  prog='dot',
                                  custom_style=None):
        """
        Visualizes the network using default styles.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): Sources and perturbation signs.
            target_dict (dict): Targets and measurement signs.
            prog (str, optional): Layout program to use. Defaults to 'dot'.
            custom_style (dict, optional): Custom style dictionary to apply.

        Returns:
            A (pygraphviz.AGraph): The visualized network graph.
        """
        default_style = _styles.get_styles()['default']
        style = _styles.merge_styles(default_style, custom_style)

        A = nx.nx_agraph.to_agraph(network)
        A.graph_attr['ratio'] = '1.2'

        sources = set(source_dict.keys())
        targets = set(target_dict.keys())

        for node in A.nodes():
            n = node.get_name()
            if n in sources:
                base_style = style['nodes']['sources']
            elif n in targets:
                base_style = style['nodes']['targets']
            else:
                base_style = style['nodes']['other']

            _styles.set_style_attributes(node, base_style)

        for edge in A.edges():
            edge_style = style['edges']['neutral']
            _styles.set_style_attributes(edge, edge_style)

        A.layout(prog=prog)
        return A

    def visualize_network_sign_consistent(self,
                                          network,
                                          source_dict,
                                          target_dict,
                                          prog='dot',
                                          custom_style=None):
        """
        Visualizes the network considering sign consistency.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): Sources and perturbation signs.
            target_dict (dict): Targets and measurement signs.
            prog (str, optional): Layout program to use. Defaults to 'dot'.
            custom_style (dict, optional): Custom style dictionary to apply.

        Returns:
            A (pygraphviz.AGraph): The visualized network graph.
        """
        default_style = _styles.get_styles()['sign_consistent']
        style = _styles.merge_styles(default_style, custom_style)

        A = self.visualize_network_default(network, source_dict, target_dict, prog, style)

        sources = set(source_dict.keys())
        target_dict_flat = {sub_key: sub_value for key, value in target_dict.items()
                            for sub_key, sub_value in value.items()}
        targets = set(target_dict_flat.keys())

        for node in A.nodes():
            n = node.get_name()
            condition_style = None
            sign_value = target_dict_flat.get(n, 1)

            if n in sources:
                nodes_type = "sources"
            elif n in targets:
                nodes_type = "targets"

            if sign_value > 0:
                condition_style = style['nodes'][nodes_type].get('positive_consistent')
            elif sign_value < 0:
                condition_style = style['nodes'][nodes_type].get('negative_consistent')

            if condition_style:
                _styles.set_style_attributes(node, {}, condition_style)

        for edge in A.edges():
            u, v = edge
            edge_data = network.get_edge_data(u, v)
            if 'interaction' in edge_data:
                edge_data['sign'] = edge_data.pop('interaction')

            if edge_data['sign'] == 1:
                edge_style = style['edges']['positive']
            elif edge_data['sign'] == -1:
                edge_style = style['edges']['negative']
            else:
                edge_style = style['edges']['neutral']

            _styles.set_style_attributes(edge, edge_style)

        return A

    def visualize_network(self,
                          network,
                          source_dict,
                          target_dict,
                          prog='dot',
                          network_type='default',
                          custom_style=None):
        """
        Main function for network visualization based on network type.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): Sources and perturbation signs.
            target_dict (dict): Targets and measurement signs.
            prog (str, optional): Layout program to use. Defaults to 'dot'.
            network_type (str, optional): Type of visualization. Defaults to "default".
            custom_style (dict, optional): Custom style dictionary to apply.

        Returns:
            A (pygraphviz.AGraph): The visualized network graph.
        """
        if network_type == 'sign_consistent':
            return self.visualize_network_sign_consistent(network, source_dict, target_dict, prog, custom_style)
        else:
            default_style = _styles.get_styles().get(network_type, 'default')
            return self.visualize_network_default(network, source_dict, target_dict, prog, custom_style)

    def visualize(self, output_file='network.png', render=False, highlight_nodes=None, style=None):
        """
        Visualizes the network and saves or shows the plot.

        Args:
            output_file (str, optional): Path to save the output image. Defaults to 'network.png'.
            render (bool, optional): If True, displays the plot. If False, saves it to a file. Defaults to False.
            highlight_nodes (list, optional): List of nodes to highlight. Defaults to None.
            style (dict, optional): Style dictionary for visualization. Defaults to None.

        """
        plt.figure(figsize=(12, 12))
        network = self.network
        pos = nx.spring_layout(network)

        if not all('color' in network.nodes[node] for node in network.nodes):
            self.color_nodes()
        if not all('color' in network.edges[edge] for edge in network.edges):
            self.color_edges()

        node_colors = [network.nodes[node]['color'] for node in network.nodes]
        edge_colors = [network.edges[edge]['color'] for edge in network.edges]

        nx.draw(network, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)

        if highlight_nodes:
            if style and style.get('highlight_color'):
                highlight_color = style['highlight_color']
            else:
                highlight_color = self.style['nodes']['default']['fillcolor']
            highlight_nodes = [_aux.wrap_node_name(node) for node in highlight_nodes]
            nx.draw_networkx_nodes(self.network, pos, nodelist=highlight_nodes, node_color=highlight_color)

        if render:
            plt.show()
        else:
            plt.savefig(output_file)
            plt.close()
