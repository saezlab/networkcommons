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
- `NetworkXVisualizer`: An advanced visualizer that extends `NetworkVisualizerBase`
    with additional features for network visualization.

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
        visualize_graph_simple(source_dict, target_dict, prog, is_sign_consistent, max_nodes):
            Visualizes the network with basic settings and styling.
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

    def visualize_network_simple(self,
                                 source_dict,
                                 target_dict,
                                 prog='dot',
                                 is_sign_consistent=True,
                                 max_nodes=75):
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
            _log("The network is too large to visualize, you can increase the max_nodes parameter if needed.")
            print("The network is too large to visualize, you can increase the max_nodes parameter if needed.")
            return None
        if len(self.network.nodes) == 0:
            _log("The network is empty, nothing to visualize.")
            print("The network is empty, nothing to visualize.")
            return None

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
                base_style = self.style['nodes']['other']['default']

            _styles.set_style_attributes(node, base_style)

        for edge in A.edges():
            u, v = edge
            edge_data = self.network.get_edge_data(u, v)
            if not edge_data:
                _log(f"Edge data not found for edge {u} -> {v}.")
                edge_style = self.style['edges']['default']
            else:
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
        visualize_network_default(network, source_dict, target_dict, prog, custom_style):
            Visualizes the network using default styles.
        visualize_network_sign_consistent(network, source_dict, target_dict, prog, custom_style):
            Visualizes the network with sign consistency.
        visualize_network(network, source_dict, target_dict, prog, network_type, custom_style):
            Main function for network visualization based on network type.
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
        default_color = default_node_styles['other']['default']['fillcolor']

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
                edge_style = edge_colors.get(edge_data[self.color_by], edge_colors['default'])
                color = edge_style['color'] if isinstance(edge_style, dict) else edge_style
            else:
                color = edge_colors['default']['color'] if isinstance(edge_colors['default'], dict) \
                    else edge_colors['default']

            edge_data['color'] = color

    def adjust_node_labels(self,
                           wrap: bool = True,
                           truncate: bool = False,
                           max_length: int = 6,
                           wrap_length: int = 6):
        """
        Adjust node labels using the adjust_node_name function.

        Args:
            wrap (bool): Whether to wrap the node labels.
            truncate (bool): Whether to truncate the node labels.
            max_length (int): The maximum length of node labels before truncation.
            wrap_length (int): The length at which to wrap the node labels.
        """
        for node in self.network.nodes():
            node_data = self.network.nodes[node]
            adjusted_name = _aux.adjust_node_name(node,
                                                  truncate=truncate,
                                                  wrap=wrap,
                                                  max_length=max_length,
                                                  wrap_length=wrap_length)
            node_data['label'] = adjusted_name


    def visualize_network_default(self,
                                  source_dict,
                                  target_dict,
                                  prog='dot',
                                  custom_style=None,
                                  max_nodes=75,
                                  wrap_names=True):
        """
        Visualizes the network using default styles.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): Sources and perturbation signs.
            target_dict (dict): Targets and measurement signs.
            prog (str, optional): Layout program to use. Defaults to 'dot'.
            custom_style (dict, optional): Custom style dictionary to apply.
            max_nodes (int, optional): Maximum number of nodes to visualize. Defaults to 75.
            wrap_names (bool, optional): Whether to wrap node names. Defaults to True.

        Returns:
            A (pygraphviz.AGraph): The visualized network graph.
        """
        if len(self.network.nodes) > max_nodes:
            _log("The network is too large to visualize, you can increase the max_nodes parameter if needed.")
            print("The network is too large to visualize, you can increase the max_nodes parameter if needed.")
            return None
        if len(self.network.nodes) == 0:
            _log("The network is empty, nothing to visualize.")
            print("The network is empty, nothing to visualize.")
            return None

        default_style = _styles.get_styles()['default']
        style = _styles.merge_styles(default_style, custom_style)

        A = nx.nx_agraph.to_agraph(self.network)
        A.graph_attr['ratio'] = '1.2'

        # Adjust node labels before visualization
        if wrap_names:
            self.adjust_node_labels(wrap=True, truncate=True)

        if source_dict:
            sources = set(source_dict.keys())
        else:
            sources = set()
        if target_dict:
            targets = set(target_dict.keys())
        else:
            targets = set()

        for node in A.nodes():
            n = node.get_name()
            if n in sources:
                base_style = style['nodes']['sources']
            elif n in targets:
                base_style = style['nodes']['targets']
            else:
                base_style = style['nodes']['other']['default']

            _styles.set_style_attributes(node, base_style)

        for edge in A.edges():
            edge_style = style['edges']['neutral']
            _styles.set_style_attributes(edge, edge_style)

        A.layout(prog=prog)
        return A

    def visualize_network_sign_consistent(self,
                                          source_dict,
                                          target_dict,
                                          prog='dot',
                                          custom_style=None,
                                          max_nodes=75):
        """
        Visualizes the network considering sign consistency.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): Sources and perturbation signs.
            target_dict (dict): Targets and measurement signs.
            prog (str, optional): Layout program to use. Defaults to 'dot'.
            custom_style (dict, optional): Custom style dictionary to apply.
            max_nodes (int, optional): Maximum number of nodes to visualize. Defaults to 75.
        Returns:
            A (pygraphviz.AGraph): The visualized network graph.
        """
        if len(self.network.nodes) > max_nodes:
            _log("The network is too large to visualize, you can increase the max_nodes parameter if needed.")
            print("The network is too large to visualize, you can increase the max_nodes parameter if needed.")
            return None
        if len(self.network.nodes) == 0:
            _log("The network is empty, nothing to visualize.")
            print("The network is empty, nothing to visualize.")
            return None

        default_style = _styles.get_styles()['sign_consistent']
        style = _styles.merge_styles(default_style, custom_style)

        A = self.visualize_network_default(source_dict, target_dict, prog=prog, custom_style=style, max_nodes=max_nodes)

        if source_dict:
            sources = set(source_dict.keys())
        else:
            sources = set()
        if target_dict:
            targets = set(target_dict.keys())
        else:
            targets = set()

        for node in A.nodes():
            n = node.get_name()
            condition_style = None
            sign_value = target_dict.get(n, 1)

            if n in sources:
                nodes_type = "sources"
            elif n in targets:
                nodes_type = "targets"
            else:
                nodes_type = "other"

            if sign_value > 0:
                condition_style = style['nodes'][nodes_type].get('positive_consistent')
            elif sign_value < 0:
                condition_style = style['nodes'][nodes_type].get('negative_consistent')

            if condition_style:
                _styles.set_style_attributes(node, {}, condition_style)

        for edge in A.edges():
            u, v = edge
            edge_data = self.network.get_edge_data(u, v)
            if not edge_data:
                _log(f"Edge data not found for edge {u} -> {v}.")
                edge_style = style['edges']['default']
            else:
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
                          source_dict,
                          target_dict,
                          prog='dot',
                          network_type='default',
                          custom_style=None,
                          max_nodes=75):
        """
        Main function for network visualization based on network type.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): Sources and perturbation signs.
            target_dict (dict): Targets and measurement signs.
            prog (str, optional): Layout program to use. Defaults to 'dot'.
            network_type (str, optional): Type of visualization. Defaults to "default".
            custom_style (dict, optional): Custom style dictionary to apply.
            max_nodes (int, optional): Maximum number of nodes to visualize. Defaults to 75.

        Returns:
            A (pygraphviz.AGraph): The visualized network graph.
        """
        if network_type == 'sign_consistent':
            return self.visualize_network_sign_consistent(source_dict, target_dict, prog, custom_style, max_nodes)
        else:
            default_style = _styles.get_styles().get(network_type, 'default')
            return self.visualize_network_default(source_dict, target_dict, prog, custom_style, max_nodes)

    def visualize(self,
                  source_dict=None,
                  target_dict=None,
                  output_file='',
                  prog='dot',
                  render=False,
                  highlight_nodes=None,
                  style=None,
                  max_nodes=75):
        """
        Visualizes the network and saves or shows the plot.

        Args:
            output_file (str, optional): Path to save the output image. Defaults to 'network.png'.
            render (bool, optional): If True, displays the plot. If False, saves it to a file. Defaults to False.
            highlight_nodes (list, optional): List of nodes to highlight. Defaults to None.
            style (dict, optional): Style dictionary for visualization. Defaults to None.
            max_nodes (int, optional): Maximum number of nodes to visualize. Defaults to 75.
        """

        if len(self.network.nodes) > max_nodes:
            _log("The network is too large to visualize, you can increase the max_nodes parameter if needed.")
            print("The network is too large to visualize, you can increase the max_nodes parameter if needed.")
            return None
        if len(self.network.nodes) == 0:
            _log("The network is empty, nothing to visualize.")
            print("The network is empty, nothing to visualize.")
            return None

        if style:
            self.set_custom_style(style)
        A = self.visualize_network(source_dict, target_dict, prog=prog, custom_style=style, max_nodes=max_nodes)

        # Highlight specific nodes if provided
        if highlight_nodes:
            for node in A.nodes():
                if node in highlight_nodes:
                    highlight_color = style['highlight_color'] if style and 'highlight_color' in style else \
                        self.style['nodes']['other']['default']['fillcolor']
                    A.get_node(node).attr['fillcolor'] = highlight_color
                    A.get_node(node).attr['style'] = 'filled'

        # Save or render the plot
        if render:
            img_data = A.draw(format='png', prog=prog)  # Get image data in-memory
            plt.imshow(plt.imread(img_data))  # Render the image from the in-memory data
            plt.axis('off')
            plt.show()
        elif output_file:
            A.draw(output_file, format='png', prog='dot')
            _log(f"Network visualization saved to {output_file}.")
            print(f"Network visualization saved to {output_file}.")
        else:
            _log("No output file specified. Set 'output_file' to save the visualization.")
            print("No output file specified. Set 'output_file' to save the visualization.")
        return A