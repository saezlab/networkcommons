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
Visualize networks.

The styles for different types of networks are defined in the `get_styles()`
function.  The `set_style_attributes()` function is used to set attributes for
nodes and edges based on the given styles.  The `visualize_network_default()`
function visualizes the graph with default style.  The
`visualize_network_sign_consistent()` function visualizes the graph considering
sign consistency.  The `visualize_network()` function is the main function to
visualize the graph based on the network type.
"""

from __future__ import annotations

__all__ = ['NetworkXVisualizer']

import lazy_import
import networkx as nx
plt = lazy_import.lazy_module('matplotlib.pyplot')

from . import _aux
from . import _styles
from networkcommons._session import _log


class NetworkXVisualizer:

    def __init__(self, network, color_by="effect"):
        self.network = network.copy()
        self.color_by = color_by
        self.edge_colors = _styles.get_styles()['default']['edges']

    def set_custom_edge_colors(self, custom_edge_colors):
        self.edge_colors.update(custom_edge_colors)

    def color_nodes(self):
        default_node_colors = _styles.get_styles()['default']['nodes']['default']
        source_node_color = default_node_colors['sources']
        target_node_color = default_node_colors['targets']
        nodes = self.network.nodes
        for node in nodes:
            nodes[node]['color'] = default_node_colors

            if nodes[node].get("type") == "source":
                nodes[node]['color'] = source_node_color

            if nodes[node].get("type") == "target":
                nodes[node]['color'] = target_node_color

    def color_edges(self):
        edge_colors = _styles.get_styles()['default']['edges']
        for edge in self.network.edges:
            u, v = edge
            edge_data = self.network.get_edge_data(u, v)
            if self.color_by in edge_data:
                color = edge_colors[edge_data[self.color_by]]
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
        Core function to visualize the graph.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): A dictionary containing the sources and sign of perturbation.
            target_dict (dict): A dictionary containing the targets and sign of measurements.
            prog (str, optional): The layout program to use. Defaults to 'dot'.
            custom_style (dict, optional): The custom style to apply. If None, the default style is used.
        """
        default_style = _styles.get_styles()['default']
        style = _styles.merge_styles(default_style, custom_style)

        A = nx.nx_agraph.to_agraph(network)
        A.graph_attr['ratio'] = '1.2'

        sources = set(source_dict.keys())
        target_dict_flat = {sub_key: sub_value for key, value in target_dict.items()
                            for sub_key, sub_value in value.items()}
        targets = set(target_dict_flat.keys())

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
        Visualize the graph considering sign consistency.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): A dictionary containing the sources and sign of perturbation.
            target_dict (dict): A dictionary containing the targets and sign of measurements.
            prog (str, optional): The layout program to use. Defaults to 'dot'.
            custom_style (dict, optional): The custom style to apply. Defaults to None.
        """
        default_style = _styles.get_styles()['sign_consistent']
        style = _styles.merge_styles(default_style, custom_style)

        # Call the core visualization function
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
                # Apply condition style without overwriting base style
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

            s_styles.et_style_attributes(edge, edge_style)

        return A

    def visualize_network(self,
                          network,
                          source_dict,
                          target_dict,
                          prog='dot',
                          network_type='default',
                          custom_style=None):
        """
        Main function to visualize the graph based on the network type.

        Args:
            network (nx.Graph): The network to visualize.
            source_dict (dict): A dictionary containing the sources and sign of perturbation.
            target_dict (dict): A dictionary containing the targets and sign of measurements.
            prog (str, optional): The layout program to use. Defaults to 'dot'.
            network_type (str, optional): The type of visualization to use. Defaults to "default".
            custom_style (dict, optional): The custom style to apply. Defaults to None.
        """
        if network_type == 'sign_consistent':
            return self.visualize_network_sign_consistent(network, source_dict, target_dict, prog, custom_style)
        else:
            default_style = _styles.get_styles().get(network_type, get_styles()['default'])
            return self.visualize_network_default(network, source_dict, target_dict, prog, custom_style)

    def visualize(self, output_file='network.png', render=False, highlight_nodes=None, style=None):
        plt.figure(figsize=(12, 12))
        network = self.network
        pos = nx.spring_layout(network)

        node_colors = [network.nodes[node]['fillcolor'] for node in network.nodes]
        edge_colors = [network.edges[edge]['color'] for edge in network.edges]

        nx.draw(network, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)

        if highlight_nodes:
            if style.get('highlight_color'):
                highlight_color = style['highlight_color']
            else:
                highlight_color = style['nodes']['default']['fillcolor']
            highlight_nodes = [_aux.wrap_node_name(node) for node in highlight_nodes]
            nx.draw_networkx_nodes(self.network, pos, nodelist=highlight_nodes, node_color=highlight_color)

        if render:
            plt.show()
        else:
            plt.savefig(output_file)
            plt.close()
