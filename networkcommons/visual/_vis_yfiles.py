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
Interactive network visualization using the yFiles HTML widget.

This module provides functionality to visualize networks interactively using the yFiles HTML widget.

Classes:
- `YFilesVisualizer`: Visualizes network graphs using the yFiles library with various layout options and styles.

Methods:
- `__init__(self, network)`: Initializes the visualizer with a network graph.
- `visualize(self, graph_layout="Organic", directed=True)`: Visualizes the network with specified layout and direction.
- `vis_comparison(self, int_comparison, node_comparison, graph_layout, directed)`: Visualizes network comparison data.
- `custom_edge_color_mapping(edge: dict)`: Custom function to map edge colors.
- `custom_node_color_mapping(node: dict)`: Custom function to map node colors.
- `custom_factor_mapping(node: dict)`: Custom function to determine node scale factors.
- `custom_label_styles_mapping(node: dict)`: Custom function to map node label styles.
"""

from __future__ import annotations

__all__ = ['YFilesVisualizer']

import lazy_import

from IPython.display import display
import yfiles_jupyter_graphs as yfiles

from . import _yfiles_styles
from networkcommons._session import _log


class YFilesVisualizer:
    """
    A class for visualizing network graphs interactively using the yFiles HTML widget.

    Attributes:
        network (nx.Graph): The network graph to visualize.
        styles (dict): A dictionary of styles for visualizing nodes and edges.

    Methods:
        __init__(network): Initializes the visualizer with a network graph.
        visualize(graph_layout="Organic", directed=True): Visualizes the network with a specified layout and direction.
        vis_comparison(int_comparison, node_comparison, graph_layout, directed): Visualizes network comparison data with specific layout and direction.
        custom_edge_color_mapping(edge): Custom function to map edge colors.
        custom_node_color_mapping(node): Custom function to map node colors.
        custom_factor_mapping(node): Custom function to determine node scale factors.
        custom_label_styles_mapping(node): Custom function to map node label styles.
    """

    def __init__(self, network):
        """
        Initializes the visualizer with a network graph and default styles.

        Args:
            network (nx.Graph): The network graph to visualize.
        """
        self.network = network.copy()
        self.styles = _yfiles_styles.get_styles()

    def visualize(self, graph_layout="Organic", directed=True):
        """
        Visualizes the network graph using the yFiles widget with the specified layout and direction.

        Args:
            graph_layout (str, optional): The layout to use for the graph. Options include "Circular", "Hierarchic", "Organic", "Orthogonal", "Radial", and "Tree". Defaults to "Organic".
            directed (bool, optional): If True, visualizes the graph as directed. Defaults to True.

        Displays:
            An interactive visualization of the network graph.
        """
        available_layouts = ["Circular", "Hierarchic", "Organic", "Orthogonal", "Radial", "Tree"]
        if graph_layout not in available_layouts:
            graph_layout = "Organic"
            _log.warning(f"Graph layout not available. Using default layout: {graph_layout}")

        # Creating empty widget for visualization
        w = yfiles.GraphWidget()

        # Adding nodes to the widget
        node_objects = []
        for node in self.network.nodes:
            styled_node = self.styles['default']['nodes']
            node_objects.append({
                "id": node,
                "properties": {'label': node},
                "color": styled_node['color'],
                "styles": {"backgroundColor": styled_node['fillcolor']}
            })
        w.nodes = node_objects

        # Adding edges to the widget
        edge_objects = []
        for edge in self.network.edges(data=True):
            styled_edge = self.styles['default']['edges']
            edge_color = _yfiles_styles.get_edge_color(edge[2].get('sign', 1), self.styles)
            edge_objects.append({
                "id": (edge[0], edge[1]),
                "start": edge[0],
                "end": edge[1],
                "properties": {"color": edge_color}
            })
        w.edges = edge_objects

        # Setting custom mappings
        w.set_edge_color_mapping(self.custom_edge_color_mapping)
        w.set_node_styles_mapping(self.custom_node_color_mapping)
        w.set_node_scale_factor_mapping(self.custom_factor_mapping)
        w.set_node_label_mapping(self.custom_label_styles_mapping)

        # Applying layout and direction
        w.directed = directed
        w.graph_layout = graph_layout

        # Displaying the widget
        display(w)

    def vis_comparison(self, int_comparison, node_comparison, graph_layout, directed):
        """
        Visualizes network comparison data with specific layout and direction.

        Args:
            int_comparison (pd.DataFrame): DataFrame containing interaction comparison data.
            node_comparison (pd.DataFrame): DataFrame containing node comparison data.
            graph_layout (str): The layout to use for the graph.
            directed (bool): If True, visualizes the graph as directed.

        Displays:
            An interactive visualization of the network comparison.
        """
        # Creating empty widget for visualization
        w = yfiles.GraphWidget()

        # Adding nodes with comparison data
        node_objects = []
        for idx, item in node_comparison.iterrows():
            node_props = {"label": item["node"], "comparison": item["comparison"]}
            node = _yfiles_styles.apply_node_style(node_props, self.styles['default']['nodes'])
            node = _yfiles_styles.update_node_property(node, "color", _yfiles_styles.get_comparison_color(item["comparison"], self.styles))
            node_objects.append({
                "id": item["node"],
                "properties": node,
                "color": node['color'],
                "styles": {"backgroundColor": node['fillcolor']}
            })
        w.nodes = node_objects

        # Adding edges with comparison data
        edge_objects = []
        for index, row in int_comparison.iterrows():
            edge_props = {"comparison": row["comparison"]}
            edge = _yfiles_styles.apply_edge_style(edge_props, self.styles['default']['edges'])
            edge = _yfiles_styles.update_edge_property(edge, "color", _yfiles_styles.get_comparison_color(row["comparison"], self.styles))
            edge_objects.append({
                "id": (row["source"], row["target"]),
                "start": row["source"],
                "end": row["target"],
                "properties": edge
            })
        w.edges = edge_objects

        # Setting custom mappings
        w.set_edge_color_mapping(self.custom_edge_color_mapping)
        w.set_node_styles_mapping(self.custom_node_color_mapping)
        w.set_node_scale_factor_mapping(self.custom_factor_mapping)
        w.set_node_label_mapping(self.custom_label_styles_mapping)

        # Applying layout and direction
        w.directed = directed
        w.graph_layout = graph_layout

        # Displaying the widget
        display(w)

    @staticmethod
    def custom_edge_color_mapping(edge: dict):
        """
        Custom function to map edge colors based on edge properties.

        Args:
            edge (dict): Dictionary containing edge properties.

        Returns:
            str: Color for the edge.
        """
        return edge["properties"]["color"]

    @staticmethod
    def custom_node_color_mapping(node: dict):
        """
        Custom function to map node colors based on node properties.

        Args:
            node (dict): Dictionary containing node properties.

        Returns:
            dict: Dictionary containing the node color.
        """
        return {"color": node["color"]}

    @staticmethod
    def custom_factor_mapping(node: dict):
        """
        Custom function to determine the scale factor for nodes.

        Args:
            node (dict): Dictionary containing node properties.

        Returns:
            int: Scale factor for the node.
        """
        return 1

    @staticmethod
    def custom_label_styles_mapping(node: dict):
        """
        Custom function to map node label styles based on node properties.

        Args:
            node (dict): Dictionary containing node properties.

        Returns:
            dict: Dictionary containing label styles for the node.
        """
        label_style = _yfiles_styles.get_styles()['default']['labels'].copy()
        label_style['text'] = node["properties"]["label"]
        return label_style
