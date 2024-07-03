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
"""

from __future__ import annotations

__all__ = ['YFilesVisualizer']

import lazy_import
yfiles = lazy_import.lazy_module('yfiles_jupyter_graphs')
ipydisplay = lazy_import.lazy_module('IPython.display')

from . import _yfiles_styles
from networkcommons._session import _log


class YFilesVisualizer:


    def __init__(self, network):

        self.network = network.copy()
        self.styles = _yfiles_styles.get_styles()


    def visualize(self, graph_layout="Organic", directed=True):

        available_layouts = ["Circular", "Hierarchic", "Organic", "Orthogonal", "Radial", "Tree"]
        if graph_layout not in available_layouts:
            graph_layout = "Organic"
            _log.warning(f"Graph layout not available. Using default layout: {graph_layout}")

        # creating empty object for visualization
        w = yfiles.GraphWidget()

        # filling w with nodes
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

        # filling w with edges
        edge_objects = []

        for edge in self.network.edges(data=True):

            styled_edge = self.styles['default']['edges']
            styled_edge = {"color": _yfiles_styles.get_edge_color(edge[2]['sign'], self.styles)}
            edge_objects.append({
                "id": (edge[0], edge[1]),
                "start": edge[0],
                "end": edge[1],
                "properties": styled_edge
            })

        w.edges = edge_objects

        w.set_edge_color_mapping(self.custom_edge_color_mapping)
        w.set_node_styles_mapping(self.custom_node_color_mapping)
        w.set_node_scale_factor_mapping(self.custom_factor_mapping)
        w.set_node_label_mapping(self.custom_label_styles_mapping)

        w.directed = directed
        w.graph_layout = graph_layout

        ipydisplay.display(w)

    def vis_comparison(self, int_comparison, node_comparison, graph_layout, directed):
        # creating empty object for visualization
        w = yfiles.GraphWidget()

        objects = []
        for idx, item in node_comparison.iterrows():
            node_props = {"label": item["node"], "comparison": item["comparison"]}
            node = _yfiles_styles.apply_node_style(node_props, self.styles['default']['nodes'])
            node = _yfiles_styles.update_node_property(
                node,
                type="color",
                value=_yfiles_styles.get_comparison_color(item["comparison"], self.styles),
            )
            objects.append({
                "id": item["node"],
                "properties": node,
                "color": node['color'],
                "styles": {"backgroundColor": node['fillcolor']}
            })
        w.nodes = objects

        objects = []
        for index, row in int_comparison.iterrows():
            edge_props = {"comparison": row["comparison"]}
            edge = _yfiles_styles.apply_edge_style(edge_props, self.styles['default']['edges'])
            edge = _yfiles_styles.update_edge_property(
                edge,
                type="color",
                value=_yfiles_styles.get_comparison_color(row["comparison"], self.styles),
            )
            objects.append({
                "id": row["comparison"],
                "start": row["source"],
                "end": row["target"],
                "properties": edge
            })
        w.edges = objects

        w.set_edge_color_mapping(self.custom_edge_color_mapping)
        w.set_node_styles_mapping(self.custom_node_color_mapping)
        w.set_node_scale_factor_mapping(self.custom_factor_mapping)
        w.set_node_label_mapping(self.custom_label_styles_mapping)

        w.directed = directed
        w.graph_layout = graph_layout

        display(w)

    @staticmethod
    def custom_edge_color_mapping(edge: dict):
        return edge["properties"]["color"]

    @staticmethod
    def custom_node_color_mapping(node: dict):
        return {"color": node["color"]}

    @staticmethod
    def custom_factor_mapping(node: dict):
        return 1

    @staticmethod
    def custom_label_styles_mapping(node: dict):
        label_style = _yfiles_styles.get_styles()['default']['labels'].copy()
        label_style['text'] = node["properties"]["label"]
        return label_style
