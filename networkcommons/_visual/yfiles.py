from yfiles_jupyter_graphs import GraphWidget
from typing import Dict
from IPython.display import display
from _aux import wrap_node_name
import pandas as pd

from yfiles_styles import (get_styles,
                           apply_node_style,
                           apply_edge_style,
                           #set_custom_node_color,
                           #set_custom_edge_color,
                           get_edge_color, get_comparison_color)


class YFilesVisualizer:

    def __init__(self, network):
        self.network = network.copy()
        self.styles = get_styles()

    def yfiles_visual(self, graph_layout, directed):
        # creating empty object for visualization
        w = GraphWidget()

        # filling w with nodes
        objects = []

        for node in self.network.nodes:
            node_props = {"label": node}
            node = apply_node_style(node_props, self.styles['default']['nodes'])
            objects.append({
                "id": node,
                "properties": node,
                "color": node['color'],
                "styles": {"backgroundColor": node['fillcolor']}
            })

        w.nodes = objects

        # filling w with edges
        objects = []

        for edge in self.network.edges:
            edge_props = {"color": get_edge_color(edge[2]['effect'], self.styles)}
            edge = apply_edge_style(edge_props, self.styles['default']['edges'])
            objects.append({
                "id": edge,
                "start": edge[0],
                "end": edge[1],
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

    def vis_comparison(self, int_comparison, node_comparison, graph_layout, directed):
        # creating empty object for visualization
        w = GraphWidget()

        objects = []
        for idx, item in node_comparison.iterrows():
            node_props = {"label": item["node"], "comparison": item["comparison"]}
            node = apply_node_style(node_props, self.styles['default']['nodes'])
            node = set_custom_node_color(node, get_comparison_color(item["comparison"], self.styles, 'nodes'))
            objects.append({
                "id": item["node"],
                "properties": node,
                "color": node['color'],
                "styles": {"backgroundColor": node['fillcolor']}
            })
        w.nodes = objects

        # filling w with edges
        objects = []
        for index, row in int_comparison.iterrows():
            edge_props = {"comparison": row["comparison"]}
            edge = apply_edge_style(edge_props, self.styles['default']['edges'])
            edge = set_custom_edge_color(edge, get_comparison_color(row["comparison"], self.styles, 'edges'))
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
    def custom_edge_color_mapping(edge: Dict):
        return edge["properties"]["color"]

    @staticmethod
    def custom_node_color_mapping(node: Dict):
        return {"color": node["color"]}

    @staticmethod
    def custom_factor_mapping(node: Dict):
        return 5

    @staticmethod
    def custom_label_styles_mapping(node: Dict):
        label_style = get_styles()['default']['labels'].copy()
        label_style['text'] = node["properties"]["label"]
        return label_style


# Example usage
# class MockNetwork:
#     def __init__(self):
#         self.edges = pd.DataFrame({
#             'source': ['A', 'B', 'C'],
#             'target': ['B', 'C', 'D'],
#             'Effect': ['stimulation', 'inhibition', 'form complex']
#         })
#         self.nodes = pd.DataFrame({
#             'Genesymbol': ['A', 'B', 'C', 'D'],
#             'Uniprot': ['P1', 'P2', 'P3', 'P4']
#         })
#         self.initial_nodes = ['A', 'B']
#
# network = MockNetwork()
# visualizer = YFilesVisualizer(network)
# visualizer.yfiles_visual(graph_layout="hierarchic", directed=True)
