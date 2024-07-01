import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from yfiles_jupyter_graphs import GraphWidget
from IPython.display import display
from typing import Dict, List
from pypath.utils import mapping
from _aux import wrap_node_name




class YFilesVisualizer:

    def __init__(self, network):
        self.network = network.copy()

    def yfiles_visual(
            self,
            graph_layout,
            directed,
    ):
        # creating empty object for visualization
        w = GraphWidget()

        # filling w with nodes
        objects = []
        for idx, item in self.dataframe_nodes.iterrows():
            obj = {
                "id": self.dataframe_nodes["Uniprot"].loc[idx],
                "properties": {"label": self.dataframe_nodes["Genesymbol"].loc[idx]},
                "color": "#ffffff",
                "styles": {"backgroundColor": "#ffffff"}
            }
            objects.append(obj)
        w.nodes = objects

        # filling w with edges
        objects = []
        for index, row in self.dataframe_edges.iterrows():
            obj = {
                "id": self.dataframe_edges["Effect"].loc[index],
                "start": self.dataframe_edges["source"].loc[index],
                "end": self.dataframe_edges["target"].loc[index],
                "properties": {"references": self.dataframe_edges["References"].loc[index]}}
            objects.append(obj)
        w.edges = objects

        def custom_edge_color_mapping(edge: Dict):
            """let the edge be red if the interaction is an inhibition, else green"""
            return ("#fa1505" if edge['id'] == "inhibition" else "#05e60c")

        w.set_edge_color_mapping(custom_edge_color_mapping)

        def custom_node_color_mapping(node: Dict):
            return {"color": "#ffffff"}

        w.set_node_styles_mapping(custom_node_color_mapping)

        def custom_factor_mapping(node: Dict):
            """choose random factor"""
            return 5

        w.set_node_scale_factor_mapping(custom_factor_mapping)

        def custom_label_styles_mapping(node: Dict):
            """let the label be the negated purple big index"""
            return {
                'text': node["properties"]["label"],
                'backgroundColor': None,
                'fontSize': 40,
                'color': '#030200',
                'shape': 'round-rectangle',
                'textAlignment': 'center'
            }

        w.set_node_label_mapping(custom_label_styles_mapping)

        w.directed = directed
        w.graph_layout = graph_layout

        display(w)

    def vis_comparison(
            self,
            int_comparison,
            node_comparison,
            graph_layout,
            directed,
    ):
        # creating empty object for visualization
        w = GraphWidget()

        objects = []
        for idx, item in node_comparison.iterrows():
            obj = {
                "id": node_comparison["node"].loc[idx],
                "properties": {"label": node_comparison["node"].loc[idx],
                               "comparison": node_comparison["comparison"].loc[idx], },
                "color": "#ffffff",
                #       "styles":{"backgroundColor":"#ffffff"}
            }
            objects.append(obj)
        w.nodes = objects

        # filling w with edges
        objects = []
        for index, row in int_comparison.iterrows():
            obj = {
                "id": int_comparison["comparison"].loc[index],
                "properties": {
                    "comparison": int_comparison["comparison"].loc[index]},
                "start": int_comparison["source"].loc[index],
                "end": int_comparison["target"].loc[index]
            }
            objects.append(obj)
        w.edges = objects

        def custom_node_color_mapping(node: Dict):
            if node['properties']['comparison'] == "Unique to Network 1":
                return {"color": "#f5f536"}
            elif node['properties']['comparison'] == "Unique to Network 2":
                return {"color": "#36f55f"}
            elif node['properties']['comparison'] == "Common":
                return {"color": "#3643f5"}

        w.set_node_styles_mapping(custom_node_color_mapping)

        def custom_factor_mapping(node: Dict):
            """choose random factor"""
            return 5

        w.set_node_scale_factor_mapping(custom_factor_mapping)

        def custom_label_styles_mapping(node: Dict):
            """let the label be the negated purple big index"""
            return {
                'text': node["id"],
                'backgroundColor': None,
                'fontSize': 20,
                'color': '#030200',
                'position': 'center',
                'maximumWidth': 130,
                'wrapping': 'word',
                'textAlignment': 'center'
            }

        w.set_node_label_mapping(custom_label_styles_mapping)

        def custom_edge_color_mapping(edge: Dict):
            if edge['id'] == "Unique to Network 1":
                return "#e3941e"
            elif edge['id'] == "Unique to Network 2":
                return "#36f55f"
            elif edge['id'] == "Common":
                return "#3643f5"
            elif edge['id'] == "Conflicting":
                return "#ffcc00"

        w.set_edge_color_mapping(custom_edge_color_mapping)

        w.directed = directed
        w.graph_layout = graph_layout

        display(w)


# Example usage
# network is assumed to be a custom object with edges, nodes, and initial_nodes attributes.
# Here is a mock example of how to create such an object. You should replace it with your actual network data.

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
# visualizer = NetworkXVisualizer(network)
# visualizer.render(view=True)
