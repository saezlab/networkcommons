from yfiles_jupyter_graphs import GraphWidget
from typing import Dict
from IPython.display import display
from networkcommons._session import _log

from networkcommons._visual.yfiles_styles import (get_styles,
                                                  apply_node_style,
                                                  apply_edge_style,
                                                  update_node_property,
                                                  update_edge_property,
                                                  get_edge_color,
                                                  get_comparison_color)


class YFilesVisualizer:

    def __init__(self, network):
        self.network = network.copy()
        self.styles = get_styles()

    def visualize(self, graph_layout="Organic", directed=True):
        available_layouts = ["Circular", "Hierarchic", "Organic", "Orthogonal", "Radial", "Tree"]
        if graph_layout not in available_layouts:
            graph_layout = "Organic"
            _log.warning(f"Graph layout not available. Using default layout: {graph_layout}")

        # creating empty object for visualization
        w = GraphWidget()

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
            styled_edge = {"color": get_edge_color(edge[2]['sign'], self.styles)}
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

        display(w)

    def vis_comparison(self, int_comparison, node_comparison, graph_layout, directed):
        # creating empty object for visualization
        w = GraphWidget()

        objects = []
        for idx, item in node_comparison.iterrows():
            node_props = {"label": item["node"], "comparison": item["comparison"]}
            node = apply_node_style(node_props, self.styles['default']['nodes'])
            node = update_node_property(node, type="color",
                                        value=get_comparison_color(item["comparison"], self.styles))
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
            edge = apply_edge_style(edge_props, self.styles['default']['edges'])
            edge = update_edge_property(edge, type="color",
                                        value=get_comparison_color(row["comparison"], self.styles))
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
        return 1

    @staticmethod
    def custom_label_styles_mapping(node: Dict):
        label_style = get_styles()['default']['labels'].copy()
        label_style['text'] = node["properties"]["label"]
        return label_style
