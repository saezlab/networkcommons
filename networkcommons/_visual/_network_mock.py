from typing import Dict, List
import networkx as nx
import pandas as pd
from pypath.utils import mapping

import vis_yfiles
import vis_networkx


class NetworkMock:

    def __init__(self):
        self.network = nx.DiGraph()
        self.color_map = {}

    def init_from_sif(self, filepath):
        """
        Initialize the network from a SIF file
        """
        with open(filepath, 'r') as f:
            for line in f:
                source, effect, target = line.strip().split()
                self.network.add_edge(source, target)
                self.update_edge_property((source, target), type="effect", value=effect)


        return self.network

    def add_nodes(self, nodes):
        self.network.add_nodes_from(nodes)
        return self.network

    def set_source_nodes(self, source_nodes):
        self.update_node_property(source_nodes, type="source", value="1")

    def set_target_nodes(self, target_nodes):
        self.update_node_property(target_nodes, type="target", value="1")

    def set_nodes_of_interest(self, nodes):
        self.update_node_property(nodes, type="noi", value="1")

    def add_edges(self, edges):
        self.network.add_edges_from(edges)

    def update_node_property(self, node, type="color", value="blue"):
        # add color to the node in networkx
        if type == "color":
            self.color_map[node] = value
        elif type == "initial_node":
            self.network.nodes[node]["initial_node"] = True
        return self.color_map

    def update_edge_property(self, edge, type="color", value="blue"):
        # add color to the edge in networkx
        if type == "color":
            self.color_map[edge] = value
        elif type == "effect":
            self.network.edges[edge]["effect"] = value
        else:
            raise ValueError("Invalid edge property")
        return self.color_map

    def visualise(self, filepath=None, render=False, visualiser="networkx"):
        if visualiser == "yfiles":
            yfiles_vis = vis_yfiles.YFilesVisualizer(self.network)
            yfiles_vis.visualise(graph_layout="hierarchical", directed=True)
        elif visualiser == "networkx":
            networkx_vis = vis_networkx.NetworkXVisualizer(self.network)
            networkx_vis.visualise(render=render)
        else:
            raise ValueError("Invalid visualiser")

    def mapping_node_identifier(self, node: str) -> List[str]:
        complex_string = None
        gene_symbol = None
        uniprot = None

        if mapping.id_from_label0(node):
            uniprot = mapping.id_from_label0(node)
            gene_symbol = mapping.label(uniprot)
        elif node.startswith("COMPLEX"):
            node = node[8:] #TODO change to wrap
            node_list = node.split("_")
            translated_node_list = [mapping.label(mapping.id_from_label0(item)) for item in node_list]
            complex_string = "COMPLEX:" + "_".join(translated_node_list)
        elif mapping.label(node):
            gene_symbol = mapping.label(node)
            uniprot = mapping.id_from_label0(gene_symbol)
        else:
            print("Error during translation, check syntax for ", node)

        return [complex_string, gene_symbol, uniprot]

    def convert_edges_into_genesymbol(self, edges: pd.DataFrame) -> pd.DataFrame:
        def convert_identifier(x):
            identifiers = self.mapping_node_identifier(x)
            return identifiers[1]  # Using GeneSymbol
        edges["source"] = edges["source"].apply(convert_identifier)
        edges["target"] = edges["target"].apply(convert_identifier)
        return edges


