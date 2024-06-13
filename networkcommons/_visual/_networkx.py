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
This module contains functions to visualize networks.
The styles for different types of networks are defined in the get_styles() function.
The set_style_attributes() function is used to set attributes for nodes and edges based on the given styles.
The visualize_network_default() function visualizes the graph with default style.
The visualize_network_sign_consistent() function visualizes the graph considering sign consistency.
The visualize_network() function is the main function to visualize the graph based on the network type.
"""

import networkx as nx

from networkcommons._session import _log

__all__ = [
    'get_styles',
    'set_style_attributes',
    'merge_styles',
    'visualize_network',
    'visualize_network_default',
    'visualize_network_sign_consistent',
    'visualize_big_graph',
    'visualize_graph_split',
]


def get_styles():
    """
    Return a dictionary containing styles for different types of networks.
    """
    styles = {
        'default': {
            'nodes': {
                'sources': {
                    'shape': 'circle',
                    'color': 'steelblue',
                    'style': 'filled',
                    'fillcolor': 'steelblue',
                    'label': '',
                    'penwidth': 3
                },
                'targets': {
                    'shape': 'circle',
                    'color': 'mediumpurple1',
                    'style': 'filled',
                    'fillcolor': 'mediumpurple1',
                    'label': '',
                    'penwidth': 3
                },
                'other': {
                    'shape': 'circle',
                    'color': 'gray',
                    'style': 'filled',
                    'fillcolor': 'gray',
                    'label': ''
                }
            },
            'edges': {
                'neutral': {
                    'color': 'gray30',
                    'penwidth': 2
                }
            }
        },
        'sign_consistent': {
            'nodes': {
                'sources': {
                    'default': {
                        'shape': 'circle',
                        'style': 'filled',
                        'fillcolor': 'steelblue',
                        'label': '',
                        'penwidth': 3,
                        'color': 'steelblue'
                    },
                    'positive_consistent': {
                        'color': 'forestgreen'
                    },
                    'negative_consistent': {
                        'color': 'tomato3'
                    }
                },
                'targets': {
                    'default': {
                        'shape': 'circle',
                        'style': 'filled',
                        'fillcolor': 'mediumpurple1',
                        'label': '',
                        'penwidth': 3,
                        'color': 'mediumpurple1'
                    },
                    'positive_consistent': {
                        'color': 'forestgreen'
                    },
                    'negative_consistent': {
                        'color': 'tomato3'
                    }
                },
                'other': {
                    'default': {
                        'shape': 'circle',
                        'color': 'gray',
                        'style': 'filled',
                        'fillcolor': 'gray',
                        'label': ''
                    }
                }
            },
            'edges': {
                'positive': {
                    'color': 'forestgreen',
                    'penwidth': 2
                },
                'negative': {
                    'color': 'tomato3',
                    'penwidth': 2
                },
                'neutral': {
                    'color': 'gray30',
                    'penwidth': 2
                }
            }
        },
        # Add more network styles here
    }

    return styles


def set_style_attributes(item, base_style, condition_style=None):
    """
    Set attributes for a graph item (node or edge) based on the given styles.

    Args:
        item (node or edge): The item to set attributes for.
        base_style (dict): The base style dictionary with default attribute settings.
        condition_style (dict, optional): A dictionary of attribute settings for specific conditions. Defaults to None.
    """
    for attr, value in base_style.items():
        item.attr[attr] = value

    if condition_style:
        for attr, value in condition_style.items():
            item.attr[attr] = value


def merge_styles(default_style, custom_style, path=""):
    """
    Merge custom styles with default styles to ensure all necessary fields are present.

    Args:
        default_style (dict): The default style dictionary.
        custom_style (dict): The custom style dictionary.
        path (str): The path in the dictionary hierarchy for logging purposes.

    Returns:
        dict: The merged style dictionary.
    """
    merged_style = default_style.copy()
    if custom_style is not None:
        for key, value in custom_style.items():
            if isinstance(value, dict) and key in merged_style:
                merged_style[key] = merge_styles(merged_style[key], value, f"{path}.{key}" if path else key)
            else:
                merged_style[key] = value

        # Log missing keys in custom_style
        for key in default_style:
            if key not in custom_style:
                _log(f"Missing key '{path}.{key}' in custom style. Using default value.")

    return merged_style


def visualize_network_default(network, source_dict, target_dict, prog='dot', custom_style=None):
    """
    Core function to visualize the graph.

    Args:
        network (nx.Graph): The network to visualize.
        source_dict (dict): A dictionary containing the sources and sign of perturbation.
        target_dict (dict): A dictionary containing the targets and sign of measurements.
        prog (str, optional): The layout program to use. Defaults to 'dot'.
        custom_style (dict, optional): The custom style to apply. If None, the default style is used.
    """
    default_style = get_styles()['default']
    style = merge_styles(default_style, custom_style)

    A = nx.nx_agraph.to_agraph(network)
    A.graph_attr['ratio'] = '1.2'

    sources = set(source_dict.keys())
    target_dict_flat = {sub_key: sub_value for key, value in target_dict.items() for sub_key, sub_value in value.items()}
    targets = set(target_dict_flat.keys())

    for node in A.nodes():
        n = node.get_name()
        if n in sources:
            base_style = style['nodes']['sources']
        elif n in targets:
            base_style = style['nodes']['targets']
        else:
            base_style = style['nodes']['other']

        set_style_attributes(node, base_style)

    for edge in A.edges():
        edge_style = style['edges']['neutral']
        set_style_attributes(edge, edge_style)

    A.layout(prog=prog)
    return A


def visualize_network_sign_consistent(network, source_dict, target_dict, prog='dot', custom_style=None):
    """
    Visualize the graph considering sign consistency.

    Args:
        network (nx.Graph): The network to visualize.
        source_dict (dict): A dictionary containing the sources and sign of perturbation.
        target_dict (dict): A dictionary containing the targets and sign of measurements.
        prog (str, optional): The layout program to use. Defaults to 'dot'.
        custom_style (dict, optional): The custom style to apply. Defaults to None.
    """
    default_style = get_styles()['sign_consistent']
    style = merge_styles(default_style, custom_style)

    # Call the core visualization function
    A = visualize_network_default(network, source_dict, target_dict, prog, style)

    sources = set(source_dict.keys())
    target_dict_flat = {sub_key: sub_value for key, value in target_dict.items() for sub_key, sub_value in value.items()}
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
            set_style_attributes(node, {}, condition_style)  # Apply condition style without overwriting base style

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

        set_style_attributes(edge, edge_style)

    return A


def visualize_network(network, source_dict, target_dict, prog='dot', network_type='default', custom_style=None):
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
        return visualize_network_sign_consistent(network, source_dict, target_dict, prog, custom_style)
    else:
        default_style = get_styles().get(network_type, get_styles()['default'])
        return visualize_network_default(network, source_dict, target_dict, prog, custom_style)


def visualize_big_graph():
    return NotImplementedError


def visualize_graph_split():
    return NotImplementedError


#-----------------------------
# Test examples
# import matplotlib.pyplot as plt

# # Create a sample graph
# G = nx.DiGraph()
# G.add_node("A")
# G.add_node("B")
# G.add_node("C")
# G.add_edge("A", "B")
# G.add_edge("B", "C")
# G.add_edge("C", "A")

# # Define source and target dictionaries
# source_dict = {"A": 1, "B": -1}
# target_dict = {"C": {"value": 1}}

# # Basic Example with Default Style
# A = visualize_network(G, source_dict, target_dict, prog='dot', network_type='default')
# A.draw("default_style.png", format='png')
# plt.imshow(plt.imread("default_style.png"))
# plt.axis('off')
# plt.show()

# # Example with Custom Style
# custom_style = {
#     'nodes': {
#         'sources': {
#             'shape': 'rectangle',
#             'color': 'red',
#             'style': 'filled',
#             'fillcolor': 'red',
#             'penwidth': 2
#         },
#         'targets': {
#             'shape': 'ellipse',
#             'color': 'blue',
#             'style': 'filled',
#             'fillcolor': 'lightblue',
#             'penwidth': 2
#         },
#         'other': {
#             'shape': 'diamond',
#             'color': 'green',
#             'style': 'filled',
#             'fillcolor': 'lightgreen',
#             'penwidth': 2
#         }
#     },
#     'edges': {
#         'neutral': {
#             'color': 'black',
#             'penwidth': 1
#         }
#     }
# }
# A = visualize_network(G, source_dict, target_dict, prog='dot', network_type='default', custom_style=custom_style)
# A.draw("custom_style.png", format='png')
# plt.imshow(plt.imread("custom_style.png"))
# plt.axis('off')
# plt.show()

# # Example with Sign Consistent Network
# G["A"]["B"]["interaction"] = 1
# G["B"]["C"]["interaction"] = -1
# G["C"]["A"]["interaction"] = 1
# A = visualize_network(G, source_dict, target_dict, prog='dot', network_type='sign_consistent')
# A.draw("sign_consistent_style.png", format='png')
# plt.imshow(plt.imread("sign_consistent_style.png"))
# plt.axis('off')
# plt.show()
