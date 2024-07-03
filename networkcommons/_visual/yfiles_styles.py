def get_styles():
    """
    Return a dictionary containing styles for different types of networks specific to yFiles visualizations.
    """
    styles = {
        'default': {
            'nodes': {
                'shape': 'round-rectangle',
                'color': '#cccccc',
                'style': 'filled',
                'fillcolor': '#cccccc',
                'label': '',
                'penwidth': 1,
                'fontSize': 12,
                'textAlignment': 'center'
            },
            'edges': {
                'color': 'gray',
                'penwidth': 1
            },
            'labels': {
                'text': '',
                'backgroundColor': None,
                'fontSize': 12,
                'color': '#030200',
                'shape': 'round-rectangle',
                'textAlignment': 'center'
            }
        },
        'highlight': {
            'nodes': {
                'shape': 'round-rectangle',
                'color': '#ffcc00',
                'style': 'filled',
                'fillcolor': '#ffcc00',
                'label': '',
                'penwidth': 2,
                'fontSize': 12,
                'textAlignment': 'center'
            },
            'edges': {
                'color': '#ffcc00',
                'penwidth': 2
            },
            'labels': {
                'text': '',
                'backgroundColor': None,
                'fontSize': 12,
                'color': '#030200',
                'shape': 'round-rectangle',
                'textAlignment': 'center'
            }
        },
        'comparison': {
            'nodes': {
                'Unique to Network 1': '#f5f536',
                'Unique to Network 2': '#36f55f',
                'Common': '#3643f5'
            },
            'edges': {
                'Unique to Network 1': '#e3941e',
                'Unique to Network 2': '#36f55f',
                'Common': '#3643f5',
                'Conflicting': '#ffcc00'
            }
        },
        'effects': {
            'activation': 'green',
            'inhibition': 'red',
            'form complex': 'blue',
            'bimodal': 'purple',
            'undefined': 'gray',
            'default': 'black'
        }
    }

    return styles


def apply_node_style(node, style):
    """
    Apply the given style to a node.

    Args:
        node (dict): The node to style.
        style (dict): The style dictionary with node attributes.
    """
    for attr, value in style.items():
        node[attr] = value
    return node


def update_node_property(node, type="color", value="blue"):
    style = {type: value}
    return apply_node_style(node, style)


def apply_edge_style(edge, style):
    """
    Apply the given style to an edge.

    Args:
        edge (dict): The edge to style.
        style (dict): The style dictionary with edge attributes.
    """
    for attr, value in style.items():
        edge[attr] = value
    return edge


def get_edge_color(effect, styles):
    """
    Get the color for an edge based on its effect.

    Args:
        effect (str): The effect type of the edge.
        styles (dict): The styles dictionary.

    Returns:
        str: The color for the edge.
    """
    return styles['effects'].get(effect, styles['effects']['default'])


def update_edge_property(edge, type="color", value="blue"):
    style = {type: value}
    return apply_edge_style(edge, style)


def get_comparison_color(category, styles, element_type='nodes'):
    """
    Get the color for nodes or edges based on the comparison category.

    Args:
        category (str): The comparison category.
        styles (dict): The styles dictionary.
        element_type (str): The type of element ('nodes' or 'edges').

    Returns:
        str: The color for the element.
    """
    return styles['comparison'][element_type].get(category, styles['default'][element_type]['color'])
