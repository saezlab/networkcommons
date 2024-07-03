from networkcommons._session import _log


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
                'default': {
                    'shape': 'circle',
                    'color': 'gray',
                    'style': 'filled',
                    'fillcolor': 'gray',
                    'label': ''
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
                'default': {
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
        }
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

    return item


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

