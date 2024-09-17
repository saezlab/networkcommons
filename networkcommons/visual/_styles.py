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
Style definitions for network visualizations.
"""

from __future__ import annotations

__all__  = ['get_styles', 'merge_styles', 'set_style_attributes']

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
                    'color': '#62a0ca',  # A deep blue
                    'style': 'filled',
                    'fillcolor': '#a6cee3',  # Light blue for the fill
                    'penwidth': 3
                },
                'targets': {
                    'shape': 'circle',
                    'color': '#70bc6b',  # A deep green
                    'style': 'filled',
                    'fillcolor': '#b2df8a',  # Light green for the fill
                    'penwidth': 3
                },
                'other': {
                    'default': {
                        'shape': 'circle',
                        'color': '#848484',  # Dark gray for default nodes
                        'style': 'filled',
                        'fillcolor': '#bdbdbd',  # Light gray for default nodes
                        'penwidth': 3
                    }
                }
            },
            'edges': {
                'positive': {
                    'color': '#70bc6b',  # Same deep green as target nodes
                    'penwidth': 2.5
                },
                'negative': {
                    'color': '#eb5e60',  # Red for negative edges
                    'penwidth': 2.5
                },
                'neutral': {
                    'color': '#3e3e3e', # Dark gray for neutral edges
                    'penwidth': 2.5
                },
                'default': {
                    'color': '#bdbdbd', # Light gray for default edges
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
                        'fillcolor': 'a6cee3',  # blue
                        'penwidth': 3,
                        'color': '#62a0ca' # Deep blue
                    },
                    'positive_consistent': {
                        'color': '#62a0ca'  # Consistent blue for positive
                    },
                    'negative_consistent': {
                        'color': '#eb5e60'  # Consistent red for negative
                    }
                },
                'targets': {
                    'default': {
                        'shape': 'circle',
                        'style': 'filled',
                        'fillcolor': '#70bc6b',  # green
                        'penwidth': 3,
                        'color': '#70bc6b'
                    },
                    'positive_consistent': {
                        'color': '#70bc6b'  # Same green for positive
                    },
                    'negative_consistent': {
                        'color': '#eb5e60'  # Same red for negative
                    }
                },
                'other': {
                    'default': {
                        'shape': 'circle',
                        'color': '#848484',  # Dark gray
                        'style': 'filled',
                        'fillcolor': '#bdbdbd',  # gray
                    },
                    'positive_consistent': {
                        'color': '#848484'  # Same gray for positive
                    },
                    'negative_consistent': {
                        'color': '#eb5e60'  # Same red for negative
                    }
                }
            },
            'edges': {
                'positive': {
                    'color': '#33a02c',  # Deep green
                    'penwidth': 2.5
                },
                'negative': {
                    'color': '#e31a1c',  # Strong red
                    'penwidth': 2.5
                },
                'neutral': {
                    'color': '#6a3d9a',  # Purple
                    'penwidth': 2.5
                },
                'default': {
                    'color': '#bdbdbd',  # Light gray
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
                _log(f"Missing key '{path}.{key}' in custom style. Using default value.", level=30)

    return merged_style
