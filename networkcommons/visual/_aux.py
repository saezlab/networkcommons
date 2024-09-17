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

from typing import List
from networkcommons._session import _log
import re


def adjust_node_name(node_name: str,
                     truncate: bool = False,
                     wrap: bool = False,
                     max_length: int = 8,
                     wrap_length: int = 8,
                     ensure_unique: bool = False,
                     ensure_unique_list: List[str] = None,
                     remove_strings: List[str] = None) -> str:
    """
    Adjust the node name by replacing special characters, truncating, wrapping, and ensuring uniqueness.

    Parameters
    ----------
    node_name : str
        The node name to adjust.
    truncate : bool, optional
        Whether to truncate the node name, by default False.
    wrap : bool, optional
        Whether to wrap the node name, by default False.
    max_length : int, optional
        The maximum length of the node name, by default 8.
    wrap_length : int, optional
        The length at which to wrap the node name, by default 8.
    ensure_unique : bool, optional
        Whether to ensure the node name is unique, by default False.
    ensure_unique_list : List[str], optional
        A list of node names to ensure uniqueness against, by default None.
    remove_strings : List[str], optional
        A list of substrings to remove from the node name, by default None.
    """

    # Replace any ':' with '_'
    node_name = node_name.replace(":", "_")

    # Remove provided substrings, if any
    if remove_strings:
        for string in remove_strings:
            node_name = node_name.replace(string, "")

    # Replace special symbols (anything non-alphanumeric or underscore) with '_'
    new_node_name = re.sub(r'[^a-zA-Z0-9_]', '_', node_name)

    # Log any replacement of special symbols
    if new_node_name != node_name:
        _log(f"Replaced special characters in '{node_name}' with underscores.", level=10)

    # If the wrap flag is set to True, wrap the node name
    if wrap:
        new_node_name = "\n".join([new_node_name[i:i + wrap_length]
                                   for i in range(0, len(new_node_name), wrap_length)])

    # If the modified node name is longer than the specified maximum length, truncate it
    if truncate and len(new_node_name) > max_length:
        new_node_name = new_node_name[:max_length] + "..."
        _log(f"Truncated node name '{node_name}' to '{new_node_name}'.", level=10)

    # If multiple underscores are present, replace them with a single underscore
    new_node_name = re.sub(r'_+', '_', new_node_name)

    # If the node name is empty, return an empty string and log a warning
    if not new_node_name:
        _log("Empty node name provided. Returning an empty string.", level=20)
        return ""

    if ensure_unique and ensure_unique_list:
        # Ensure the modified node name is unique by appending a number
        if new_node_name in ensure_unique_list:
            i = 1
            while f"{new_node_name}_{i}" in ensure_unique_list:
                i += 1
            new_node_name = f"{new_node_name}_{i}"
            _log(f"Ensured unique node name by appending a number: '{new_node_name}'.", level=10)

    # Return the modified node name, stripped of leading and trailing spaces, and _ characters
    return new_node_name.strip("_")
