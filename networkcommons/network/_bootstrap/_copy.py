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
Create Network object by copying another Network object.
"""

__all__ = ['BootstrapCopy']

import copy

from . import _base as _bsbase


class BootstrapCopy(_bsbase.BootstrapBase):


    def __init__(self, edges):

        directed = edges.directed
        super().__init__(locals())


    def _bootstrap(self, edges):

        self._edges = copy.deepcopy(edges.edges)
        self._nodes = copy.deepcopy(edges.nodes)
        self._node_attrs = copy.deepcopy(edges.node_attrs)
        self._edge_attrs = copy.deepcopy(edges.edge_attrs)
        self._edge_node_attrs = copy.deepcopy(edges.edge_node_attrs)
        self.node_key = edges.node_key
