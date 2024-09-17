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

from __future__ import annotations

"""
Base class for classes preprocessing Network object inputs.
"""

__all__ = ['BootstrapBase']

import abc

import pandas as pd
from pypath_common import _misc

from .. import _constants as _nconstants


class BootstrapBase(abc.ABC):


    def __init__(self, args: dict):
        """
        Attributes:
            _edges:
                Adjacency information. A dict with numeric edge IDs as keys and
                tuples of two sets of node IDs (source and target nodes) as
                values.
            _nodes:
                Adjacency information. A dict with node IDs as keys and tuples
                of two sets of edge IDs (source vs target side) as values.
            _node_attrs:
                Data frame of node attributes.
            _edge_attrs:
                Data frame of edge attributes.
            _edge_node_attrs:
                Data frame of node edge attributes.
            directed:
                Whether the network is directed.
        """

        self._edges = {}
        self._nodes = {}
        self._node_attrs = pd.DataFrame()
        self._edge_attrs = pd.DataFrame()
        self._edge_node_attrs = pd.DataFrame()

        args.pop('self')
        args.pop('__class__', None)
        self.directed = args.pop('directed')

        self._bootstrap(**args)


    @abc.abstractmethod
    def _bootstrap(self, *args, **kwargs):

        raise NotImplementedError


    @staticmethod
    def _sides():

        return enumerate(('source', 'target'))


    def _set_node_key(self, node_key: str | tuple | None = None):

        node_key = node_key or _nconstants.DEFAULT_KEY
        self.node_key = _misc.to_tuple(node_key)
