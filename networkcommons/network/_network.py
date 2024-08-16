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
Representation of molecular interaction networks.

The `Network` object represents a graph of molecular interactions with custom
node and edge attributes and metadata. It is a central piece of the
`networkcommons` package, and integrates virtually all the functionalities
provided.
"""

from __future__ import annotations

__all__ = ['Network']

import lazy_import
import pandas as pd
cn = lazy_import.lazy_import('corneto')

from networkcommons.data import _network as _universe
from networkcommons.noi._noi import Noi

from networkcommons import utils


class Network:
    """
    A molecular interaction network.
    """


    def __init__(
        self,
        universe: str | None = "omnipath", # this will be the initial graph ()
        noi: Noi | list[str] | list[list[str]] | dict[str, list[str]] = None,
    ):
        """
        Args:
            universe:
                The prior knowledge universe: a complete set of interactions
                that the instance uses to extract subnetworks from and that
                will be queried in operations applied on the instance.
            noi:
                Nodes of interest.
        """

        self._co: cn.Graph = None
        self._edges: pd.DataFrame = None
        self._nodes: pd.DataFrame = None
        self.universe = universe
        self.noi = noi


    def _load(self):
        """
        Populates the object from the universe (initial graph).
        """
        if isinstance(self.universe, str):
            self.universe = _universe.getattr(f'get_{self.universe}')

        if callable(self.universe):
            self.universe = self.universe()

        type_dict = {
            cn._graph.Graph: 'corneto',
            nx.Graph: 'networkx',
            nx.DiGraph: 'networkx',
            pd.DataFrame: 'pandas',
        }

        getattr(self, f'_from_{type_dict[type(self.universe)]}')()


    def _from_corneto(self):

        self._co = self.universe
        self._attrs_from_corneto()


    def _from_networkx(self):

        self._co = utils.to_cornetograph(self.universe)
        self._attrs_from_corneto()

    
    def _from_pandas(self):

        nxgraph = utils.network_from_df(self.universe)
        self._co = utils.to_cornetograph(nxgraph)
        self._attrs_from_corneto()


    def _attrs_from_corneto(self):

        self._nodes = utils.node_attrs_from_corneto(self._co)
        self._edges = utils.edge_attrs_from_corneto(self._co)


    def as_igraph(self, attrs: str | list[str]) -> "igraph.Graph":
        """
        Return the graph as an igraph object with the desired attributes.
        """

        pass

    def as_nx(self, attrs: str | list[str]) -> nx.DiGraph:
        """
        Return the graph as an igraph object with the desired attributes.
        """

        pass


    def as_corneto(self, attrs: str | list[str]) -> cn.Graph:
        """
        Return the graph as an igraph object with the desired attributes.
        """

        return self._co.copy()
