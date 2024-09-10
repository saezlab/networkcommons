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

from collections.abc import Hashable, Iterable, Iterator
from typing import Any, NamedTuple
import inspect
import importlib as imp
import warnings

import lazy_import
import pandas as pd
cn = lazy_import.lazy_module('corneto')

from pypath_common import _misc
from pypath_common import _constants

from networkcommons.data import network as _universe
from networkcommons.noi._noi import Noi

from networkcommons import utils

from . import _bootstrap
from . import _constants as _nconstants
from networkcommons import _log


class NetworkBase:


    def __init__(
            self,
            edges: Iterable[dict | tuple] | pd.DataFrame | None = None,
            nodes: list[str | dict] | dict[dict] | pd.DataFrame | None = None,
            node_key: str | tuple[str] | None = None,
            source_key: str = 'source',
            target_key: str = 'target',
            edge_attrs: list[str] | None = None,
            inner_sep: str | None = ';',
            node_key_sep: str | None = ',',
            edge_node_attrs: pd.DataFrame | None = None,
            directed: bool = True,
            ignore: list[str] | None = None,
        ):

        if isinstance(edges, self.__class__):

            bs = _bootstrap.BootstrapCopy

        elif any(isinstance(d, pd.DataFrame) for d in (edges, nodes)):

            bs = _bootstrap.BootstrapDf

        else:

            bs = _bootstrap.Bootstrap

        the_locals = locals()

        args = {
            k: the_locals[k]
            for k in inspect.signature(bs.__init__).parameters.keys()
            if k not in {'self'}
        }

        proc = bs(**args)

        self.edges = proc._edges
        self.nodes = proc._nodes
        self.edge_attrs = proc._edge_attrs
        self.node_attrs = proc._node_attrs
        self.edge_node_attrs = proc._edge_node_attrs
        self.node_key = proc.node_key
        self.directed = proc.directed

        self._set_index()
        self._sort()


    def reload(self):
        """
        Reloads the object from the module level.
        """

        modname = self.__class__.__module__
        mod = __import__(modname, fromlist=[modname.split('.')[0]])
        imp.reload(mod)
        new = getattr(mod, self.__class__.__name__)
        setattr(self, '__class__', new)


    @property
    def hyper(self) -> bool:
        """
        Whether the network is a hypergraph.
        """

        return any(len(s | t) > 2 for s, t in self.edges.values())


    def __len__(self) -> int:

        return self.ecount


    @property
    def ecount(self) -> int:
        """
        Number of edges.
        """

        return len(getattr(self, 'edges', ()))


    @property
    def ncount(self) -> int:
        """
        Number of nodes.
        """

        return len(getattr(self, 'nodes', ()))


    def __repr__(self) -> str:

        return f'<{self.__class__.__name__} {self.ncount}N x {self.ecount}E>'


    def __getitem__(self, key: Hashable) -> tuple[set, set]:

        try:

            return self.nodes.get(key, self.edges[key])

        except KeyError:

            raise KeyError(f'No such node or edge: `{key}`.')


    def __contains__(self, key: Hashable) -> bool:

        return key in self.nodes


    def _set_index(self) -> None:

        INDEX_COLS = {
            'edge': _nconstants.EDGE_ID,
            'node': _nconstants.NODE_KEY,
            'edge_node': [_nconstants.EDGE_ID, _nconstants.NODE_KEY],
        }

        for entity, cols in INDEX_COLS.items():

            if not (df := getattr(self, f'{entity}_attrs')).empty:

                df.index = df[cols]


    def _sort(self) -> None:

        self.edge_attrs.sort_index(inplace = True)
        self.node_attrs.sort_index(inplace = True)
        self.edge_node_attrs.sort_index(inplace = True)


    def iteredges(self, *attrs: str) -> Iterator[tuple]:
        """
        Iterate edges as named tuples.

        Args:
            attrs:
                Attributes to include.
        """

        if (noattr := set(attrs) - set(self.edge_attrs.columns)):

            msg = f'No such edge attribute(s): {", ".join(noattr)}.'
            warnings.warn(msg)

        Edge = NamedTuple(
            'Edge',
            [
                ('source', str | int | tuple | set),
                ('target', str | int | tuple | set),
            ] + [
                (a, _misc.df_dtype_to_builtin(self.edge_attrs, a))
                for a in attrs
            ]
        )
        hyper = self.hyper

        for i in self.edge_attrs.iterrows():

            source, target = self.edges[i[0]]

            if not hyper:

                if self.directed:

                    source = _misc.first(source)
                    target = _misc.first(target)

                else:

                    source, *target = sorted(source | target)
                    target = _misc.first(target) or source  # loop edge

            yield Edge(
                source,
                target,
                *(getattr(i[1], a, None) for a in attrs)
            )


    __iter__ = iteredges


    def iternodes(self, *attrs: str) -> Iterator[tuple]:
        """
        Iterate node data.

        Args:
            attrs:
                Node attributes (variables) to include. By default, only the
                "key" is included, which is a variable or combination of
                variables that uniquely define each node.
        """

        Node = NamedTuple(
            'Node',
            [
                ('key', str | int | tuple)
            ] + [
                (a, _misc.df_dtype_to_builtin(self.node_attrs, a))
                for a in attrs
            ]
        )

        for key, vars in self.node_attrs.iterrows():

            yield Node(key, *(getattr(vars, a, None) for a in attrs))


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
