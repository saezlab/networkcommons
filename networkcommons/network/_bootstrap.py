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
Preprocess arguments of the Network object into its internal representation.
"""

from typing import Iterable

import abc
import itertools

import pandas as pd
from pypath_common import _misc
from pypath_common import _constants

from . import _constants as _nconstants


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
            _node_edge_attrs:
                Data frame of node edge attributes.
            directed:
                Whether the network is directed.
        """

        self._edges = {}
        self._nodes = {}
        self._node_attrs = pd.DataFrame()
        self._edge_attrs = pd.DataFrame()
        self._node_edge_attrs = pd.DataFrame()

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


class Bootstrap(BootstrapBase):
    """
    Bootstrap network data structures from a variety of Python objects.
    """

    def __init__(
            self,
            edges: Iterable[dict | tuple] | None = None,
            nodes: list[str | dict] | dict[dict] | None = None,
            node_key: str | tuple[str] | None = None,
            source_key: str = 'source',
            target_key: str = 'target',
            edge_attrs: list[str] | None = None,
            directed: bool = True,
        ):
        """
        Args:
            node_key:
                Name of the variable(s) among node attributes that uniquely
                define each node. If `nodes` is provided, unless it has an "id"
                column, `node_key` must be defined. If only an `edges` data
                frame is provided, the contents of the source and target
                columns will be used as node key, and `node_key` will be used
                as column name for this node attribute.
            source_key:
                Key in the edge dictionary containing the source node identifiers.
            target_key:
                Key in the edge dictionary containing the target node identifiers.
            directed:
                Is the network directed? If yes, nodes of each edge will
                be separated to source and target, otherwise all nodes will be
                considered "source", rendering the edges undirected.
        """

        super().__init__(locals())


    def _bootstrap(
            self,
            edges: Iterable[dict | tuple] | None = None,
            nodes: list[str | dict] | dict[dict] | None = None,
            node_key: str | tuple[str] | None = None,
            source_key: str = _nconstants.SOURCE_KEY,
            target_key: str = _nconstants.TARGET_KEY,
            edge_attrs: list[str] | None = None,
        ):

        # preprocessing
        self._set_node_key(node_key)

        proc_edges, proc_nodes = self._bootstrap_edges(
            edges = edges,
            source_key = source_key,
            target_key = target_key,
            edge_attrs = edge_attrs,
        )

        nodes = self._bootstrap_nodes(nodes)

        nodes = {
            k: {
                **nodes.get(k, {}),
                **proc_nodes.get(k, {}),
            }
            for k in set(itertools.chain(
                nodes.keys(),
                proc_nodes.keys(),
            ))
        }

        # adding nodes & node attrs
        self._nodes = {key: (set(), set()) for key in nodes}
        self._node_attrs = pd.DataFrame(nodes.values())

        # adding edges, edge attrs, edge-node attrs
        self._edges = {}
        _edge_attrs = []
        _node_edge_attrs = []

        for i, edge in enumerate(proc_edges):

            source = edge.pop(source_key)
            target = edge.pop(target_key)
            source_attrs = edge.pop(f'{source_key}_attrs')
            target_attrs = edge.pop(f'{target_key}_attrs')
            edge[_nconstants.EDGE_ID] = i

            # adjacency information
            self._edges[i] = (source, target)

            for si, side in self._sides():

                for node_id in locals()[side]:

                    self._nodes[node_id][si].add(i)

            # edge and node-edge attributes
            _edge_attrs.append(edge)

            _node_edge_attrs.extend([
                {
                    _nconstants.NODE_KEY: key,
                    _nconstants.EDGE_ID: i,
                    _nconstants.SIDE: side,
                    **node_attrs,
                }
                for side, side_attrs in zip(
                    ('source', 'target'),
                    (source_attrs, target_attrs)
                )
                for key, node_attrs in side_attrs.items()
            ])

        self._edge_attrs = pd.DataFrame(_edge_attrs)
        self._node_edge_attrs = pd.DataFrame(_node_edge_attrs)


    def _bootstrap_edges(
            self,
            edges: Iterable[dict | tuple],
            source_key: str = 'source',
            target_key: str = 'target',
            edge_attrs: list[str] | None = None,
        ) -> tuple[list[dict], set[tuple]]:

        edge_vars = (source_key, target_key) + _misc.to_tuple(edge_attrs)

        proc_edges = []
        proc_nodes = {}
        mandatory_keys = {source_key}
        if self.directed: mandatory_keys.add(target_key)

        for e in edges or ():

            if isinstance(e, dict):

                if (missing := mandatory_keys - set(e)):

                    raise ValueError(f'Edge `{e}` has no key(s) `{missing}`.')

                e = e.copy()

            else:

                edge = _misc.to_tuple(e)

                if len(edge) != (n := len(edge_vars)):

                    raise ValueError(f'Expected `{n}` variables: `{edge}`.')

                e = dict(zip(edge_vars, edge))

            self._bootstrap_node_in_edge(e, source_key)
            self._bootstrap_node_in_edge(e, target_key)
            proc_edges.append(e)

            for side_key in (source_key, target_key):

                for node in e[side_key]:

                    proc_nodes[node] = {
                        _nconstants.NODE_KEY: node,
                        **dict(zip(self.node_key, _misc.to_tuple(node))),
                    }

        return proc_edges, proc_nodes


    def _bootstrap_node_in_edge(self, edge: dict, side: str):

        neattr_key = f'{side}_attrs'
        nodes = edge.get(side, {})

        if isinstance(nodes, (_constants.SIMPLE_TYPES, tuple)):

            nodes = {nodes: edge.pop(neattr_key, {})}

        if not isinstance(nodes, dict):

            attrs = edge.pop(neattr_key, {})

            if len(nodes) == 1 and (n :=_misc.first(nodes)) not in attrs:

                attrs = {n: attrs}

            nodes = {n: attrs.get(n, {}) for n in nodes}

        edge[side] = set(nodes.keys())
        edge[neattr_key] = nodes


    def _set_node_key(self, node_key: str | tuple | None = None):

        node_key = node_key or _nconstants.DEFAULT_KEY
        self.node_key = _misc.to_tuple(node_key)


    def _bootstrap_nodes(
            self,
            nodes: list[str | dict] | dict[dict] | None = None,
        ) -> dict[tuple, dict]:

        nodes = nodes or {}

        if isinstance(nodes, dict):

            nodes_proc = {}

            for k, v in nodes.items():

                k = _misc.to_tuple(k)
                nodes_proc[k] = {
                    _nconstants.NODE_KEY: k,
                    **dict(zip(self.node_key, k)),
                    **v,
                }

            nodes = nodes_proc

        elif isinstance(nodes, list):

            nodes = dict(
                self._bootstrap_node(v, i)
                for i, v in enumerate(nodes)
            )

        return nodes


    def _bootstrap_node(
            self,
            node: str | int | dict,
            idx: int,
        ) -> tuple[tuple, dict]:

        if not isinstance(node, dict):

            node = {self.node_key[0]: node}

        key = tuple(
            node.get(k, idx if k == _nconstants.DEFAULT_KEY else None)
            for k in self.node_key
        )

        if all(k is None for k in key):

            raise ValueError(f'Node with empty key: `{node}`.')

        node[_nconstants.NODE_KEY] = key

        return key, node


class BootstrapDf(BootstrapBase):
    """
    Bootstrap Network object data structures from data frames.
    """

    def __init__(
            self,
            edges: pd.DataFrame,
            nodes: pd.DataFrame | None = None,
            node_key: str | tuple[str] | None = None,
            source_col: str = 'source',
            target_col: str = 'target',
            edge_node_attrs: pd.DataFrame | None = None,
            directed: bool = True,
            ignore: list[str] | None = None,
        ):
        """
        Args:
            edges:
                Adjacency information, edge attributes and node edge
                attributes. The columns defined in `source_col` and
                `target_col` contain the source and target nodes. These have to
                be single node identifiers (for a binary graph) or sets of node
                identifiers. Node identifiers are either atomic variables or
                tuples, as defined by the `node_key` argument. The rest of the
                columns will be used as edge attributes.
            nodes:
                Node attributes. Each node must be represented by a single row.
                One or more of the attributes defined in `node_key`, the value
                of this attribute or the combination of these has to be unique.
            node_key:
                Name of the variable(s) among node attributes that uniquely
                define each node. If `nodes` is provided, unless it has an "id"
                column, `node_key` must be defined. If only an `edges` data
                frame is provided, the contents of the source and target
                columns will be used as node key, and `node_key` will be used
                as column name(s) for this/these node attribute(s).
            source_col:
                Name of the column containing the source node identifiers.
            target_col:
                Name of the column containing the target node identifiers.
            directed:
                Is the network directed? If yes, nodes of each edge will
                be separated to source and target, otherwise all nodes will be
                considered "source", rendering the edges undirected.
            ignore:
                List of columns to ignore: won't be processed into edge
                attributes.
        """

        super().__init__(locals())


    def _bootstrap(
            self,
            edges: pd.DataFrame,
            nodes: pd.DataFrame | None = None,
            node_key: str | tuple[str] | None = None,
            source_col: str = 'source',
            target_col: str = 'target',
            edge_node_attrs: pd.DataFrame | None = None,
            ignore: list[str] | None = None,
        ):

        node_key = _misc.to_tuple(node_key or _nconstants.DEFAULT_KEY)

        if nodes is not None:

            if missing := set(node_key) - set(nodes.columns):

                raise ValueError(
                    'Columns in `node_key` not found '
                    f'in `nodes` data frame: `{", ".join(missing)}`.'
                )

            self._nodes = {
                tuple(node_id): (set(), set())
                for node_id in nodes[list(node_key)].itertuples()
            }
