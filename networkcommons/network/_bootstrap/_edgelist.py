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
Preprocess arguments of the Network object from nested lists and dicts.
"""

__all__ = ['Bootstrap']

from typing import Iterable

import abc
import itertools
import copy

import pandas as pd
from pypath_common import _misc
from pypath_common import _constants

from .. import _constants as _nconstants
from . import _base as _bsbase


class Bootstrap(_bsbase.BootstrapBase):
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

        nodes = copy.deepcopy(nodes)
        edges = copy.deepcopy(edges)

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
        _edge_node_attrs = []

        for i, edge in enumerate(proc_edges):

            source = edge.pop(source_key)
            target = edge.pop(target_key)
            source_attrs = edge.pop(f'{source_key}_attrs')
            target_attrs = edge.pop(f'{target_key}_attrs')
            edge[_nconstants.EDGE_ID] = i

            if not self.directed:

                source.update(target)
                target = set()

            # adjacency information
            self._edges[i] = (source, target)

            for si, side in self._sides():

                for node_id in locals()[side]:

                    self._nodes[node_id][si].add(i)

            # edge and node-edge attributes
            _edge_attrs.append(edge)

            _edge_node_attrs.extend([
                {
                    _nconstants.NODE_KEY: key,
                    _nconstants.EDGE_ID: i,
                    _nconstants.SIDE: side if self.directed else 'source',
                    **node_attrs,
                }
                for side, side_attrs in zip(
                    ('source', 'target'),
                    (source_attrs, target_attrs)
                )
                for key, node_attrs in side_attrs.items()
            ])

        self._edge_attrs = pd.DataFrame(_edge_attrs)
        self._edge_node_attrs = pd.DataFrame(_edge_node_attrs)


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

                e = copy.deepcopy(e)

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

            attrs = edge.pop(neattr_key, {})
            attrs = attrs.pop(nodes, None) or attrs
            nodes = {nodes: attrs}

        if not isinstance(nodes, dict):

            attrs = edge.pop(neattr_key, {})

            if len(nodes) == 1 and (n :=_misc.first(nodes)) not in attrs:

                attrs = {n: attrs}

            _nodes = []

            for n in nodes:

                if isinstance(n, dict):

                    key = self._get_node_key(n, pop = True)
                    attrs[key] = n
                    n = key

                _nodes.append(n)

            nodes = {n: attrs.get(n, {}) for n in _nodes}

        edge[side] = set(nodes.keys())
        edge[neattr_key] = nodes


    def _set_node_key(self, node_key: str | tuple | None = None):

        node_key = node_key or _nconstants.DEFAULT_KEY
        self.node_key = _misc.to_tuple(node_key)


    def _get_node_key(
            self,
            node: dict,
            pop: bool = False,
            idx: int | None = None,
        ) -> str | int | tuple:

        method = dict.pop if pop else dict.get
        key = tuple(
            method(node, k, idx if k == _nconstants.DEFAULT_KEY else None)
            for k in self.node_key
        )
        key = key[0] if len(key) == 1 else key

        return key


    def _bootstrap_nodes(
            self,
            nodes: list[str | dict] | dict[dict] | None = None,
        ) -> dict[tuple, dict]:

        nodes = nodes or {}

        if isinstance(nodes, dict):

            nodes_proc = {}

            for k, v in nodes.items():

                nodes_proc[k] = {
                    _nconstants.NODE_KEY: k,
                    **dict(zip(self.node_key, _misc.to_tuple(k))),
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

            node = dict(zip(self.node_key, _misc.to_tuple(node)))

        key = self._get_node_key(node, idx = idx)

        if (
            key is None or
            (
                isinstance(key, Iterable) and
                all(k is None for k in key)
            )
        ):

            raise ValueError(f'Node with empty key: `{node}`.')

        node[_nconstants.NODE_KEY] = key

        return key, node
