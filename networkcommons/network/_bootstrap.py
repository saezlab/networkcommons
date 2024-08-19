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

import itertools

import pandas as pd
from pypath_common import _misc
from pypath_common import _constants

from . import _constants as _nconstants


class Bootstrap:
    """
    Bootstrap network data structures from a variety of Python objects.
    """

    def __init__(
            self,
            edges: Iterable[dict | tuple],
            nodes: list[str | dict] | dict[dict],
            node_key: str | tuple[str] | None,
            source_key: str = 'source',
            target_key: str = 'target',
            edge_attrs: list[str] | None = None,
            directed: bool = True,
        ):

        self._edges = {}
        self._nodes = {}
        self._node_attrs = pd.DataFrame()
        self._edge_attrs = pd.DataFrame()
        self._node_edge_attrs = pd.DataFrame()

        args = locals()
        args.pop('self')
        self.directed = args.pop('directed')

        self._bootstrap(**args)


    def _bootstrap(
            self,
            edges: Iterable[dict | tuple],
            nodes: list[str | dict] | dict[dict],
            node_key: str | tuple[str] | None,
            source_key: str = 'source',
            target_key: str = 'target',
            edge_attrs: list[str] | None = None,
        ):

        proc_edges, proc_nodes = self._bootstrap_edges(
            edges = edges,
            source_key = source_key,
            target_key = target_key,
            edge_attrs = edge_attrs,
        )

        nodes = self._bootstrap_nodes(nodes, node_key)
        self._node_attrs = pd.DataFrame(nodes.values())
        self._nodes = {
            key: (set(), set())
            for key in itertools.chain(nodes.keys(), proc_nodes)
        }

        self._edges = {}
        _edge_attrs = []
        _node_edge_attrs = []

        for i, edge in enumerate(proc_edges):

            source = edge.pop(source_key)
            target = edge.pop(target_key)
            source_attrs = nodes.pop(f'{source_key}_attrs')
            target_attrs = nodes.pop(f'{target_key}_attrs')
            edge[_nconstants.EDGE_ID] = i

            self._edges[i] = (source, target)
            _edge_attrs.append(edge)

            _node_edge_attrs.extend([
                {
                    **dict(zip(node_key, key)),
                    **node_attrs,
                    _nconstants.EDGE_ID: i,
                    _nconstants.SIDE: side,
                }
                for side, side_attrs in zip(
                    (source, target),
                    (source_attrs, target_attrs)
                )
                for key, node_attrs in side_attrs
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
        proc_nodes = set()
        mandatory_keys = {source_key}
        if self.directed: mandatory_keys.add(target_key)

        for e in edges:

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
            proc_nodes.update(e[source_key])
            proc_nodes.update(e[target_key])

        return proc_edges, proc_nodes


    def _bootstrap_node_in_edge(self, edge: dict, key: str):

        node = edge.get(key, {})

        if isinstance(node, (_constants.SIMPLE_TYPES, tuple)):

            node = {node}

        if not isinstance(node, dict):

            node = {n: {} for n in node}

        neattr_key = f'{key}_attrs'
        edge[key] = frozenset(node.keys())
        edge[neattr_key] = node


    def _bootstrap_nodes(
            self,
            nodes: list[str | dict] | dict[dict],
            node_key: str | tuple[str] | None,
        ) -> dict[tuple, dict]:


        node_key = node_key or _nconstants.DEFAULT_KEY
        node_key = _misc.to_tuple(node_key)

        if isinstance(nodes, dict):

            nodes = {_misc.to_tuple(k): v for k, v in nodes.items()}

        if isinstance(nodes, list):

            nodes = dict(
                self._bootstrap_node(v, i, node_key)
                for i, v in enumerate(nodes)
            )

        return nodes


    def _bootstrap_node(
            self,
            node: str | int | dict,
            idx: int,
            node_key: tuple,
        ) -> tuple[tuple, dict]:

        if not isinstance(node, dict):

            node = {node_key[0]: node}

        key = tuple(
            node.get(k, idx if k == _nconstants.DEFAULT_KEY else None)
            for k in node_key
        )

        if all(k is None for k in key):

            raise ValueError(f'Node with empty key: `{node}`.')

        return key, node
