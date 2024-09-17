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
Preprocess inputs for the Network object from data frames.
"""

__all__ = ['BootstrapDf']

import copy

import pandas as pd
from pypath_common import _misc
from pypath_common import _constants

from .. import _constants as _nconstants
from . import _base as _bsbase


class BootstrapDf(_bsbase.BootstrapBase):
    """
    Bootstrap Network object data structures from data frames.
    """

    def __init__(
            self,
            edges: pd.DataFrame,
            nodes: pd.DataFrame | None = None,
            node_key: str | tuple[str] | None = None,
            source_key: str = 'source',
            target_key: str = 'target',
            inner_sep: str | None = ';',
            node_key_sep: str | None = ',',
            edge_node_attrs: pd.DataFrame | None = None,
            directed: bool = True,
            ignore: list[str] | None = None,
        ):
        """
        Args:
            edges:
                Adjacency information, edge attributes and node edge
                attributes. The columns defined in `source_key` and
                `target_key` contain the source and target nodes. These have to
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
            node_key_col:
                In case the node key variables are provided in a single column,
                the name of the column has to be provided here. The column will
                be split by `node_key_sep` if the `node_key` consists of
                more than one variable.
            source_key:
                Name of the column containing the source node identifiers.
            target_key:
                Name of the column containing the target node identifiers.
            inner_sep:
                If the values in source and target columns are strings, use
                this separator to split them.
            node_key_sep:
                If the nodes are provided as strings, but the key consists of
                multiple values, use this separator to split them into tuples.
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
            node_key_col: str | None = None,
            source_key: str = 'source',
            target_key: str = 'target',
            inner_sep: str | None = ';',
            node_key_sep: str | None = ',',
            edge_node_attrs: pd.DataFrame | None = None,
            ignore: list[str] | None = None,
        ):

        nodes = copy.deepcopy(nodes)
        edges = copy.deepcopy(edges)

        self._set_node_key(node_key)
        self.node_key_sep = node_key_sep
        self.inner_sep = inner_sep
        self._bootstrap_nodes(
            nodes = nodes,
            node_key_col = node_key_col,
            node_key_sep = node_key_sep,
        )
        self._bootstrap_edges(
            edges = edges,
            source_key = source_key,
            target_key = target_key,
            ignore = ignore,
        )


    def _bootstrap_edges(
            self,
            edges: pd.DataFrame,
            source_key: str = 'source',
            target_key: str = 'target',
            ignore: list[str] | None = None,
        ):

        edges.insert(0, _nconstants.EDGE_ID, range(len(edges)))

        for col in (source_key, target_key):

            edges[col] = edges[col].apply(self._proc_nodes_in_edge)

        ignore = _misc.to_set(ignore) & set(edges.columns)
        edges = edges.drop(columns = ignore)

        new_nodes = (
            (
                set.union(*edges[source_key]) |
                set.union(*edges[target_key])
            ) -
            set(self._nodes.keys())
        )

        self._nodes = {
            **self._nodes,
            **{n: (set(), set()) for n in new_nodes}
        }


        self._node_attrs = pd.concat([
            self._node_attrs,
            pd.DataFrame([dict(zip(self.node_key, n)) for n in new_nodes])
        ])
        self._update_node_key_col()

        cols = [_nconstants.EDGE_ID, source_key, target_key]
        self._edges = {
            i: (s, t) if self.directed else (s | t, set())
            for j, (i, s, t) in edges[cols].iterrows()
        }
        self._edge_attrs = edges.drop(columns = [source_key, target_key])

        for eid, nodes in self._edges.items():

            for si, side in self._sides():

                for node in nodes[si]:

                    self._nodes[node][si].add(eid)


    def _proc_nodes_in_edge(
            self,
            nodes: str | int | tuple | list | set,
        ):

        if self.inner_sep and isinstance(nodes, str):

            nodes = nodes.split(self.inner_sep)

        if isinstance(nodes, (_constants.SIMPLE_TYPES, tuple)):

            nodes = [nodes]

        nodes = {self._proc_node_key(n) for n in nodes}

        return nodes


    def _bootstrap_nodes(
            self,
            nodes: pd.DataFrame | None = None,
            node_key_col: str | None = None,
            node_key_sep: str | None = ',',
        ):


        if nodes is not None:

            if not node_key_col and _nconstants.NODE_KEY in nodes.columns:

                node_key_col = _nconstants.NODE_KEY

            if node_key_col:

                if node_key_col not in nodes.columns:

                    raise ValueError(
                        f'`node_key_col` provided ({node_key_col}), but not '
                        'found in the `nodes` data frame.'
                    )

                nodes[node_key_col] = nodes[node_key_col].apply(
                    self._proc_node_key,
                )

                nodes.rename(
                    {node_key_col: _nconstants.NODE_KEY},
                    axis = 1,
                    inplace = True,
                )

                keys = pd.DataFrame(
                    nodes[_nconstants.NODE_KEY].tolist(),
                    index = nodes.index,
                )
                keys.columns = self.node_key
                nodes = nodes.join(keys)

            if missing := set(self.node_key) - set(nodes.columns):

                raise ValueError(
                    'Columns in `node_key` not found '
                    f'in `nodes` data frame: `{", ".join(missing)}`.'
                )

            self._node_attrs = nodes
            self._update_node_key_col()
            self._nodes = {
                k: (set(), set())
                for k in nodes[_nconstants.NODE_KEY]
            }


    def _proc_node_key(self, key: str | tuple) -> str | tuple:

        sep = self.node_key_sep
        key = key.split(sep) if sep and isinstance(key, str) else key
        key = _misc.to_tuple(key)
        key = (key + (None,) * len(self.node_key))[:len(self.node_key)]

        if len(key) == 1:

            key = key[0]

        return key

    def _update_node_key_col(self):

        nattrs = self._node_attrs

        # bravo, pandas...
        nattrs[_nconstants.NODE_KEY] = pd.Series(zip(
            *nattrs[list(self.node_key)].T.values
        )).apply(lambda x: x[0] if len(x) == 1 else x)

        for col in reversed((_nconstants.NODE_KEY,) + self.node_key):

            col = nattrs.pop(col)
            nattrs.insert(0, col.name, col)

        self._node_attrs = nattrs
