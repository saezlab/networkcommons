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
Handling nodes of interest for network lookup queries.
"""

from __future__ import annotations

__all__ = ['Noi']

from typing import Any, Callable, Hashable
import collections
import itertools
import json
import yaml
import inspect

from pypath_common import misc as _common

from . import _node


class Noi(collections.abc.Mapping):

    _DEFAULTS = {
        'organism': 9606,
    }

    def __init__(
            self,
            noi: Noi | list[str] | list[list[str]] | dict[str, list[str]],
            groups: list[str] | dict[str, str] | None = None,
            id_type: str | None = None,
            entity_type: str | None = None,
            organism: int | str | None = 9606,
        ):
        """
        Nodes of interest.

        Args:
            noi:
                Nodes of interest, accepted notations are as follows:
                    - One string: one single node
                    - Iterable of strings: multiple nodes, grouping can be
                      specified by ``groups``
                    - Iterable of interables of strings: multiple groups of
                      nodes, group names can be specified by ``groups``
        """

        self._defaults = {
            'id_type': id_type,
            'entity_type': entity_type,
            'organism': organism,
        }
        self._setup(noi, groups)


    def _setup(
            self,
            noi: Noi | list[str] | list[list[str]] | dict[str, list[str]],
            groups: list[str] | dict[str, str] | None = None,
        ):

        self.nodes = self._parse(noi, groups)


    def _parse(
            self,
            noi: Noi | list[str] | list[list[str]] | dict[str, list[str]],
            groups: list[str] | dict[str, str] | None = None,
            id_type: str | None = 'uniprot',
            entity_type: str | None = 'protein',
            organism: int | str | None = 9606,
        ) -> dict[str, list[str]]:

        if isinstance(noi, Noi):

            return noi.nodes

        if isinstance(noi, dict):

            groups, noi = zip(*(
                (grp, n)
                for grp, nodes in noi.items()
                for n in nodes
            ))
            groups = list(groups)

        noi = [self._parse_node(n, locals()) for n in _common.to_list(noi)]

        if isinstance(groups, dict):

            groups = [groups[n] for n in noi]

        if isinstance(groups, list):

            if len(noi) != len(groups):

                raise ValueError('Nodes and groups are not the same length')

            _noi = collections.defaultdict(list)
            [_noi[g].append(n) for n, g in zip(noi, groups)]

            groups = dict(_noi)

        return groups or {'noi': noi}


    def _parse_node(
            self,
            node: str | _node.Node | tuple | dict,
            attrs: dict,
        ) -> _node.Node:

        return self.parse_node(node, self._defaults, attrs)


    @classmethod
    def parse_node(
            cls,
            node: str | _node.Node | tuple | dict,
            *attrs: dict,
        ) -> _node.Node:
        """
        Parse various node notations into Node object.

        Args:
            attrs:
                A series of dicts with node properties, from lowest to highest
                priority; elements with None values will be ignored.
        """

        if not isinstance(node, _node.Node):

            node_attrs = _node.Node._attrs

            if isinstance(node, str):

                node = (node,)

            if isinstance(node, tuple):

                node = dict(zip(node_attrs, node))

            if isinstance(node, dict):

                attrs += (node,)

            _attrs = cls._DEFAULTS.copy()

            for a in attrs:

                _attrs.update({
                    k: v
                    for k, v in (a or {}).items()
                    if k in node_attrs and v
                })

            if 'identifier' not in _attrs:

                raise ValueError(
                    'Nodes must have identifiers. '
                    f'Could not find identifier for `{node}`.',
                )

            node = _node.Node(**_attrs)

        return node


    def __getitem__(self, key: Hashable) -> Any:

        return self.nodes[key]


    def __iter__(self):

        return self.nodes.__iter__()


    def __len__(self) -> int:

        return len(self.nodes)


    def __delitem__(self, key: Hashable):

        del self.nodes[key]


    def __setitem__(self, key: Hashable, value: Any):

        self.nodes[key] = value


    def as_idtype(self, id_type: str, entity_type: str = 'protein') -> Noi:
        """
        Translate nodes to a certain ID type.

        The translation applies to one entity type in one call, the rest of
        the nodes will be included unchanged.
        """

        return self.apply(
            lambda n: n.as_idtype(id_type),
            entity_type = entity_type,
        )


    def as_organism(
            self,
            organism: str | int = 10090,
            source_organism: str | int = 9606,
            entity_type: str = 'protein',
        ) -> Noi:
        """
        Translate nodes to a certain ID type.

        The translation applies to one entity type in one call, the rest of
        the nodes will be included unchanged.
        """

        return self.apply(
            lambda n: n.as_organism(organism),
            entity_type = entity_type,
        )


    def apply(self, proc: Callable, *args, **kwargs) -> Noi:
        """
        Apply a certain processing step to all matching nodes.

        Non matching nodes will be included unchanged.
        """

        return Noi(
            {
                grp: list(itertools.chain(*(
                    proc(n) if n.match(*args, **kwargs) else (n,)
                    for n in nodes
                )))
                for grp, nodes in self.nodes.items()
            }
        )


    def keep(self, *args, **kwargs) -> Noi:
        """
        Keep only matching nodes.
        """

        return self._filter(_common.identity, *args, **kwargs)


    def drop(self, *args, **kwargs) -> Noi:
        """
        Drop matching nodes.
        """

        return self._filter(_common.negate, *args, **kwargs)


    def _filter(self, op: Callable, *args, **kwargs) -> Noi:

        return Noi({
            grp: [n for n in nodes if op(n.match(*args, **kwargs))]
            for grp, nodes in self.nodes.items()
        })


    @classmethod
    def from_json(cls, path: str) -> Noi:
        """
        Read nodes of interest from JSON.

        The JSON contents should result a valid object for the ``noi``
        argument or all arguments of the class.
        """

        return cls._from_file(path, json.load)


    @classmethod
    def from_yaml(cls, path: str) -> Noi:
        """
        Read nodes of interest from YAML (YML).

        The YAML contents should result a valid object for the ``noi``
        argument or all arguments of the class.
        """

        return cls._from_file(path, yaml.safe_load)


    @classmethod
    def _from_file(cls, path: str, module: Callable) -> Noi:

        with open(path, 'r') as fp:

            contents = module.load(fp)

        param = set(list(inspect.signature(cls.__init__).parameters)[1:])

        if not (isinstance(contents, dict) and set(contents.keys()) & param):

            contents = {'noi': contents}

        return cls(**contents)
