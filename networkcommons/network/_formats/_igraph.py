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

from collections.abc import Iterable

from pypath_common import _misc

from networkcommons import _imports


with _imports.optional():

    import igraph



def to_igraph(
        net: 'NetworkBase',
        eattrs: str | Iterable[str] | None = None,
        nattrs: str | Iterable[str] | None = None,
    ) -> 'igraph.Graph':

    net._not_hyper('Can not convert hypergraph to igraph.')

    g = igraph.Graph()
    g.add_vertices(net.ncount)

    eattrs = _misc.to_set(eattrs)
    nattrs = _misc.to_set(nattrs)
