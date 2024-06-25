from __future__ import annotations

import pandas as pd
import corneto

from networkcommons._data import _network as _universe
from networkcommons._noi._noi import Noi


class Network:

    _co: corneto.Graph
    _edges: pd.DataFrame
    _nodes: pd.DataFrame

    def __init__(
        self,
        universe: str | None = "omnipath",
        noi: Noi | list[str] | list[list[str]] | dict[str, list[str]] = None,
    ):

        self.universe = universe
        self.noi = noi


    def _load(self):

        self._co = _universe.getattr(f'get_{self.universe}')()


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


    def as_corneto(self, attrs: str | list[str]) -> corneto.Graph:
        """
        Return the graph as an igraph object with the desired attributes.
        """

        return self._co.copy()
