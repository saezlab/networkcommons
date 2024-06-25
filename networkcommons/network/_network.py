from __future__ import annotations

from networkcommons._noi._noi import Noi

class Network:

    def __init__(self,
                 noi: Noi | list[str] | list[list[str]] | dict[str, list[str]],
                 universe: str | None = "omnipath"):
        self.universe = universe
        self.noi = noi




