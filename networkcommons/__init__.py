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
Integrated framework for network inference and evaluation
using prior knowledge and omics data.
"""

__all__ = [
    '__version__',
    '__author__',
]

from ._metadata import __author__, __version__
from ._session import log, _log, session
from ._conf import config, get, setup
from .utils import *  # noqa: F401 F403
from .methods import *  # noqa: F401 F403
from .evaluation import *  # noqa: F401 F403
from .moon import *  # noqa: F401 F403
from . import _data as data
from . import _noi as noi
from . import _visual as visual
