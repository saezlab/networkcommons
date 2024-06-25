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
    'log',
    'session',
    'config',
    'setup',
    'data',
    'eval',
    '_methods',
    'noi',
    'utils',
    'visual',
]

from ._metadata import __author__, __version__
from ._session import log, _log, session
from ._conf import config, setup

from . import _data as data
from . import _eval as eval
from . import _methods as _methods
from . import _noi as noi
from . import _utils as utils
from . import _visual as visual
