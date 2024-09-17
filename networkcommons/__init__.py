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
    'methods',
    'utils',
    'visual',
]

import lazy_import

from ._metadata import __author__, __version__
from ._session import log, _log, session
from ._conf import config, setup

from . import utils as utils


_MODULES = [
    'data',
    'eval',
    'methods',
    'visual',
]

for _mod in _MODULES:

    globals()[_mod] = lazy_import.lazy_module(f'{__name__}.{_mod}')
