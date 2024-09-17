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

__all__ = ['Bootstrap', 'BootstrapDf', 'BootstrapCopy']

from ._df import BootstrapDf
from ._edgelist import Bootstrap
from ._copy import BootstrapCopy
