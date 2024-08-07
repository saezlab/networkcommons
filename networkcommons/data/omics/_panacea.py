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
RNA-Seq data from the 'Pancancer Analysis of Chemical Entity Activity'
resource.
"""

from __future__ import annotations

__all__ = ['panacea']

import pandas as pd

from . import _common


def panacea() -> tuple[pd.DataFrame]:
    """
    Pancancer Analysis of Chemical Entity Activity RNA-Seq data.

    Returns:
        Two data frames: counts and meta data.
    """

    return tuple(
        _common._open(
            _common._commons_url('panacea', table = table),
            df = {'sep': '\t'},
        )
        for table in ('count', 'meta')
    )
