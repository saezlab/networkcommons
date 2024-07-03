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
Access to Moon example omics data.
"""

from __future__ import annotations

import pandas as pd

from . import common

__all__ = ['moon']


def moon() -> dict[str, pd.DataFrame]:
    """
    Example data for Moon.

    Returns:
        Three data frames: signaling, metabolite and gene activity
        measurements.
    """

    return {
        table: _common._open(
            _common._commons_url('moon', table = table),
            df = {'sep': '\t'},
        )
        for table in ('sig', 'metab', 'rna')
    }
