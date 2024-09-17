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

__all__ = ['nci60_datasets', 'nci60_datatypes', 'nci60_table']

import pandas as pd
import os
import urllib.parse

from . import _common
from networkcommons import _conf

from networkcommons._session import _log


def nci60_datasets(update: bool = False) -> pd.DataFrame:
    """
    Table of all NCI60 datasets (cell types).

    Args:
        update:
            Force download and update cache.

    Returns:
        Data frame of all NCI60 datasets, with columns "experiment",
        "data_type" and "fname".
    """

    path = os.path.join(_conf.get('pickle_dir'), 'nci60_datasets.pickle')

    if update or not os.path.exists(path):

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'NCI60')

        datasets = pd.DataFrame(
            [
                (
                    cell_line,
                )
                for cell_line in _common._ls(baseurl)
            ],
            columns = ['cell_line']
        )
        datasets.to_pickle(path)

    else:

        datasets = pd.read_pickle(path)

    return datasets


def nci60_datatypes() -> pd.DataFrame:
    """
    Table of all NCI60 data types.

    Returns:
        Data frame of all NCI60 data types, with columns "data_type",
        and "description".
    """
    df = pd.DataFrame({
        'data_type': ['TF_scores', 'RNA', 'metabolomic'],
        'description': ['TF scores', 'RNA expression', 'metabolomic data']
        }
    )

    return df


def nci60_table(cell_line: str, data_type: str) -> pd.DataFrame:
    """
    One table of omics data from NCI60.

    Args:
        cell_line:
            Name of the cell line. For a complete list see
            `nci60_datasets()`.
        data_type:
            Type of data. For a complete list see `nci60_datatypes()`.

    Returns:
        The table as a pandas DataFrame.
    """
    _log(f"DATA: Retrieving NCI60 data for {cell_line} ({data_type})...")

    return _common._open(
        _common._commons_url('NCI60', **locals()),
        df = {'sep': '\t'},
    )
