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
CPTAC (Clinical Proteomic Tumor Analysis Consortium data) 
transcriptomics, proteomics and phosphoproteomics data.
"""

from __future__ import annotations

__all__ = ['cptac_fileinfo', 'cptac_cohortsize', 'cptac_table', 'cptac_datatypes']

import os
import urllib.parse

import pandas as pd

from . import _common
from networkcommons import _conf


def cptac_fileinfo(update: bool = False) -> pd.DataFrame:
    """
    Table describing the files contained in the CPTAC dataset.

    Args:
        update:
            Force download and update cache.

    Returns:
        Data frame explaining the contents of each file.
    """
    pd.set_option('display.max_colwidth', None)

    path = os.path.join(_conf.get('pickle_dir'), 'cptac_info.pickle')

    if update or not os.path.exists(path):

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'CPTAC')

        file_legend = pd.read_excel(baseurl + '/CPTAC_pancancer_data_freeze_file_description.xlsx')

        file_legend.to_pickle(path)

    else:

        file_legend = pd.read_pickle(path)

    return file_legend


def cptac_datatypes() -> list:
    """
    List describing the data types available in the CPTAC dataset.

    Returns:
        Data frame with data type information.
    """
    baseurl = urllib.parse.urljoin(_common._baseurl(), 'CPTAC')

    directories = _common._ls(baseurl)

    files = ['CPTAC_pancancer_data_freeze_cohort_size.xlsx', 
             'CPTAC_pancancer_data_freeze_file_description.xlsx']
    directories = [directory for directory in directories if directory not in files]

    return directories


def cptac_cohortsize(update: bool = False) -> pd.DataFrame:
    """
    Table describing the number of tumor and normal samples per CPTAC 
    cancer type.

    Args:
        update:
            Force download and update cache.

    Returns:
        Data frame with cohort size information.
    """

    path = os.path.join(_conf.get('pickle_dir'), 'cptac_cohort.pickle')

    if update or not os.path.exists(path):

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'CPTAC')

        file_legend = pd.read_excel(baseurl + '/CPTAC_pancancer_data_freeze_cohort_size.xlsx')

        file_legend.to_pickle(path)

    else:

        file_legend = pd.read_pickle(path)

    return file_legend


def cptac_table(data_type: str, cancer_type: str, fname: str) -> pd.DataFrame:
    """
    One table of omics data from CPTAC.

    Args:
        data_type:
            Type of data. For a complete list see `cptac_datatypes()`.
        cancer_type:
            Name of the cancer type. For a complete list see
            `cptac_cohortsize()`.
        fname:
            File name. For the available files and description,
            see `cptac_fileinfo()`.

    Returns:
        The table as a pandas DataFrame.
    """

    return _common._open(
        _common._commons_url('CPTAC', **locals()),
        df = {'sep': '\t'},
    )