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

from __future__ import annotations

import os
import urllib.parse

import pandas as pd

from . import _common
from networkcommons import _conf

__all__ = ['decryptm_datasets', 'decryptm_handler']


def decryptm_datasets(update: bool = False) -> pd.DataFrame:
    """
    Table of all DecryptM datasets.

    Args:
        update:
            Force download and update cache.

    Returns:
        Data frame of all DecryptM datasets, with columns "experiment",
        "data_type" and "fname".
    """

    path = os.path.join(_conf.get('pickle_dir'), 'decryptm_datasets.pickle')

    if update or not os.path.exists(path):

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'decryptm')

        datasets = pd.DataFrame(
            [
                (
                    experiment,
                    data_type,
                    fname,
                )
                for experiment in _common._ls(baseurl)
                for data_type in _common._ls(f'{baseurl}/{experiment}')
                for fname in _common._ls(f'{baseurl}/{experiment}/{data_type}')
                if fname.startswith('curve')
            ],
            columns = ['experiment', 'data_type', 'fname']
        )
        datasets.to_pickle(path)

    else:

        datasets = pd.read_pickle(path)

    return datasets


def decryptm_handler(experiment, data_type='Phosphoproteome'):
    """

    """
    save_path = f'./data/decryptm/{experiment}/{data_type}/'

    curve_files = list_directories(f'decryptm/{experiment}/{data_type}')[1]

    curve_files = [
        os.path.basename(file) for file in curve_files if 'curves' in file
    ]

    for curve_file in curve_files:
        if not os.path.exists(save_path + curve_file):
            download_url(
                f'https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B/download?path=%2Fdecryptm%2F{experiment}%2F{data_type}&files={curve_file}',  # noqa: E501
                save_path + curve_file
            )  # noqa: E501

    file_list = {}
    for curve_file in curve_files:
        df = pd.read_csv(
            f'./data/decryptm/{experiment}/{data_type}/{curve_file}',
            sep='\t'
        )
        file_list[curve_file] = df

    return file_list
