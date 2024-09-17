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
DecryptM proteomics and phosphoproteomics data.
"""

from __future__ import annotations

__all__ = ['decryptm_datasets', 'decryptm_table', 'decryptm_experiment']

import os
import urllib.parse

import pandas as pd

from . import _common
from networkcommons import _conf
from networkcommons._session import _log


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


def decryptm_table(experiment: str, data_type: str, fname: str) -> pd.DataFrame:
    """
    One table of omics data from DecryptM.

    Args:
        experiment:
            Name of the experiment. For a complete list see
            `decryptm_datasets()`.
        data_type:
            Omics modality of the data. For the available modalities in each
            experiment, see `decryptm_datasets()`.
        fname:
            Name of the table in the experiment.

    Returns:
        The table as a pandas data frame.
    """

    _log(f'DATA: Retrieving DecryptM table `{experiment}/{data_type}/{fname}`...')

    return _common._open(
        _common._commons_url('decryptm', **locals()),
        df = {'sep': '\t'},
    )


def decryptm_experiment(
        experiment: str,
        data_type: str = 'Phosphoproteome',
    ) -> list[pd.DataFrame]:
    """
    Tables from one DecrptM experiment of one omics modality.

    Args:
        experiment:
            Name of the experiment. For a complete list see
            `decryptm_datasets()`.
        data_type:
            Omics modality of the data. For the available modalities in each
            experiment, see `decryptm_datasets()`.

    Returns:
        All tables in the seleted dataset as a list of pandas data frames.
    """

    datasets = {
        k: g.fname.tolist()
        for k, g in decryptm_datasets().groupby(['experiment', 'data_type'])
    }
    key = (experiment, data_type)

    if key not in datasets:

        raise ValueError(
            f'No such dataset in DecryptM: `{experiment}/{data_type}`.'
        )

    return [
        decryptm_table(experiment, data_type, fname)
        for fname in datasets[key]
    ]
