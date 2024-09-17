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
Prior knowledge network used by LIANA+.
"""

__all__ = ['get_lianaplus']

# import lazy_import

# liana = lazy_import.lazy_module('liana')

import pandas as pd

from networkcommons._session import _log
import os
import urllib
from networkcommons import _conf
from networkcommons.data.omics import _common


def get_lianaplus(resource='consensus', update: bool = False):
    """
    Retrieves the metabolic network used in COSMOS from the server

    Returns:
        network (pandas.DataFrame): metabolic network with
        source, target, and sign columns.
    """
    path = os.path.join(_conf.get('pickle_dir'), 'lianaplus.pickle')

    _log('LIANA+: Retrieving prior knowledge network...')

    if update or not os.path.exists(path):
        _log('LIANA+: Network not found in cache. Downloading...')

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'prior_knowledge')

        file_legend = pd.read_csv(baseurl + '/liana_ligrec.csv', sep=',')

        # removing duplicated interactions
        file_legend = file_legend[file_legend['resource'] == resource]

        if file_legend.empty:
            _log(f'LIANA+: No data found for resource {resource}')
            return

        file_legend = file_legend[['source_genesymbol', 'target_genesymbol']].drop_duplicates()
        file_legend.columns = ['source', 'target']
        file_legend['sign'] = 1

        file_legend.to_pickle(path)

    else:
        _log('LIANA+: Network found in cache. Loading...')

        file_legend = pd.read_pickle(path)

    _log(f'LIANA+: Done. Network has {len(file_legend)} interactions.')


    return file_legend
