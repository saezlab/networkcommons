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
Prior knowledge network used by MOON.
"""

__all__ = ['get_cosmos_pkn']

import lazy_import
import numpy as np
import pandas as pd

from networkcommons import utils
from . import _omnipath
from . import _liana

import os
import urllib
from networkcommons import _conf
from networkcommons.data.omics import _common

# dc = lazy_import.lazy_module('decoupler')
import decoupler as dc
from networkcommons._session import _log

def get_cosmos_pkn(update: bool = False):
    """
    Retrieves the metabolic network used in COSMOS from the server

    Returns:
        network (pandas.DataFrame): metabolic network with
        source, target, and sign columns.
    """
    path = os.path.join(_conf.get('pickle_dir'), 'metapkn.pickle')
    
    _log('COSMOS: Retrieving prior knowledge network...')

    if update or not os.path.exists(path):
        _log('COSMOS: Network not found in cache. Downloading...')

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'prior_knowledge')

        file_legend = pd.read_csv(baseurl + '/meta_network.sif', sep='\t')

        # removing duplicated interactions
        file_legend = file_legend.drop_duplicates(subset=['source', 'target', 'sign'], keep='first')
        file_legend = file_legend.drop_duplicates(subset=['source', 'target'], keep=False)

        file_legend.to_pickle(path)

    else:
        _log('COSMOS: Network found in cache. Loading...')

        file_legend = pd.read_pickle(path)

    _log(f'COSMOS: Done. Network has {len(file_legend)} interactions.')


    return file_legend
