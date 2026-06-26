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

__all__ = ['get_cosmos_pkn', 'get_hmdb_mapper']

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


def get_hmdb_mapper(update: bool = False) -> dict:
    """
    Retrieves the HMDB ID to metabolite name mapping from cosmosR.

    Downloads ``HMDB_mapper_vec.RData`` from the cosmosR GitHub repository
    and converts it to a Python dict mapping HMDB IDs to human-readable
    metabolite names.

    Args:
        update: Force re-download even if cached.

    Returns:
        dict: Mapping of HMDB IDs (e.g. ``'HMDB0000122'``) to metabolite
        names (e.g. ``'Glucose'``).
    """
    import rdata as _rdata

    path = os.path.join(_conf.get('pickle_dir'), 'hmdb_mapper.pickle')

    _log('MOON: Retrieving HMDB mapper...')

    if update or not os.path.exists(path):
        _log('MOON: HMDB mapper not found in cache. Downloading...')

        url = (
            'https://raw.githubusercontent.com/saezlab/cosmosR/'
            'master/data/HMDB_mapper_vec.RData'
        )
        rdata_path = _common._maybe_download(url)

        parsed = _rdata.parser.parse_file(rdata_path)
        obj = parsed.object.value[0]
        values = obj.value
        names = obj.attributes.value[0].value

        hmdb_ids = [x.value.decode() for x in names]
        metab_names = [values[i].value.decode() for i in range(len(values))]

        mapper = dict(zip(hmdb_ids, metab_names))

        pd.to_pickle(mapper, path)

    else:
        _log('MOON: HMDB mapper found in cache. Loading...')
        mapper = pd.read_pickle(path)

    _log(f'MOON: Done. HMDB mapper has {len(mapper)} entries.')

    return mapper
