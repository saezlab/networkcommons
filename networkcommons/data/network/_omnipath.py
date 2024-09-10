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
Access to the OmniPath database.
"""

__all__ = ['get_omnipath', 'get_phosphositeplus']

import lazy_import
import numpy as np
import os
import urllib
from networkcommons import _conf
from networkcommons.data.omics import _common
import pandas as pd

# op = lazy_import.lazy_module('omnipath')
import omnipath as op
from networkcommons._session import _log


def get_omnipath(genesymbols: bool = True, directed_signed: bool = True):
    """
    Retrieves the Omnipath network with directed interactions
    and specific criteria.

    Returns:
        network (pandas.DataFrame): Omnipath network with
        source, target, and sign columns.
    """
    _log('OmniPath: Retrieving prior knowledge network...')

    network = op.interactions.AllInteractions.get('omnipath',
                                                  genesymbols=genesymbols)

    network.rename(columns={'source': 'source_uniprot',
                            'target': 'target_uniprot'},
                   inplace=True)
    network.rename(columns={'source_genesymbol': 'source',
                            'target_genesymbol': 'target'},
                   inplace=True)

    # get only directed and signed interactions
    if directed_signed:
        _log('OmniPath: Filtering for directed and signed interactions...')
        network = network[(network['consensus_direction']) &
                          (
                              (network['consensus_stimulation']) |
                              (network['consensus_inhibition'])
                          ) &
                          (network['curation_effort'] >= 2)]

    network['sign'] = np.where(network['consensus_stimulation'], 1, -1)

    # write the resulting omnipath network in networkx format
    network = network[['source', 'target', 'sign']].reset_index(drop=True)

    _log(f'OmniPath: Done. Network has {len(network)} interactions.')

    return network


def get_phosphositeplus(update: bool = False):
    """
    Retrieves the PhosphoSitePlus network from the server

    Returns:
        network (pandas.DataFrame): PhosphoSitePlus network with
        source, target, and sign columns.
    """
    _log('PhosphoSitePlus: Retrieving prior knowledge network...')

    path = os.path.join(_conf.get('pickle_dir'), 'phosphositeplus.pickle')

    if update or not os.path.exists(path):
        _log('PhosphoSitePlus: Network not found in cache. Downloading...')

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'prior_knowledge')

        file_legend = pd.read_csv(baseurl + '/kinase-substrate.tsv', sep='\t')

        file_legend.to_pickle(path)

    else:
        _log('PhosphoSitePlus: Network found in cache. Loading...')

        file_legend = pd.read_pickle(path)
    
    _log(f'PhosphoSitePlus: Done. Network has {len(file_legend)} interactions.')

    return file_legend
