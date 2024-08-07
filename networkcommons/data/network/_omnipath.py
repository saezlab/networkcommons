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

__all__ = ['get_omnipath']

import lazy_import
import numpy as np

# op = lazy_import.lazy_module('omnipath')
import omnipath as op


def get_omnipath(genesymbols: bool = True, directed_signed: bool = True):
    """
    Retrieves the Omnipath network with directed interactions
    and specific criteria.

    Returns:
        network (pandas.DataFrame): Omnipath network with
        source, target, and sign columns.
    """
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
        network = network[(network['consensus_direction']) &
                          (
                              (network['consensus_stimulation']) |
                              (network['consensus_inhibition'])
                          ) &
                          (network['curation_effort'] >= 2)]

    network['sign'] = np.where(network['consensus_stimulation'], 1, -1)

    # write the resulting omnipath network in networkx format
    network = network[['source', 'target', 'sign']].reset_index(drop=True)

    return network
