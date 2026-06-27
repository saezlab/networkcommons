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

import os
import urllib

import pandas as pd

from networkcommons import _conf
from networkcommons.data.omics import _common
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
    Retrieves HMDB ID to metabolite name mapping via omnipath-client.

    Extracts HMDB identifiers from the COSMOS PKN and resolves them to
    human-readable metabolite names using the OmniPath entity resolution
    service (metabo.omnipathdb.org).

    Note: this implementation uses ``omnipath_client.OmniPath()._fetch``
    (a private method) because ``omnipath_client.utils.translate`` does not
    yet support HMDB → name translation — the ``utils.omnipathdb.org``
    service only covers cross-database ID mapping (e.g. HMDB → ChEBI).
    Once ``oc.utils.translate('hmdb', 'traditional_iupac')`` is supported
    server-side this function should be updated to use the public API.

    Args:
        update: Force re-resolution even if cached.

    Returns:
        dict: Mapping of HMDB IDs (e.g. ``'HMDB0000122'``) to metabolite
        names (e.g. ``'glucose'``).
    """
    import re
    import omnipath_client as oc

    path = os.path.join(_conf.get('pickle_dir'), 'hmdb_mapper.pickle')

    _log('MOON: Retrieving HMDB mapper...')

    if update or not os.path.exists(path):
        _log('MOON: HMDB mapper not found in cache. Resolving via OmniPath...')

        pkn = get_cosmos_pkn()
        all_nodes = set(pkn['source']).union(pkn['target'])
        hmdb_ids = list({
            m.group(1)
            for n in all_nodes
            for m in [re.search(r'(HMDB\d+)', str(n))]
            if m
        })

        # _fetch hits the entities/resolve endpoint on metabo.omnipathdb.org,
        # which aggregates names from ChEBI, HMDB, RefMet, etc.
        client = oc.OmniPath()
        mapper = {}
        batch_size = 200

        for i in range(0, len(hmdb_ids), batch_size):
            batch = hmdb_ids[i:i + batch_size]
            result = client._fetch('entities/resolve', identifiers=batch)

            for match in result.get('matches', []):
                hmdb_id = match.get('identifier')
                candidates = match.get('candidates', [])
                if not candidates or not hmdb_id:
                    continue
                idents = candidates[0].get('identifiers', [])
                names = [
                    x['identifier'] for x in idents
                    if x.get('identifierType') == 'Iupac Traditional Name:OM:0211'
                ]
                if not names:
                    names = [
                        x['identifier'] for x in idents
                        if x.get('identifierType') == 'Name:OM:0202'
                    ]
                if names:
                    mapper[hmdb_id] = names[0]

        pd.to_pickle(mapper, path)

    else:
        _log('MOON: HMDB mapper found in cache. Loading...')
        mapper = pd.read_pickle(path)

    _log(f'MOON: Done. HMDB mapper has {len(mapper)} entries.')

    return mapper
