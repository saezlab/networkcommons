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

__all__ = ['build_moon_regulons', 'get_cosmos_pkn']

import lazy_import
import numpy as np
import pandas as pd

from networkcommons import _utils
from . import _omnipath
from . import _liana

import os
import urllib
from networkcommons import _conf
from networkcommons.data.omics import _common

# dc = lazy_import.lazy_module('decoupler')
import decoupler as dc


def get_cosmos_pkn(update: bool = False):
    """
    Retrieves the metabolic network used in COSMOS from the server

    Returns:
        network (pandas.DataFrame): metabolic network with
        source, target, and sign columns.
    """
    path = os.path.join(_conf.get('pickle_dir'), 'metapkn.pickle')

    if update or not os.path.exists(path):

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'prior_knowledge')

        file_legend = pd.read_csv(baseurl + '/meta_network.sif', sep='\t')

        file_legend.to_pickle(path)

    else:

        file_legend = pd.read_pickle(path)

    return file_legend




def build_moon_regulons(include_liana=False):

    dorothea_df = dc.get_collectri()

    TFs = np.unique(dorothea_df['source'])

    full_pkn = _omnipath.get_omnipath(genesymbols=True, directed_signed=True)

    if include_liana:

        ligrec_resource = _liana.get_lianaplus()

        full_pkn = pd.concat([full_pkn, ligrec_resource])
        full_pkn['edgeID'] = full_pkn['source'] + '_' + full_pkn['target']

        # This prioritises edges coming from OP
        full_pkn = full_pkn.drop_duplicates(subset='edgeID')
        full_pkn = full_pkn.drop(columns='edgeID')

    kinTF_regulons = full_pkn[full_pkn['target'].isin(TFs)].copy()
    kinTF_regulons.columns = ['source', 'target', 'mor']
    kinTF_regulons = kinTF_regulons.drop_duplicates()

    kinTF_regulons = kinTF_regulons.groupby(['source', 'target']).mean() \
        .reset_index()

    layer_2 = {}
    activation_pkn = full_pkn[full_pkn['sign'] == 1].copy()

    pkn_graph = _utils.network_from_df(activation_pkn, directed=True)

    relevant_nodes = list(activation_pkn['source'].unique())
    relevant_nodes = [node for node in relevant_nodes if node in list(kinTF_regulons['source'])]

    for node in relevant_nodes:
        intermediates = activation_pkn[activation_pkn['source'] == node]['target'].tolist()
        targets = [n for i in intermediates for n in pkn_graph.neighbors(i)]
        targets = np.unique([n for n in targets if n in TFs])
        layer_2[node] = targets

    layer_2_df = pd.concat([pd.DataFrame({'source': k, 'target': v, 'mor': 0.25}) for k, v in layer_2.items()], ignore_index=True)
    kinTF_regulons = pd.concat([kinTF_regulons, layer_2_df])
    kinTF_regulons = kinTF_regulons.groupby(['source', 'target']).sum().reset_index()

    return kinTF_regulons
