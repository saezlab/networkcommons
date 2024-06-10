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

import numpy as np
import pandas as pd

from networkcommons import _utils


def build_moon_regulons(include_liana=False):

    import decoupler as dc
    from . import _omnipath

    dorothea_df = dc.get_collectri()

    TFs = np.unique(dorothea_df['source'])

    full_pkn = _omnipath.get_omnipath(genesymbols=True, directed_signed=True)

    if include_liana:
        from . import _lianaplus
        ligrec_resource = _lianaplus.get_lianaplus()

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
