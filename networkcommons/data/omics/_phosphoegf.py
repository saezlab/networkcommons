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
Meta-analysis of phosphoproteomics response to EGF stimulus.
"""

import pandas as pd
import warnings

def phospho_egf_datatypes() -> pd.DataFrame:
    """
    Table describing the available data types in the Phospho EGF dataset.

    Returns:
        DataFrame with all data types.
    """

    return pd.DataFrame({
        'type': ['phosphosite', 'kinase'],
        'description': ['Differential phosphoproteomics at the site level for all studies in the meta-analysis',
                        'Kinase activities obtained using each of the kinase-substrate prior knowledge resources'],
    })


def phospho_egf_tables(type='phosphosite'):
    """
    A table with the corresponding data type for the phospho EGF dataset. 

    Args:
        type:
            Either 'phosphosite' or 'kinase'.

    Returns:
        A DataFrame with the corresponding data.
    """

    if type == 'phosphosite':
        out_table = pd.read_csv('https://www.biorxiv.org/content/biorxiv/early/2024/10/22/2024.10.21.619348/DC3/embed/media-3.gz', compression='gzip', low_memory=False)
    elif type == 'kinase':
        out_table = pd.read_csv('https://www.biorxiv.org/content/biorxiv/early/2024/10/22/2024.10.21.619348/DC4/embed/media-4.gz', compression='gzip', low_memory=False)
    else:
        warnings.warn(f'Unknown data type "{type}"')
        return None
    return out_table
        
