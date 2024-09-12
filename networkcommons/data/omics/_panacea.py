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
RNA-Seq data from the 'Pancancer Analysis of Chemical Entity Activity'
resource.
"""

from __future__ import annotations

__all__ = ['panacea_experiments', 'panacea_datatypes', 'panacea_tables', 'panacea_gold_standard']

import pandas as pd
import os
import urllib.parse

from . import _common

from networkcommons import _conf
from networkcommons._session import _log


def panacea_experiments(update=True) -> pd.DataFrame:
    """
    Table describing the experiments (drug-cell combinations) contained
    in the Panacea dataset and merging it with the presence of TF_scores.

    Args:
        update (bool): Whether to update the dataframe by fetching new metadata.
        tf_scores_dir (str): Directory where TF_scores files are located.

    Returns:
        Data frame with all drug-cell line combinations and TF_scores availability.
    """

    path = os.path.join(_conf.get('pickle_dir'), 'panacea_exps.pickle')

    if update or not os.path.exists(path):

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'panacea')

        file_legend = pd.read_csv(baseurl + '/panacea__metadata.tsv', sep='\t')

        processed_dir = baseurl + '/processed'

        file_legend[['cell', 'drug']] = file_legend['group'].str.split('_', expand=True)
        file_legend.drop(columns='sample_ID', inplace=True)
        file_legend.drop_duplicates(inplace=True)
        file_legend.reset_index(drop=True, inplace=True)

        response = urllib.request.urlopen(processed_dir)
        tf_scores_html = response.read().decode('utf-8')

        tf_scores_files = []
        for line in tf_scores_html.splitlines():
            if '__TF_scores.tsv' in line:
                filename = line.split('"')[1]
                tf_scores_files.append(filename)

        tf_scores_groups = [f.split('__TF_scores.tsv')[0] for f in tf_scores_files]

        file_legend['tf_scores'] = file_legend['group'].apply(lambda x: x in tf_scores_groups)

        file_legend.to_pickle(path)


    else:
        file_legend = pd.read_pickle(path)

    return file_legend


def panacea_datatypes() -> pd.DataFrame:
    """
    Table describing the available data types in the Panacea dataset.

    Returns:
        Data frame with all data types.
    """

    return pd.DataFrame({
        'type': ['raw', 'diffexp', 'TF_scores'],
        'description': ['RNA-Seq raw counts and metadata containing sample, name, and group',
                        'Differential expression analysis with filterbyExpr+DESeq2',
                        'Transcription factor activity scores with CollecTRI + T-values'],
    })


def panacea_tables(cell_line=None, drug=None, type='raw'):
    """
    One table of countdata and one table of metadata from Panacea if raw data is selected.
    If diffexp or TF_scores is selected, the corresponding table is returned.

    Args:
        cell_line:
            Name of the cell line(s). For a complete list see `panacea_experiments()`.
        drug:
            Name of the drug(s). For a complete list see `panacea_experiments()`.
        type:
            Type of data. For a complete list see `panacea_datatypes()`.

    Returns:
        tuple[pd.DataFrame]: Two data frames: counts and meta data.
    """
    _log(f"DATA: Retrieving Panacea {type} table for cell line {cell_line} and drug {drug}...")
    if (cell_line is None or drug is None) and type != 'raw':
        raise ValueError('Please specify cell line and drug.')

    if type == 'raw':
        df_meta = _common._open(
            _common._commons_url('panacea', table='metadata'),
            df = {'sep': '\t'},
        )

        df_meta[['cell', 'drug']] = df_meta['group'].str.split('_', expand=True)

        if isinstance(cell_line, str):
            cell_line = [cell_line]

        if isinstance(drug, str):
            drug = [drug]

        if cell_line is not None:
            df_meta = df_meta[df_meta['cell'].isin(cell_line)]

        if drug is not None:
            df_meta = df_meta[df_meta['drug'].isin(drug)]

        df_count = _common._open(
            _common._commons_url('panacea', table='countdata'),
            df={'sep': '\t'},
        )

        subset_cols = df_meta['sample_ID'].tolist()
        df_count = df_count.loc[:, ['gene_symbol'] + subset_cols]

        return df_count, df_meta

    elif type == 'diffexp' or type == 'TF_scores':
        baseurl = urllib.parse.urljoin(_common._baseurl(), 'panacea/processed')

        proc_file = pd.read_csv(baseurl + f'/{cell_line}_{drug}__{type}.tsv', sep='\t')

        return proc_file

    else:
        raise ValueError(f'Unknown data type: {type}.')


def panacea_gold_standard(update: bool = False) -> pd.DataFrame:
    """
    Retrieves the gold standard (primary targets and offtargets linked to drugs) 
    used in Panacea from the server. To be used in the offtarget evaluation strategy.

    Returns:
        network (pandas.DataFrame): gold standard network with
        cmpd, cmpd_id, target and rank
    """
    _log('DATA: Retrieving Panacea offtarget gold standard...')
    path = os.path.join(_conf.get('pickle_dir'), 'panacea_gold_standard.pickle')

    if update or not os.path.exists(path):
        _log('DATA: not found in cache, downloading from server...')

        baseurl = urllib.parse.urljoin(_common._baseurl(), 'panacea')

        file_legend = pd.read_csv(baseurl + '/panacea_gold_standard.csv')

        file_legend.to_pickle(path)

    else:
        _log('DATA: found in cache, loading...')

        file_legend = pd.read_pickle(path)

    return file_legend
