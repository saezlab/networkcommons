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
General procedures for downloading omics datasets.
"""

from __future__ import annotations

__all__ = ['datasets', 'get_ensembl_mappings', 'convert_ensembl_to_gene_symbol']

from typing import IO
import zipfile
import os
import hashlib
import contextlib
import functools as ft
import urllib.parse

import requests
import bs4
import pandas as pd

from networkcommons import _conf
from networkcommons._session import _log
from .._builtin import _module_data


def _datasets() -> dict[str, dict]:

    return _module_data('datasets').get('omics', {})


def datasets() -> pd.DataFrame:
    """
    Built-in omics datasets.

    Returns:
        A DataFrame with dataset details.
    """
    data = _datasets().get('datasets', {})
    df = pd.DataFrame.from_dict(data, orient='index')
    pd.set_option('display.max_colwidth', None)

    df = df[df.index != 'test']  # Exclude the 'test' dataset

    return df[['name', 'description', 'publication_link', 'detailed_description']]


def _baseurl() -> str:

    return _datasets()['baseurl']


def _commons_url(dataset: str, **kwargs) -> str:

    dsets = _datasets()
    baseurl = dsets['baseurl']
    path = dsets['datasets'][dataset]['path'].format(**kwargs)

    return urllib.parse.urljoin(baseurl, path)


def _requests_session() -> requests.Session:

    ses = requests.Session()
    retries = requests.adapters.Retry(
        total = _conf.get('http_retries'),
        backoff_factor = _conf.get('http_backoff_factor'),
        status_forcelist = _conf.get('http_status_forcelist'),
    )
    ses.mount('http://', requests.adapters.HTTPAdapter(max_retries = retries))

    return ses


def _download(url: str, path: str) -> None:

    timeouts = tuple(_conf.get(f'http_{k}_timout') for k in ('read', 'connect'))

    _log(f'Utils: Downloading `{url}` to `{path}`.')

    ses = _requests_session()

    with ses.get(url, timeout = timeouts, stream = True) as req:

        req.raise_for_status()

        with open(path, 'wb') as f:

            for chunk in req.iter_content(chunk_size = 8192):

                f.write(chunk)

    _log(f'Utils: Finished downloading `{url}` to `{path}`.')


def _maybe_download(url: str, **kwargs) -> str:

    url = url.format(**kwargs)
    cachedir = _conf.get('cachedir')
    md5 = hashlib.md5(url.encode()).hexdigest()
    fname = os.path.basename(urllib.parse.urlparse(url).path)
    path = os.path.join(cachedir, f'{md5}-{fname}')
    _log(f'Utils: Looking up in cache: `{url}` -> `{path}`.')

    if not os.path.exists(path):

        _log(f'Utils: Not found in cache, initiating download: `{url}`.')
        _download(url, path)

    return path


def _open(
        url: str,
        ftype: str | None = None,
        df: bool | dict = False,
        **kwargs
    ) -> IO | pd.DataFrame:
    """
    Args:
        url:
            URL of the file to open.
        ftype:
            File type (extension).
        df:
            Read into a pandas DataFrame. If a dict, will be passed as
            arguments to the reader.
        **kwargs:
            Values to insert into the URL template. Will be passed to
            `str.format`.
    """

    PANDAS_READERS = {
        'tsv': ft.partial(pd.read_table, sep = '\t'),
        'csv': pd.read_csv,
        'txt': pd.read_csv,
        'xls': pd.read_excel,
        'xlsx': pd.read_excel,
    }

    path = _maybe_download(url, **kwargs)
    ftype = (ftype or os.path.splitext(path)[1]).lower().strip('.')

    if not ftype:

        raise RuntimeError(f'Cannot determine file type for {url}.')


    if df is not False and ftype in PANDAS_READERS:

        df = df if isinstance(df, dict) else {}
        return PANDAS_READERS[ftype](path, **df)

    elif ftype in {'tsv', 'csv', 'txt'}:

        return contextlib.closing(open(path, 'r'))

    elif ftype == 'zip':

        return contextlib.closing(zipfile.ZipFile(path, 'r'))

    elif ftype in {'html', 'htm'}:

        with open(path, 'r') as fp:

            html = fp.read()

        return bs4.BeautifulSoup(html, 'html.parser')

    else:

        raise NotImplementedError(f'Can not open file type `{ftype}`.')


def _ls(path: str) -> list[str]:
    """
    List files in a remote directory.

    Args:
        path:
            HTTP URL of a directory with standard nginx directory listing.
    """

    ses = _requests_session()
    resp = ses.get(path)

    if resp.status_code == 200:

        soup = bs4.BeautifulSoup(resp.content, 'html.parser')

        return [
            href for a in soup.find_all('a')
            if (href := a['href'].strip('/')) != '..'
        ]

    else:

        raise FileNotFoundError(
            f'URL {path} returned status code {resp.status_code}'
        )


def get_ensembl_mappings(update: bool = False) -> pd.DataFrame:
    """
    Retrieves the mapping between Ensembl attributes for human genes.

    Args:
        update(Boolean): Force download and update cache.

    Returns:
        pandas.DataFrame: A DataFrame containing the mapping between Ensembl attributes.
            The DataFrame has the following columns:
            - ensembl_transcript_id: Ensembl transcript ID
            - gene_symbol: HGNC symbol for the gene
            - ensembl_gene_id: Ensembl gene ID
            - ensembl_peptide_id: Ensembl peptide ID
    """

    import biomart

    path = os.path.join(_conf.get('pickle_dir'), 'ensembl_map.pickle')

    _log('Utils: Retrieving Ensembl mappings...')

    if update or not os.path.exists(path):

        _log('Utils: Ensembl mappings not found in cache. Downloading...')

        # Set up connection to server
        server = biomart.BiomartServer('http://ensembl.org/biomart')
        mart = server.datasets['hsapiens_gene_ensembl']

        # List the types of data we want
        attributes = ['ensembl_transcript_id', 'hgnc_symbol', 'ensembl_gene_id', 'ensembl_peptide_id']

        # Get the mapping between the attributes
        response = mart.search({'attributes': attributes})
        data = response.raw.data.decode('ascii')

        # Convert the raw data into a list of tuples
        data_tuples = [tuple(line.split('\t')) for line in data.splitlines()]

        # Convert the list of tuples into a dataframe
        df = pd.DataFrame(data_tuples, columns=['ensembl_transcript_id', 'gene_symbol', 'ensembl_gene_id', 'ensembl_peptide_id'])

        # Melt the dataframe to long format
        melted_df = pd.melt(df, id_vars=['gene_symbol'], value_vars=['ensembl_transcript_id', 'ensembl_gene_id', 'ensembl_peptide_id'],
                            var_name='ensembl_type', value_name='ensembl_id')

        # Drop rows with empty 'ensembl_id' and drop the 'ensembl_type' column
        melted_df = melted_df[(melted_df['gene_symbol'] != '') & (melted_df['ensembl_id'] != '')].drop(columns=['ensembl_type'])

        # Set 'ensembl_id' as the index
        melted_df.drop_duplicates(inplace=True)

        melted_df = melted_df.reset_index(drop=True)

        melted_df.to_pickle(path)

    else:
        _log('Utils: Ensembl mappings found in cache. Loading...')
        melted_df = pd.read_pickle(path)
    
    _log(f'Utils: Ensembl mappings retrieved. Dataframe has {len(melted_df)} entries.')

    return melted_df


def convert_ensembl_to_gene_symbol(dataframe, equivalence_df, column_name='idx', summarisation=max):
    """
    Converts Ensembl IDs to gene symbols using an equivalence dataframe, handles partial matches,
    and summarizes duplicated entries by taking the maximum value.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe with Ensembl IDs.
        equivalence_df (pd.DataFrame): The equivalence dataframe with Ensembl IDs as index and gene symbols.
        You can either use a custom one or use the one retrieved by get_ensembl_mappings().
        column_name (str): The name of the column containing Ensembl IDs in the input dataframe.
        summarisation (function): The method to summarize duplicated entries.
    Returns:
        pd.DataFrame: The dataframe with gene symbols and summarized duplicated entries.
    """
    dataframe = dataframe.copy()
    equivalence_df = equivalence_df.copy()

    _log('Utils: Converting Ensembl IDs to gene symbols...')

    if column_name not in dataframe.columns:
        dataframe.reset_index(inplace=True)

    # Extract partial match from the Ensembl IDs in the input dataframe
    dataframe['partial_id'] = dataframe[column_name].str.extract(r'([A-Za-z0-9]+)', expand=False)

    # Reset index of equivalence dataframe for merging
    # Merge dataframes using partial matches
    merged_df = pd.merge(dataframe, equivalence_df, left_on='partial_id', right_on='ensembl_id', how='left')

    # Calculate and print the number and percentage of non-matched Ensembl IDs
    total_count = len(merged_df)
    non_matched_count = merged_df['gene_symbol'].isna().sum()
    non_matched_percentage = (non_matched_count / total_count) * 100
    _log(f"Utils: Number of non-matched Ensembl IDs: {non_matched_count} ({non_matched_percentage:.2f}%)")

    # Drop temporary and original index columns
    merged_df.drop(columns=['partial_id', 'ensembl_id', column_name], inplace=True)

    # Summarize duplicated entries by applying the summarisation function
    non_numeric_cols = merged_df.select_dtypes(exclude='number').columns.values
    summarized_df = merged_df.groupby(list(non_numeric_cols)).agg(summarisation).reset_index()

    # Calculate and print the number and percentage of summarized duplicated entries
    summarized_count = len(merged_df) - len(summarized_df)
    summarized_percentage = (summarized_count / total_count) * 100
    _log(f"Utils: Number of summarized duplicated entries: {summarized_count} ({summarized_percentage:.2f}%)")

    return summarized_df
