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

__all__ = ['datasets', 'get_ensembl_mappings']

from typing import IO
import zipfile
import os
import hashlib
import contextlib
import urllib.parse

import requests
import bs4
import pandas as pd

import biomart

from networkcommons import _conf
from networkcommons._session import _log
from .._builtin import _module_data


def _datasets() -> dict[str, dict]:

    return _module_data('datasets').get('omics', {})


def datasets() -> dict[str, str]:
    """
    Built-in omics datasets.

    Returns:
        A dict with dataset labels as keys and descriptions as values.
    """

    return {
        k: v['description']
        for k, v in _datasets().get('datasets', {}).items()
    }


def _baseurl() -> str:

    return _datasets()['baseurl']


def _commons_url(dataset: str, **kwargs) -> str:

    dsets = _datasets()
    baseurl = dsets['baseurl']
    path = dsets['datasets'][dataset]['path'].format(**kwargs)

    return urllib.parse.urljoin(baseurl, path)


def _dataset(key: str) -> dict | None:

    return _datasets()['datasets'].get(key.lower(), None)


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

    _log(f'Downloading `{url}` to `{path}`.')

    ses = _requests_session()

    with ses.get(url, timeout = timeouts, stream = True) as req:

        req.raise_for_status()

        with open(path, 'wb') as f:

            for chunk in req.iter_content(chunk_size = 8192):

                f.write(chunk)

    _log(f'Finished downloading `{url}` to `{path}`.')


def _maybe_download(url: str, **kwargs) -> str:

    url = url.format(**kwargs)
    cachedir = _conf.get('cachedir')
    md5 = hashlib.md5(url.encode()).hexdigest()
    fname = os.path.basename(urllib.parse.urlparse(url).path)
    path = os.path.join(cachedir, f'{md5}-{fname}')
    _log(f'Looking up in cache: `{url}` -> `{path}`.')

    if not os.path.exists(path):

        _log(f'Not found in cache, initiating download: `{url}`.')
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
        'tsv': pd.read_csv,
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


def get_ensembl_mappings():
    """
    Retrieves the mapping between Ensembl attributes for human genes.

    Returns:
        pandas.DataFrame: A DataFrame containing the mapping between Ensembl attributes.
            The DataFrame has the following columns:
            - ensembl_transcript_id: Ensembl transcript ID
            - gene_symbol: HGNC symbol for the gene
            - ensembl_gene_id: Ensembl gene ID
            - ensembl_peptide_id: Ensembl peptide ID
    """

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
    melted_df = melted_df[melted_df['gene_symbol'] != ''].drop(columns=['ensembl_type'])
    
    # Set 'ensembl_id' as the index
    melted_df.drop_duplicates(inplace=True)
    
    return melted_df