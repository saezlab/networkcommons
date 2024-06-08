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

from __future__ import annotations

from typing import Any, IO
import zipfile
import os
import re
import glob
import hashlib
import urllib.parse

import shutil
import requests
import bs4
import owncloud as oc
import pandas as pd

from .._builtin import _module_data
from networkcommons import _conf
from networkcommons._session import _log

__all__ = ['datasets']


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


def _dataset(key: str) -> dict | None:

    return _datasets()['datasets'].get(key.lower(), None)


def _download(url: str, path: str) -> None:

    timeouts = tuple(_conf.get(f'http_{k}_timout') for k in ('read', 'connect'))

    _log(f'Downloading `{url}` to `{path}`.')

    with requests.get(url, timeout = timeouts, stream = True) as r:

        r.raise_for_status()

        with open(path, 'wb') as f:

            for chunk in r.iter_content(chunk_size = 8192):

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
    ftype = (ftype or os.path.splitext(path)[1]).lower()

    if not ftype:

        raise RuntimeError(f'Cannot determine file type for {url}.')


    if df is not False and ftype in PANDAS_READERS:

        df = df if isinstance(df, dict) else {}
        return PANDAS_READERS[ftype](path, **df)

    elif ftype in {'tsv', 'csv', 'txt'}:

        with open(path, 'r') as fp:

            yield fp

    elif ftype == 'zip':

        with zipfile.ZipFile(path, 'r') as fp:

            yield fp

    else:

        raise NotImplementedError(f'Can not open file type `{ftype}`.')


def download_dataset(dataset, **kwargs):
    """
    Downloads a dataset and returns a list of pandas DataFrames.

    Args:
        dataset (str): The name of the dataset to download.
        **kwargs: Additional keyword arguments to pass to the decryptm call:
            - experiment (str): The name of the experiment.
            - data_type (str): The type of data to download (Phosphoproteome,
            Full proteome, Acetylome...). Not all data types are available for
            all experiments.

    Returns:
        list: A list of pandas DataFrames, each representing a file in
            the downloaded dataset.

    Raises:
        ValueError: If the specified dataset is not available.

    """
    available_datasets = get_available_datasets()

    if dataset not in available_datasets:
        error_message = f"Dataset {dataset} not available. Check available datasets with get_available_datasets()"  # noqa: E501
        raise ValueError(error_message)
    elif dataset == 'decryptm':
        file_list = decryptm_handler(**kwargs)
    else:
        save_path = f'./data/{dataset}.zip'
        if not os.path.exists(save_path):
            download_url(f'https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B/download?path=%2F&files={dataset}', save_path)  # noqa: E501

            # download?path=%2Fdecryptm%2F3_EGFR_Inhibitors%2FFullproteome&files=curves.txt

        # unzip files
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall('./data/')

        # list contents of dir, read them and append to list
        files = os.listdir(f'./data/{dataset}')
        file_list = []
        for file in files:
            df = pd.read_csv(f'./data/{dataset}/{file}', sep='\t')
            file_list.append(df)

    return file_list


def download_url(url, save_path, chunk_size=128):
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The path where the downloaded file will be saved.
        chunk_size (int, optional): The size of each chunk to download.
            Defaults to 128.
    """
    r = requests.get(url, stream=True)
    # mkdir if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def list_directories(path):
    public_link = (
        "https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B"
        "?path=%2Fdecryptm"
    )
    password = "networkcommons_datasaezlab"
    occontents = oc.Client.from_public_link(public_link,
                                            folder_password=password)
    response = occontents.list(path)
    file_paths = [item.path.strip('/') for item in response]
    return response, file_paths


def _ls(path: str) -> list[str]:
    """
    List files in a remote directory.

    Args:
        path:
            HTTP URL of a directory with standard nginx directory listing.
    """

    resp = requests.get(path)

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
