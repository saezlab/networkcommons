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
import contextlib
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


def _commons_url(dataset: str, **kwargs) -> str:

    dsets = _datasets()
    baseurl = dsets['baseurl']
    path = dsets['datasets'][dataset]['path'].format(**kwargs)

    return urllib.parse.urljoin(baseurl, path)


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
    ftype = (ftype or os.path.splitext(path)[1]).lower()[1:]

    if not ftype:

        raise RuntimeError(f'Cannot determine file type for {url}.')


    if df is not False and ftype in PANDAS_READERS:

        df = df if isinstance(df, dict) else {}
        return PANDAS_READERS[ftype](path, **df)

    elif ftype in {'tsv', 'csv', 'txt'}:

        return contextlib.closing(open(path, 'r'))

    elif ftype == 'zip':

        return contextlib.closing(zipfile.ZipFile(path, 'r'))

    else:

        raise NotImplementedError(f'Can not open file type `{ftype}`.')


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
