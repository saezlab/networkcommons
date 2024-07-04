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

__all__ = []

import os
import shutil
import glob
import urllib.parse

from . import _common
from networkcommons._session import _log


def _get_decryptm(path: str):
    """
    Download DecryptM from EBI.

    Args:
        path:
            Download the data into this directory.
    """

    os.makedirs(path, exist_ok = True)
    url = 'https://ftp.pride.ebi.ac.uk/pride/data/archive/2023/03/PXD037285/'
    files = [f for f in _common._ls(url) if f.endswith('Curves.zip')]

    for fname in files:

        zip_url = urllib.parse.urljoin(url, fname)

        with _common._open(zip_url) as zip_file:

            _log(f'Extracting zip `{zip_file.filename}` to `{path}`.')
            zip_file.extractall(path)

    _ = [
        shutil.rmtree(pdfdir)
        for pdfdir in glob.glob(f'{path}/*/*/*/pdfs', recursive = True)
    ]
