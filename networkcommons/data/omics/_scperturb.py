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
Single-cell RNA-Seq data from the 'scPerturb' resource.
"""

from __future__ import annotations

__all__ = ['scperturb', 'scperturb_metadata', 'scperturb_datasets']

from typing import Any
import json

import anndata as ad
import bs4

from . import _common
from networkcommons._session import _log

_URL = 'https://zenodo.org/records/10044268'
_METADATA_FNAME = 'scperturb-record-10044268.html'


def _scperturb_artifacts() -> dict[str, dict[str, str | None]]:

    meta = scperturb_metadata()

    return {
        name: {
            'url': entry['links']['content'],
            'known_hash': entry.get('checksum'),
        }
        for name, entry in meta['files']['entries'].items()
    }


def scperturb_datasets() -> dict[str, Any]:
    """
    List the datasets available in scPerturb.

    Each dataset is an h5ad (HDF5 AnnData) file, stored in Zenodo:
    https://zenodo.org/records/10044268.
    """

    return {
        name: artifact['url']
        for name, artifact in _scperturb_artifacts().items()
    }


def scperturb_metadata() -> dict[str, Any]:
    """
    Metadata for the scPerturb deposited datasets.

    Retrieves the metadata as provided by the Zenodo API. The scPerturb Zenodo
    record is https://zenodo.org/records/10044268.
    """

    path = _common._pooch_retrieve(
        _URL,
        fname = _METADATA_FNAME,
        subdir = 'scperturb',
    )

    with open(path, encoding = 'utf-8') as fp:

        soup = bs4.BeautifulSoup(fp.read(), 'html.parser')

    data = soup.find(id = 'recordCitation').attrs['data-record']

    return json.loads(data)


def scperturb(dataset: str) -> ad.AnnData:
    """
    Access an scPerturb dataset.

    Args:
        dataset:
            Name of the dataset (which is the same as the original file name).
            It should be a key in the dictionary returned by
            `scperturb_datasets()`.

    Downloads (or retrieves from cache) one h5ad (HDF5 AnnData) file from the
    scPerturb repository. For a complete list of available datasets, see
    `scperturb_datasets()`.
    """
    _log(f"DATA: Retrieving scPerturb dataset {dataset}...")

    artifact = _scperturb_artifacts()[dataset]
    path = _common._pooch_retrieve(
        artifact['url'],
        fname = dataset,
        known_hash = artifact['known_hash'],
        subdir = 'scperturb',
    )

    return ad.read_h5ad(path)
