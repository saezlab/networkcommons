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

from . import _common
from networkcommons._session import _log

_URL = 'https://zenodo.org/record/10044268'


def scperturb_datasets() -> dict[str, Any]:
    """
    List the datasets available in scPerturb.

    Each dataset is an h5ad (HDF5 AnnData) file, stored in Zenodo:
    https://zenodo.org/records/10044268.
    """

    meta = scperturb_metadata()

    return {
        k: v['links']['content']
        for k, v in meta['files']['entries'].items()
    }


def scperturb_metadata() -> dict[str, Any]:
    """
    Metadata for the scPerturb deposited datasets.

    Retrieves the metadata as provided by the Zenodo API. The scPerturb Zenodo
    record is https://zenodo.org/records/10044268.
    """

    soup = _common._open(_URL, ftype = 'html')
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

    urls = scperturb_datasets()
    path = _common._maybe_download(urls[dataset])

    return ad.read_h5ad(path)
