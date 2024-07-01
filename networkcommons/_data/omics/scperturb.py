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

from typing import Any
import json

import bs4

from . import common

_URL = 'https://zenodo.org/record/10044268'


def scperturb_datasets() -> dict[str, Any]:
    """
    List the datasets available in scPerturb.

    Each dataset is an h5ad (HDF5 AnnData) file, stored in Zenodo:
    https://zenodo.org/records/10044268.
    """

    meta = scperturb_metadata()

    return {
        k: v['links']['self']
        for k, v in meta['files']['entries'].items()
    }


def scperturb_metadata() -> dict[str, Any]:
    """
    Metadata for the scPerturb deposited datasets.

    Retrieves the metadata as provided by the Zenodo API. The scPerturb Zenodo
    record is https://zenodo.org/records/10044268.
    """

    soup = common._open(_URL, ftype = 'html')
    data = soup.find(id = 'recordCitation').attrs['data-record']

    return json.loads(data)
