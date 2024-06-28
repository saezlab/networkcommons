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

from typing import TYPE_CHECKING
import multiprocessing
import importlib

if TYPE_CHECKING:

    import pandas as pd

from pypath_common import _misc as _ppcommon
import pydeseq2 as _deseq2

for _mod in ('default_inference', 'dds', 'ds'):
    importlib.import_module(f'pydeseq2.{_mod}')

from networkcommons import _conf
from networkcommons._session import _log

__all__ = ['deseq2']


def deseq2(
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        test_group: str,
        ref_group: str,
        covariates: list | None = None,
    ) -> pd.DataFrame:
    """
    Runs DESeq2 analysis on the given counts and metadata.

    Args:
        counts (pd.DataFrame): The counts data with gene symbols as index.
        metadata (pd.DataFrame): The metadata with sample IDs as index.
        test_group (str): The name of the test group.
        ref_group (str): The name of the reference group.
        covariates (list, optional): List of covariates to include in the analysis.
            Defaults to None.

    Returns:
        pd.DataFrame: The results of the DESeq2 analysis as a data frame.
    """

    _log('Running differential expression analysis using DESeq2.')
    # TODO: hardcoding these column names in a hidden way not the best
    # solution:
    counts.set_index('gene_symbol', inplace=True)
    metadata.set_index('sample_ID', inplace=True)
    design_factors = ['group'] + _ppcommon.to_list(covariates)

    # TODO: this seems really arbitrary and specific to certain tables,
    # is this suitable and useful in a general DESeq2 analysis?
    test_group = test_group.replace('_', '-')
    ref_group = ref_group.replace('_', '-')

    n_cpus = _conf.get('cpu_count', multiprocessing.cpu_count())
    inference = _deseq2.default_inference.DefaultInference(n_cpus = n_cpus)

    dds = _deseq2.dds.DeseqDataSet(
        counts=counts.T,
        metadata=metadata,
        design_factors=design_factors,
        refit_cooks=True,
        inference=inference
    )

    dds.deseq2()

    results = _deseq2.ds.DeseqStats(
        dds,
        contrast=['group', test_group, ref_group],
        inference=inference
    )
    results.summary()
    result = results.results_df.astype('float64')
    _log('Finished running DESeq2.')

    return result
