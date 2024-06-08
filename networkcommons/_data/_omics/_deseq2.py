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

import pydeseq2


def run_deseq2_analysis(counts,
                        metadata,
                        test_group,
                        ref_group,
                        covariates=[]):
    """
    Runs DESeq2 analysis on the given counts and metadata.

    Args:
        counts (DataFrame): The counts data with gene symbols as index.
        metadata (DataFrame): The metadata with sample IDs as index.
        test_group (str): The name of the test group.
        ref_group (str): The name of the reference group.
        covariates (list, optional): List of covariates to include in the
            analysis. Defaults to an empty list.

    Returns:
        DataFrame: The results of the DESeq2 analysis as a DataFrame.
    """
    counts.set_index('gene_symbol', inplace=True)
    metadata.set_index('sample_ID', inplace=True)

    design_factors = ['group']

    if len(covariates) > 0:
        if isinstance(covariates, str):
            covariates = [covariates]
        design_factors += covariates

    inference = pydeseq2.default_inference.DefaultInference(n_cpus = 8)
    dds = pydeseq2.dds.DeseqDataSet(
        counts=counts.T,
        metadata=metadata,
        design_factors=design_factors,
        refit_cooks=True,
        inference=inference
    )
    dds.deseq2()

    results = pydeseq2.ds.DeseqStats(
        dds,
        contrast=["group", test_group, ref_group],
        inference=inference
    )
    results.summary()
    return results.results_df.astype('float64')
