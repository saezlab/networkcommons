#!/usr/bin/env python

#
# This file is part of the `networkcommons` Python module
#
# Copyright 2024
# Heidelberg University Hospital
#
# File author(s): Perfetto Lab (livia.perfetto@uniroma1.com)
#
# Distributed under the GPLv3 license
# See the file `LICENSE` or read a copy at
# https://www.gnu.org/licenses/gpl-3.0.txt
#

"""
SignalingProfiler: a multi-step pipeline integrating topological and causal
inference methods to derive context-specific signaling networks
"""

from __future__ import annotations

__all__ = [
    'run_signalingprofiler',
]

from typing import Literal
from collections.abc import Collection
from collections import ChainMap
import functools as ft

import pandas as pd
import networkx as nx
from pypath_common import _misc

from networkcommons._session import _log
from networkcommons.methods import _causal, _graph
from networkcommons.data.omics import _common as _downloader


def _mf_classifier(
        proteins: dict[str, float],
        with_exp: bool = False,
        only_proteins: Collection | None = None,
    ):
    """
    Classify proteins in four broad molecular functions (MF) according to Gene
    Ontology: kinases (kin), phosphatases (phos), transcription factors (tf),
    and all other types (other).

    Args:
        proteins: A dictionary of protein names and exp values to classify.
        with_exp: If True, return a dictionary with exp value -1 or 1. If
            False, return a dictionary with sets.
        only_proteins: A list of protein
        names to use to subset proteins

    Returns:
        A dictionary where keys are MF categories and values are sets or
        dictionaries with experimental values.
    """

    if only_proteins:

        proteins = {k: v for k, v in proteins.items() if k in only_proteins}

    # Read the GO molecular function (MF) data
    GO_mf_df = _downloader._open(
        (
            'https://filedn.eu/ld7S7VEWtgOf5uN0V7fbp84/'
            'gomf_annotation_networkcommons.tsv'
        ),
        ftype = 'tsv',
    )

    mf_dict = GO_mf_df.groupby('mf')['gene_name'].apply(set).to_dict()

    proteins_dict = {
        mf: {
            gene: proteins[gene] if with_exp else ''
            for gene in genes
            if gene in proteins
        }
        for mf, genes in mf_dict.items()
    }

    # Identify unclassified proteins
    classified_proteins = {
        gene
        for genes in proteins_dict.values()
        for gene in set(genes)
    }
    unclassified_proteins = set(proteins) - classified_proteins

    if unclassified_proteins:

        proteins_dict['other'] = (
            {
                protein: proteins[protein]
                for protein in unclassified_proteins
            }
            if with_exp else {}
        )

    return proteins_dict


def _validate_inputs(
        sources: dict,
        measurements: dict,
        graph: nx.Graph,
        layers: int,
        max_length: int | list[int],
    ) -> None:

    err = []

    if not isinstance(graph, nx.Graph):
        err.append(
            "The 'graph' parameter must be an instance of networkx.Graph."
        )

    if not isinstance(sources, dict) or not sources:
        err.append(
            "The 'sources' parameter must be a non-empty list or dictionary."
        )

    if not isinstance(measurements, dict) or not measurements:
        err.append(
            "The 'measurements' parameter must be "
            "a non-empty list or dictionary."
        )

    if not (isinstance(layers, int) and 0 < layers < 4):
        err.append("The 'layers' parameter must be 1, 2, or 3.")

    if layers == 1 and not isinstance(max_length, int):
        err.append("For 1 layers, 'max_length' must be an integer.")

    if (
        layers in [2, 3] and (
            not isinstance(max_length, list) or
            len(max_length) != 2
        )
    ):
        err.append(
            "For 2 or 3 layers, 'max_length' "
            "must be a list of two integers."
        )

    if (
        isinstance(max_length, list) and
        any(
            not isinstance(x, int) or
            x <= 0 for x in max_length
        )
    ):
        err.append("'max_length' values must be positive integers.")

    if isinstance(max_length, int) and max_length <= 0:
        err.append("'max_length' must be a positive integer.")

    if err:

        msg = 'Problem(s) with SignalingProfiler inputs: '

        for e in err:

            _log(f'{msg}{e}')

        raise ValueError(f'{msg}{"; ".join(err)}')


def _generate_naive_network(
        sources: dict,
        measurements: dict,
        graph: nx.Graph,
        layers: int,
        max_length: int | list[int],
    ) -> nx.Graph:

    """
    Generates a hierarchical (multi)layersed network from source nodes defining
    layers by distinguishing measured nodes by molecular function.

    Args:
        sources: A dictionary containing the sources and sign of
            perturbation.
        measurements: A dictionary containing the targets and sign of
            measurements.
        graph: The network.
        layers: specifies the number of layers to generate.
            Must be > 0 and < 4.
        max_length: The depth cutoff for finding paths.
            If `layers` is 1, this should be an int. For 2 or 3,
            it should be a list of ints.

    Returns:
        The constructed multi-layersed network.
    """

    _log('SignalingProfiler naive network building via all paths algorithm...')

    _validate_inputs(sources, measurements, graph, layers, max_length)


    def _by_func(
            classes: dict,
            funcs: Collection[Literal['kin', 'phos', 'other', 'tf']] | None = None,
        ) -> dict:

        return (
            classes
                if funcs is None else
            ChainMap(*(classes.get(func, {}) for func in _misc.to_set(funcs)))
        )


    # Define targets with molecular function classification
    targets = _mf_classifier(measurements, with_exp=True)
    max_length = _misc.to_list(max_length)

    stages = (
        None,
        ('kin', 'phos'),
        ('kin', 'phos', 'other'),
        None if layers == 0 else 'tf',
    )
    stages = (stages[0],) + stages[-layers:]
    networks = []

    for i, (src_funcs, tgt_funcs) in enumerate(zip(stages[:-1], stages[1:])):

        _log(f'SignalingProfiler naive network: stage {i + 1}')

        networks.append(
            _graph.run_all_paths(
                graph,
                _by_func(sources, src_funcs),
                _by_func(targets, tgt_funcs),
                depth_cutoff = max_length[i],
            )
        )

        if i == layers - 1:

            break

        sources = _mf_classifier(
            measurements,
            with_exp = True,
            only_proteins = networks[-1].nodes(),
        )

    naive_network = ft.reduce(nx.compose, networks)

    _log('SignalingProfiler naive network building ready.')

    return naive_network


def run_signalingprofiler(
        sources: dict,
        measurements: dict,
        graph: nx.Graph,
        layers: int,
        max_length: int | list[int],
        betaWeight: float = 0.2,
        solver: str | None = None,
        verbose: bool = False,
    ) -> nx.Graph:
    """
    Contextualize networks by the SignalingProfiler algorithm.

    Generates a hierarchical (multi)layersed network from source nodes defining
    layers by distinguishing measured nodes by molecular function and run the
    Vanilla Carnival algorithm via CORNETO to retrieve sign-coherent edges with
    nodes measured activity.

    Args:
        sources: A dictionary containing the sources and sign of perturbation.
        measurements: A dictionary containing the targets and sign of
            measurements.
        graph: The network.
        layers: specifies the number of layers to generate.
            Must be > 0 and < 4.
        max_length: The depth cutoff for finding paths. If `layers` is 1,
            this should be an int. For 2 or 3, it should be a list of
            ints.

    Returns:
        The constructed multi-layersed network.
    """

    # Generate naive_network
    naive_network = _generate_naive_network(
        sources = sources,
        measurements = measurements,
        graph = graph,
        layers = layers,
        max_length = max_length
    )

    # Optimize network using CORNETO
    opt_net = _causal.run_corneto_carnival(
        naive_network,
        sources,
        measurements,
        betaWeight = betaWeight,
        solver = solver,
        verbose = verbose
    )

    return opt_net
