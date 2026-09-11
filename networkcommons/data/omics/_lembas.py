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
Signaling network, ligand concentration and TF activity data from the
LEMBAS (Learning Mechanistic Biological Activity from Signaling) resource.

Nilsson et al. 2022, Nature Communications
https://doi.org/10.1038/s41467-022-30684-y
"""

from __future__ import annotations

__all__ = [
    'lembas_datasets',
    'lembas_network',
    'lembas_ligands',
    'lembas_tfs',
    'lembas_annotation',
]

import pandas as pd

from . import _common
from networkcommons._session import _log

_ZENODO_BASE = 'https://zenodo.org/records/10815391/files'
_GITHUB_BASE = 'https://raw.githubusercontent.com/Lauffenburger-Lab/LEMBAS/main/Model/data'

_URLS: dict[str, dict[str, str]] = {
    'macrophage': {
        'network':    f'{_ZENODO_BASE}/macrophage_network.tsv',
        'ligands':    f'{_ZENODO_BASE}/macrophage_ligands.tsv',
        'tfs':        f'{_ZENODO_BASE}/macrophage_TFs.tsv',
        'annotation': f'{_GITHUB_BASE}/macrophage-Annotation.tsv',
    },
    'ligand_screen': {
        'network':    f'{_GITHUB_BASE}/ligandScreen-Model.tsv',
        'ligands':    f'{_GITHUB_BASE}/ligandScreen-Ligands.tsv',
        'tfs':        f'{_GITHUB_BASE}/ligandScreen-TFs.tsv',
        'annotation': f'{_GITHUB_BASE}/ligandScreen-Annotation.tsv',
    },
}

_DATASET_INFO: dict[str, dict[str, str]] = {
    'macrophage': {
        'name': 'Macrophage (low-coverage)',
        'description': (
            'Macrophage signaling dataset with ~10 extracellular ligands across '
            '~170 experimental conditions. TF activities derived from RNA-seq '
            'via DoRothEA/VIPER.'
        ),
    },
    'ligand_screen': {
        'name': 'Ligand screen (high-coverage)',
        'description': (
            'High-throughput ligand screen with ~60 ligands across ~500 '
            'experimental conditions. TF activities derived from RNA-seq '
            'via DoRothEA/VIPER.'
        ),
    },
}


def lembas_datasets() -> pd.DataFrame:
    """
    Available LEMBAS datasets.

    Returns:
        DataFrame with dataset keys, names and descriptions.
    """
    return pd.DataFrame.from_dict(_DATASET_INFO, orient='index').rename_axis('dataset')


def _lembas_table(dataset: str, table: str) -> pd.DataFrame:

    valid_datasets = list(_URLS.keys())
    if dataset not in valid_datasets:
        raise ValueError(
            f'Unknown dataset {dataset!r}. '
            f'Available: {valid_datasets}. '
            'See lembas_datasets() for details.'
        )

    url = _URLS[dataset][table]
    _log(f'LEMBAS: Fetching {table} for dataset {dataset!r} from {url}')

    result = _common._open(url, df={'sep': '\t'})
    assert isinstance(result, pd.DataFrame)
    return result


def lembas_network(dataset: str = 'macrophage') -> pd.DataFrame:
    """
    Prior knowledge signaling network for a LEMBAS dataset.

    The returned DataFrame has columns ``source``, ``target``,
    ``stimulation`` (1/0) and ``inhibition`` (1/0), among others.
    Pass it through ``LEMBAS.model.model_utilities.format_network``
    to add the ``mode_of_action`` column before building a
    ``SignalingModel``.

    Args:
        dataset:
            One of ``'macrophage'`` or ``'ligand_screen'``.
            See :func:`lembas_datasets` for details.

    Returns:
        DataFrame with one row per directed protein–protein interaction.
    """
    return _lembas_table(dataset, 'network')


def lembas_ligands(dataset: str = 'macrophage') -> pd.DataFrame:
    """
    Ligand concentration (input) matrix for a LEMBAS dataset.

    Args:
        dataset:
            One of ``'macrophage'`` or ``'ligand_screen'``.
            See :func:`lembas_datasets` for details.

    Returns:
        DataFrame of shape (conditions × ligands). Index contains
        condition names; columns are UniProt IDs of extracellular
        ligands; values are ligand concentrations or binary presence (0/1).
    """
    df = _lembas_table(dataset, 'ligands')
    df = df.set_index(df.columns[0])
    df.index.name = 'condition'
    return df


def lembas_tfs(dataset: str = 'macrophage') -> pd.DataFrame:
    """
    Transcription factor activity (output) matrix for a LEMBAS dataset.

    TF activities are inferred from RNA-seq data using DoRothEA/VIPER
    and normalised to [0, 1].

    Args:
        dataset:
            One of ``'macrophage'`` or ``'ligand_screen'``.
            See :func:`lembas_datasets` for details.

    Returns:
        DataFrame of shape (conditions × TFs). Index contains condition
        names; columns are UniProt IDs of transcription factors; values
        are TF activity scores in [0, 1].
    """
    df = _lembas_table(dataset, 'tfs')
    df = df.set_index(df.columns[0])
    df.index.name = 'condition'
    return df


def lembas_annotation(dataset: str = 'macrophage') -> pd.DataFrame:
    """
    Node annotation table for a LEMBAS dataset.

    Maps UniProt IDs to gene names and marks which nodes are ligands
    and which are transcription factors.

    Args:
        dataset:
            One of ``'macrophage'`` or ``'ligand_screen'``.
            See :func:`lembas_datasets` for details.

    Returns:
        DataFrame with columns including ``code`` (UniProt ID),
        ``name`` (gene symbol), ``ligand`` (bool) and ``TF`` (bool).
    """
    return _lembas_table(dataset, 'annotation')
