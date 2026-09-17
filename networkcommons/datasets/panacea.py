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
PANACEA — original raw data access.

Douglass et al., A community challenge for a pancancer drug mechanism of
action inference from perturbational profile data. Cell Reports Medicine
(2022). DOI: 10.1016/j.xcrm.2021.100492. Challenge data DOI:
10.7303/syn20968331. Raw data: NCBI GEO accession GSE186341.

The data is fetched from the original GEO deposition, not from the
NetworkCommons server. The count files list samples by treatment group
(`drug_dose_time`), while the series matrix identifies each sample
individually. `counts()` and `metadata()` can be joined on the `group` and
canonical cell line columns; `sample_id` is the unique per-sample key of the
GEO annotation.
"""

from __future__ import annotations

import csv
import gzip
import re
from typing import TYPE_CHECKING

from ._data import fetch
from ._io import read_table, to_frame

if TYPE_CHECKING:

    import pandas as pd

__all__ = [
    'cell_line_label',
    'cell_lines',
    'counts',
    'metadata',
    'samples',
]

_SERIES_MATRIX = 'panacea/matrix/GSE186341_series_matrix.txt.gz'

_ZERO_DOSE = re.compile(r'_0_')

_COUNTS = {
    'ASPC': 'panacea/counts/GSE186341_ASPC_dream_counts.csv.gz',
    'DU145': 'panacea/counts/GSE186341_DU145_dream_counts.csv.gz',
    'EFO21': 'panacea/counts/GSE186341_EFO21_dream_counts.csv.gz',
    'H1793': 'panacea/counts/GSE186341_H1793_dream_counts.csv.gz',
    'HCC1143': 'panacea/counts/GSE186341_HCC1143_dream_counts.csv.gz',
    'HF2597': 'panacea/counts/GSE186341_HF2597_dream_counts.csv.gz',
    'HSTS': 'panacea/counts/GSE186341_HSTS_dream_counts.csv.gz',
    'KRJ1': 'panacea/counts/GSE186341_KRJ1_dream_counts.csv.gz',
    'LNCAP': 'panacea/counts/GSE186341_LNCAP_dream_counts.csv.gz',
    'PANC1': 'panacea/counts/GSE186341_PANC1_dream_counts.csv.gz',
    'U87': 'panacea/counts/GSE186341_U87_dream_counts.csv.gz',
}

_LABELS = {
    'ASPC': 'AsPC-1',
    'DU145': 'DU145',
    'EFO21': 'EFO-21',
    'H1793': 'NCI-H1793',
    'HCC1143': 'HCC1143',
    'HF2597': 'HF2597',
    'HSTS': 'HSTS',
    'KRJ1': 'KRJ1',
    'LNCAP': 'LNCaP',
    'PANC1': 'PANC-1',
    'U87': 'U87 MG',
}


def cell_lines() -> tuple:
    """
    Return the canonical cell line identifiers.

    Returns:
        The cell line identifiers, as used by the other functions.
    """

    return tuple(_COUNTS)


def cell_line_label(cell_line: str) -> str:
    """
    Return the GEO cell line name for a canonical identifier.

    Args:
        cell_line:
            A cell line identifier from `cell_lines()`.

    Returns:
        The cell line name as used in the GEO sample annotation.

    Raises:
        ValueError:
            If the cell line is not part of the dataset.
    """

    if cell_line not in _LABELS:

        raise ValueError(
            f'Unknown PANACEA cell line {cell_line!r}; '
            f'expected one of {cell_lines()}'
        )

    return _LABELS[cell_line]


def counts(cell_line: str) -> pd.DataFrame:
    """
    Raw RNA-seq counts for one PANACEA cell line.

    Genes are indexed by the `gene_id` column (NCBI Gene identifiers). Sample
    columns are canonical treatment group labels that join to the `group`
    column of `metadata()`; vehicle controls are spelled `drug__time`
    regardless of the two spellings used in the source files.

    Args:
        cell_line:
            A cell line identifier from `cell_lines()`.

    Returns:
        Genes as rows and sample groups as columns, with raw counts.

    Raises:
        ValueError:
            If the cell line is not part of the dataset.
    """

    if cell_line not in _COUNTS:

        raise ValueError(
            f'Unknown PANACEA cell line {cell_line!r}; '
            f'expected one of {cell_lines()}'
        )

    table = read_table(fetch(_COUNTS[cell_line]), sep = ',')

    table = table.rename(columns = {table.columns[0]: 'gene_id'})
    table.columns = _canonical_columns(table.columns)

    return table


def metadata() -> pd.DataFrame:
    """
    Sample annotation from the original GEO series matrix.

    Returns:
        One row per sample with the unique `sample_id` and `title`, the
        treatment `group` that matches the count columns, the `cell_line`
        name, the GEO accession, and the remaining sample characteristics.
    """

    fields = _series_matrix_fields()

    titles = fields.get('!Sample_title', [[]])[0]
    characteristics = fields.get('!Sample_characteristics_ch1', [])

    table = {
        'sample_id': [title.split(' ')[0] for title in titles],
        'title': titles,
        'group': [_group_from_title(title) for title in titles],
    }

    for key in ('!Sample_geo_accession', '!Sample_organism_ch1'):

        if key in fields:

            table[key.replace('!Sample_', '')] = fields[key][0]

    for row in characteristics:

        pairs = [value.split(': ', 1) for value in row]
        table[_column_name(pairs[0][0])] = [pair[1].strip() for pair in pairs]

    return to_frame(table)


def samples(cell_line: str) -> pd.DataFrame:
    """
    Sample annotation for one cell line.

    Args:
        cell_line:
            A cell line identifier from `cell_lines()`.

    Returns:
        The `metadata()` rows of the cell line, with a reset index.

    Raises:
        ValueError:
            If the cell line is not part of the dataset.
    """

    label = cell_line_label(cell_line)
    table = metadata()

    return table.loc[table['cell_line'] == label].reset_index(drop = True)


def _column_name(key: str) -> str:
    """
    Normalize a GEO characteristic name to a column name.
    """

    return key.strip().lower().replace(' ', '_')


def _canonical_columns(columns) -> list:
    """
    Canonicalize group labels and keep duplicate columns unique.

    The source count files spell vehicle controls both as `drug__time` and
    as `drug_0_time`; both become `drug__time`. Repeated labels keep a
    numeric suffix, counted per group.
    """

    seen = {}
    result = []

    for column in columns:

        if column == 'gene_id':

            result.append(column)

            continue

        base = _ZERO_DOSE.sub('__', re.sub(r'\.\d+$', '', column))
        index = seen.get(base, 0)

        result.append(base if not index else f'{base}.{index}')
        seen[base] = index + 1

    return result


def _group_from_title(title: str) -> str:
    """
    Derive the count-file group label from a GEO sample title.

    Titles look like `aspc_A2_Icotinib_7.5_24 ASPCPS1AF02_CAGGCGTA`, with
    cell line, plate well, drug, dose, and time, followed by a barcode. The
    corresponding count column is `ICOTINIB_7.5_24`, with an empty dose for
    the vehicle and untreated controls.
    """

    parts = title.partition(' ')[0].split('_')

    if len(parts) != 5:

        return ''

    _, _, drug, dose, time = parts

    return f'{drug.upper()}_{"" if dose == "0" else dose}_{time}'


def _series_matrix_fields() -> dict:
    """
    Read the `!Sample_*` lines of the series matrix into a field mapping.
    """

    with gzip.open(fetch(_SERIES_MATRIX), 'rt') as handle:

        lines = [line for line in handle if line.startswith('!Sample_')]

    fields = {}

    for line in lines:

        key, _, values = line.rstrip('\n').partition('\t')
        fields.setdefault(key, []).append(
            next(csv.reader([values], delimiter = '\t'))
        )

    return fields
