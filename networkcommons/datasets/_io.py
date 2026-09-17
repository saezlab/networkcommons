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
Readers for dataset file formats.

This is the only module that reads table files with pandas. Parameters are
library-neutral so the reader can be repointed at another frame library
without touching dataset modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Optional, Sequence, Union

import pandas as pd

__all__ = ['read_excel', 'read_h5ad', 'read_table', 'to_frame']

PathLike = Union[str, Path, IO]


def to_frame(data: dict) -> pd.DataFrame:
    """
    Build a data frame from a column mapping.

    Args:
        data:
            Mapping of column names to column values.

    Returns:
        The columns as a data frame.
    """

    return pd.DataFrame(data)


def read_table(
    path: PathLike,
    *,
    sep: str = '\t',
    columns: Optional[Sequence[str]] = None,
    schema: Optional[dict] = None,
    header: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Read a delimited text table.

    Args:
        path:
            Path of the file. Compression is inferred from the extension.
        sep:
            Field separator.
        columns:
            Read only these columns, when given.
        schema:
            Column name to dtype mapping, when given.
        header:
            Row number to use as the column names.

    Returns:
        The table as a data frame.
    """

    return pd.read_csv(
        path,
        sep = sep,
        usecols = columns,
        dtype = schema,
        header = header,
    )


def read_excel(
    path: PathLike,
    *,
    sheet: Union[int, str] = 0,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Read one sheet of an Excel workbook.

    Args:
        path:
            Path of the workbook.
        sheet:
            Sheet index or name.
        columns:
            Read only these columns, when given.

    Returns:
        The sheet as a data frame.
    """

    return pd.read_excel(path, sheet_name = sheet, usecols = columns)


def read_h5ad(path: PathLike):
    """
    Read an AnnData file.

    Args:
        path:
            Path of the H5AD file.

    Returns:
        The data as an AnnData object.
    """

    import anndata as ad

    return ad.read_h5ad(path)
