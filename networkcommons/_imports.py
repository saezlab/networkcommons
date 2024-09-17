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
Handling of imports.
"""

import contextlib
import warnings


@contextlib.contextmanager
def optional(error: str = 'ignore'):
    """
    Import optional dependencies in this context.

    Examples:
        >>> with optional('raise'):
        ...     import networkx as nx

    Args:
        error:
            Raise exception, emit warning or ignore if the import failed.

    From https://stackoverflow.com/a/73838937/854988
    """

    assert error in {'raise', 'warn', 'ignore'}

    try:

        yield None

    except ImportError as e:

        if error == 'raise':

            raise e

        if error == 'warn':

            warnings.warn(
                f'Missing optional dependency "{e.name}". '
                'Use pip or conda to install.'
            )
