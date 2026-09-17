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
Acquisition of registered dataset files: download, cache, and verify.

Artifacts are declared in a Pooch registry file that maps each cache-relative
filename to its checksum and download URL. `fetch()` is the only public entry
point.
"""

from __future__ import annotations

from pathlib import Path

import pooch

from networkcommons import _conf

__all__ = ['DataError', 'fetch']

_REGISTRY_PATH = Path(__file__).parent / '_registry.txt'


class DataError(RuntimeError):
    """
    Raised when an artifact cannot be retrieved or fails verification.
    """


def _cache_root() -> Path:
    """
    The resolved cache directory of the current session.
    """

    return Path(_conf.get('cachedir')).resolve()


def _cache() -> pooch.Pooch:
    """
    Build the Pooch cache from the current session configuration.

    Raises:
        DataError:
            If the registry file cannot be read or is malformed.
    """

    cache = pooch.create(path = _cache_root(), base_url = '')

    try:

        cache.load_registry(_REGISTRY_PATH)

    except (OSError, ValueError) as err:

        raise DataError(
            f'Could not read the artifact registry: {err}'
        ) from err

    return cache


def _target(filename: str) -> Path:
    """
    Resolve a cache filename, rejecting paths outside the cache root.
    """

    root = _cache_root()
    target = (root / filename).resolve()

    if not target.is_relative_to(root):

        raise DataError(f'Unsafe cache path {filename!r}.')

    return target


def fetch(name: str, refresh: bool = False) -> Path:
    """
    Return a local, checksum-verified path for a registered artifact.

    Args:
        name:
            Artifact name as declared in the registry, a cache-relative
            filename.
        refresh:
            Delete a cached copy first and download the artifact again.

    Returns:
        The path of the cached artifact.

    Raises:
        DataError:
            If the artifact is unknown, its cache path is unsafe, it cannot
            be downloaded, or its checksum does not match the registry.
    """

    cache = _cache()

    if name not in cache.registry:

        raise DataError(f'Unknown artifact {name!r}.')

    target = _target(name)

    if refresh:

        target.unlink(missing_ok = True)

    try:

        return Path(cache.fetch(name))

    except Exception as err:

        raise DataError(f'Could not fetch {name!r}: {err}') from err
