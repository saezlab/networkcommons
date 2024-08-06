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
Prior knowledge network used by LIANA+.
"""

__all__ = ['get_lianaplus']

# import lazy_import

# liana = lazy_import.lazy_module('liana')

import liana


def get_lianaplus(resource='Consensus'):
    """
    Retrieves the Liana+ ligand-receptor interaction network with
    directed interactions and specific criteria.

    Args:
        resource (str, optional): The resource to retrieve the network from.
            Defaults to 'Consensus'.

    Returns:
        pandas.DataFrame: Liana+ network with source, target, and sign columns.
    """

    network = liana.resource.select_resource(resource).drop_duplicates()
    network.columns = ['source', 'target']
    network['sign'] = 1

    return network
