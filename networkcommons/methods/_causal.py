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
Causal network inference methods.
"""

from __future__ import annotations

__all__ = [
    'run_corneto_carnival',
]

import lazy_import
import networkx as nx
# cn = lazy_import.lazy_module('corneto')
# cn_nx = lazy_import.lazy_module('corneto.contrib.networkx')
import corneto as cn
import corneto.contrib.networkx as cn_nx
import sys
import io

from networkcommons._session import _log

from .. import utils


def run_corneto_carnival(network,
                         source_dict,
                         target_dict,
                         betaWeight=0.2,
                         solver=None,
                         verbose=False):
    """
    Run the Vanilla Carnival algorithm via CORNETO.

    Args:
        network (nx.Graph): The network.
        source_dict (dict): A dictionary containing the sources and sign
            of perturbation.
        target_dict (dict): A dictionary containing the targets and sign
            of measurements.

    Returns:
        nx.Graph: The subnetwork containing the paths found by CARNIVAL.
    """  
    _log('Running Vanilla Carnival algorithm via CORNETO...')
    
    # Capture stdout and stderr for the entire function
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout = sys.stdout  # Save original stdout
    old_stderr = sys.stderr  # Save original stderr
    sys.stdout = stdout_capture  # Redirect stdout to capture
    sys.stderr = stderr_capture  # Redirect stderr to capture

    try:
        # Convert network to CORNETO format
        corneto_net = utils.to_cornetograph(network)

        # Run Vanilla Carnival method
        problem, graph = cn.methods.runVanillaCarnival(
            perturbations=source_dict,
            measurements=target_dict,
            priorKnowledgeNetwork=corneto_net,
            betaWeight=betaWeight,
            solver=solver,
            verbose=True  # This verbose controls internal print/logging within runVanillaCarnival
        )

        # Subnetwork and solution
        network_sol = graph.edge_subgraph(
            cn.methods.carnival.get_selected_edges(problem, graph),
        )

        network_nx = utils.to_networkx(network_sol, skip_unsupported_edges=True)
        network_nx.remove_nodes_from(['_s', '_pert_c0', '_meas_c0'])
    
    # when network is empty
    except TypeError:
        network_nx = nx.Graph()
        _log('WARNING: Network is empty. No solution found.')

    finally:
        # Restore original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Capture the output from stdout and stderr
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Handle stdout (typically the logger output and print statements)
        for line in stdout_output.splitlines():
            if verbose:
                print(line)  # Print each line if verbose
            else:
                _log(line)  # Log each line individually

        # Handle stderr (warnings and errors, often used by solvers)
        for line in stderr_output.splitlines():
            if verbose:
                print(line)  # Print each line if verbose
            else:
                _log(line)  # Log stderr output as errors

        # Final logging
        _log('CORNETO-Carnival finished.')
        _log(f'Network solution with {len(network_nx.nodes)} nodes and {len(network_nx.edges)} edges.')

    return network_nx