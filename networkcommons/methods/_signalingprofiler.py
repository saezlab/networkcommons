#!/usr/bin/env python

#
# This file is part of the `networkcommons` Python module
#
# Copyright 2024
# Heidelberg University Hospital
#
# File author(s): Perfetto Lab (livia.perfetto@gmail.com)
#
# Distributed under the GPLv3 license
# See the file `LICENSE` or read a copy at
# https://www.gnu.org/licenses/gpl-3.0.txt
#

"""
SignalingProfiler: a multi-step pipeline integrating topological and causal inference methods to derive context-specific signaling networks
"""

__all__ = [
    'run_phosphoscore',
    'run_signalingprofiler',
]

import networkcommons as nc
import pandas as pd
import networkx as nx
from typing import Union, List
from networkcommons._session import _log


def mf_classifier(proteins, with_exp=False, comp_list=None):
    """
    Classify proteins in four broad molecular functions (MF) according to Gene Ontology: kinases (kin), phosphatases (phos), transcription factors (tf), and all other types (other).

    Args:
    - proteins (dict): A dictionary of protein names and exp values to classify.
    - with_exp (bool): If True, return a dictionary with exp value -1 or 1. If False, return a dictionary with sets.
    - comp_list (list): A list of protein names to use to subset proteins

    Returns:
    - dict: A dictionary where keys are MF categories and values are sets or dictionaries with experimental values.
    """

    if comp_list:
        proteins = {k: v for k, v in proteins.items() if k in comp_list}

    # Read the GO molecular function (MF) data
    GO_mf_df = pd.read_csv('https://filedn.eu/ld7S7VEWtgOf5uN0V7fbp84/gomf_annotation_networkcommons.tsv', sep='\t')
    mf_dict = GO_mf_df.groupby('mf')['gene_name'].apply(set).to_dict()

    proteins_dict = {
        mf: {gene: proteins[gene] if with_exp else '' for gene in genes if gene in proteins}
        for mf, genes in mf_dict.items()
    }

    # Identify unclassified proteins
    classified_proteins = {gene for genes in proteins_dict.values() for gene in (genes if isinstance(genes, set) else genes.keys())}
    unclassified_proteins = set(proteins) - classified_proteins

    if unclassified_proteins:
        proteins_dict['other'] = (
            {protein: proteins[protein] for protein in unclassified_proteins} if with_exp else ''
        )

    return proteins_dict

def validate_inputs(sources, measurements, graph, layers, max_length):
    if not isinstance(graph, nx.Graph):
        raise TypeError("The 'graph' parameter must be an instance of networkx.Graph.")
    
    if not isinstance(sources, dict) or not sources:
        raise ValueError("The 'sources' parameter must be a non-empty list or dictionary.")
    
    if not isinstance(measurements, dict) or not measurements:
        raise ValueError("The 'measurements' parameter must be a non-empty list or dictionary.")
    
    if layers not in ['one', 'two', 'three']:
        raise ValueError("The 'layers' parameter must be 'one', 'two', or 'three'.")
    
    if layers == 'one' and not isinstance(max_length, int):
        raise TypeError("For 'one' layer, 'max_length' must be an integer.")
    
    if layers in ['two', 'three'] and (not isinstance(max_length, list) or len(max_length) != 2):
        raise ValueError("For 'two' or 'three' layers, 'max_length' must be a list of two integers.")
    
    if isinstance(max_length, list) and any(not isinstance(x, int) or x <= 0 for x in max_length):
        raise ValueError("'max_length' values must be positive integers.")
    
    if isinstance(max_length, int) and max_length <= 0:
        raise ValueError("'max_length' must be a positive integer.")


def generate_naive_network(source_dict, measurements, graph, layers, max_length: Union[int, List[int]]):

    """
    Generates a hierarchical (multi)layered network from source nodes defining layers by distinguishing measured nodes by molecular function. 

    Args:
    - source_dict (dict): A dictionary containing the sources and sign of perturbation.
    - measurements (dict): A dictionary containing the targets and sign of measurements.
    - graph (nx.Graph): The network.
    - layers (str): specifies the number of layers to generate ('one', 'two', or 'three').
    - max_length (int or list of int): The depth cutoff for finding paths. If `layers` is 'one', this should be an int. For 'two' or 'three', it should be a list of ints.

    Returns:
    - naive_network (nx.Graph): The constructed multi-layered network.
    """

    _log('SignalingProfiler naive network building via all paths algorithm...')

    validate_inputs(source_dict, measurements, graph, layers, max_length)

    # Define targets with molecular function classification
    targets = mf_classifier(measurements, with_exp=True)

    if layers == 'one':
        # Generate one-layered network
        naive_network, _ = nc.methods.run_all_paths(graph, source_dict, targets, depth_cutoff=max_length)


    elif layers == 'two':
        # Generate the first layer
        all_paths_network1, _ = nc.methods.run_all_paths(
            graph, 
            source_dict, 
            {**targets.get('kin'), **targets.get('phos'), **targets.get('other')}, 
            depth_cutoff=max_length[0]
        )

        # Prepare new sources for the second layer
        gene_nodes = list(all_paths_network1.nodes())
        sources_new = mf_classifier(measurements, with_exp=True, comp_list=gene_nodes)

        # Generate the second layer
        all_paths_network2, _ = nc.methods.run_all_paths(
            graph, 
            {**sources_new.get('kin'), **sources_new.get('phos'), **sources_new.get('other')}, 
            targets.get('tf'),
            depth_cutoff=max_length[1]
        )

        # Combine the first and second layer
        naive_network = nx.compose(all_paths_network1, all_paths_network2)
       
    elif layers == 'three':
        # Generate the first layer
        all_paths_network1, _ = nc.methods.run_all_paths(
            graph, 
            source_dict, 
            {**targets.get('kin'), **targets.get('phos')}, 
            depth_cutoff=max_length[0]
        )

        # Prepare new sources for the second layer
        gene_nodes = list(all_paths_network1.nodes())
        sources_new = mf_classifier(measurements, with_exp=True,comp_list=gene_nodes)

        # Generate the second layer
        all_paths_network2, _ = nc.methods.run_all_paths(
            graph,
            {**sources_new.get('kin'), **sources_new.get('phos')}, 
            {**targets.get('kin'), **targets.get('phos'), **targets.get('other')},
            depth_cutoff=1
        )
        
        # Combine the first and second layers
        combined_network = nx.compose(all_paths_network1, all_paths_network2)

        # Prepare new sources for the thirs layer
        gene_nodes2 = list(combined_network.nodes())
        sources_new2 = mf_classifier(measurements,  with_exp=True, comp_list=gene_nodes2)

        ### Third layer
        all_paths_network3, _ = nc.methods.run_all_paths(
            graph,
            {**sources_new2.get('kin'), **sources_new2.get('phos'), **sources_new2.get('other')},
            targets.get('tf'),
            depth_cutoff=max_length[1]
        )

        naive_network = nx.compose(combined_network, all_paths_network3)
    
    else:
        raise ValueError("Invalid value for 'layers'. Choose 'one', 'two', or 'three'.")
   
    return(naive_network)

def run_signalingprofiler(source_dict, measurements, graph, layers, max_length, betaWeight=0.2, solver=None, verbose=False):
    """
    Generates a hierarchical (multi)layered network from source nodes defining layers by distinguishing measured nodes by molecular function and run the Vanilla Carnival algorithm via CORNETO to retrieve sign-coherent edges with nodes measured activity.
    Args:
    - source_dict (dict): A dictionary containing the sources and sign of perturbation.
    - measurements (dict): A dictionary containing the targets and sign of measurements.
    - graph (nx.Graph): The network.
    - layers (str): specifies the number of layers to generate ('one', 'two', or 'three').
    - max_length (int or list of int): The depth cutoff for finding paths. If `layers` is 'one', this should be an int. For 'two' or 'three', it should be a list of ints.

    Returns:
    - naive_network (nx.Graph): The constructed multi-layered network.
    """
    # Generate naive_network
    naive_network = generate_naive_network(
        source_dict=source_dict, 
        measurements=measurements, 
        graph=graph, 
        layers=layers, 
        max_length=max_length
    )

    # Optimize network using CORNETO
    opt_net = nc.methods.run_corneto_carnival(
        naive_network, 
        source_dict, 
        measurements, 
        betaWeight=betaWeight, 
        solver=solver,
        verbose=verbose
    )
    
    return(opt_net)