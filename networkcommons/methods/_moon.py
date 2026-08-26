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
Moon: multi-omics??
"""

from __future__ import annotations

__all__ = [
    'meta_network_cleanup',
    'prepare_metab_inputs',
    'is_expressed',
    'filter_pkn_expressed_genes',
    'filter_pkn_expressed_genes_fast',
    'filter_input_nodes_not_in_pkn',
    'keep_controllable_neighbours',
    'keep_observable_neighbours',
    'compress_same_children',
    'run_moon_core',
    'run_moon',
    'filter_incoherent_TF_target',
    'filter_incohrent_TF_target',
    'decompress_moon_result',
    'reduce_solution_network',
    'reduce_solution_network_double_thresh',
    'get_moon_scoring_network',
    'translate_column_HMDB',
    'translate_res',
]

import collections
import numbers
import re
from collections.abc import Mapping

import networkx as nx
import pandas as pd
import decoupler as dc
import numpy as np

from networkcommons._session import _log


_SIGN_COLUMNS = ('sign', 'interaction', 'mor', 'weight')


def _iter_edges_with_data(graph):
    """Yield graph edges while hiding the MultiDiGraph key convention."""
    if not isinstance(graph, nx.Graph):
        raise TypeError('meta_network must be a NetworkX graph.')
    if not graph.is_directed():
        raise TypeError('MOON requires a directed prior-knowledge network.')

    if graph.is_multigraph():
        for source, target, _, attributes in graph.edges(data=True, keys=True):
            yield source, target, attributes
    else:
        yield from graph.edges(data=True)


def _canonical_sign(attributes, context='edge'):
    """Read one signed interaction from supported R/Python field names."""
    values = []
    for column in _SIGN_COLUMNS:
        if column not in attributes or attributes[column] is None:
            continue
        try:
            value = float(attributes[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'{context} has a non-numeric {column!r} value: '
                f'{attributes[column]!r}.'
            ) from exc
        if np.isnan(value):
            continue
        if not np.isfinite(value):
            raise ValueError(f'{context} has a non-finite {column!r} value.')
        values.append((column, value))

    if not values:
        raise ValueError(
            f'{context} must define one of {", ".join(_SIGN_COLUMNS)}.'
        )

    sign = values[0][1]
    if any(not np.isclose(value, sign) for _, value in values[1:]):
        aliases = ', '.join(f'{column}={value}' for column, value in values)
        raise ValueError(f'{context} has conflicting sign aliases: {aliases}.')
    return sign


def _pkn_to_regulons(graph):
    """Return a decoupler-ready edge table with a canonical ``sign`` column."""
    records = []
    for source, target, attributes in _iter_edges_with_data(graph):
        records.append({
            'source': source,
            'target': target,
            'sign': _canonical_sign(attributes, f'edge {source!r}->{target!r}'),
        })
    return pd.DataFrame(records, columns=['source', 'target', 'sign'])


def _signed_digraph(graph):
    """Copy a PKN to a simple graph with a canonical ``sign`` attribute."""
    result = nx.DiGraph()
    result.add_nodes_from(graph.nodes(data=True))
    for source, target, attributes in _iter_edges_with_data(graph):
        copied_attributes = dict(attributes)
        copied_attributes['sign'] = _canonical_sign(
            attributes, f'edge {source!r}->{target!r}'
        )
        result.add_edge(source, target, **copied_attributes)
    return result


def _as_named_scores(values, name):
    """Normalize the named-vector convention shared by R and Python MOON."""
    if isinstance(values, pd.Series):
        return values.to_dict()
    if isinstance(values, Mapping):
        return dict(values)
    raise TypeError(f'{name} must be a mapping or pandas Series keyed by node.')


def _input_nodes(values, name):
    if isinstance(values, Mapping):
        return list(values)
    if isinstance(values, (str, bytes)):
        raise TypeError(f'{name} must be a collection of node names.')
    try:
        return list(values)
    except TypeError as exc:
        raise TypeError(
            f'{name} must be a mapping or collection of nodes.'
        ) from exc


def _validated_n_steps(n_steps):
    if n_steps is None:
        return None
    if isinstance(n_steps, bool) or not isinstance(n_steps, numbers.Integral):
        raise ValueError('n_steps must be a non-negative integer or None.')
    if n_steps < 0:
        raise ValueError('n_steps must be a non-negative integer or None.')
    return int(n_steps)


def _reachable_nodes(graph, seeds, n_steps):
    """Return all directed descendants at most ``n_steps`` from any seed."""
    seeds = list(seeds)
    missing = [seed for seed in seeds if seed not in graph]
    if missing:
        raise ValueError(f'Input nodes are not in the PKN: {missing}.')

    n_steps = _validated_n_steps(n_steps)
    reached = set()
    for seed in seeds:
        if n_steps is None:
            reached.add(seed)
            reached.update(nx.descendants(graph, seed))
        else:
            reached.update(nx.single_source_shortest_path_length(
                graph, seed, cutoff=n_steps
            ))
    return reached


def _edge_only_subgraph(graph, nodes=None):
    """Return an induced graph containing only nodes represented by an edge.

    R's edge-table helpers discard isolated vertices after each filtering
    operation. NetworkX's ``subgraph`` retains them, so this small adapter
    makes the graph-native implementation follow the same contract where it
    matters (notably ``get_moon_scoring_network``).
    """
    nodes = None if nodes is None else set(nodes)
    result = nx.DiGraph()
    for source, target, attributes in graph.edges(data=True):
        if nodes is None or (source in nodes and target in nodes):
            result.add_edge(source, target, **dict(attributes))
    return result


def _empty_att(columns):
    """Return a stable empty attribute table for graph-native result helpers."""
    return pd.DataFrame({
        column: pd.Series(dtype='object') for column in columns
    })


def _attach_rna(att, rna_input, node_column):
    result = att.copy()
    if rna_input is None:
        result['RNA_input'] = np.nan
    else:
        result['RNA_input'] = result[node_column].map(
            _as_named_scores(rna_input, 'rna_input')
        )
    return result


def _score_table(moon_res, graph=None, require_level=False):
    """Select scores in the identifier domain of ``graph`` when possible."""
    if not isinstance(moon_res, pd.DataFrame):
        raise TypeError('moon_res must be a pandas DataFrame.')
    if 'score' not in moon_res.columns:
        raise ValueError("moon_res must contain a 'score' column.")

    source_column = 'source' if 'source' in moon_res.columns else None
    if 'source_original' in moon_res.columns:
        if source_column is None:
            source_column = 'source_original'
        elif graph is not None:
            original_hits = moon_res['source_original'].isin(graph.nodes).sum()
            source_hits = moon_res['source'].isin(graph.nodes).sum()
            if original_hits > source_hits:
                source_column = 'source_original'
    if source_column is None:
        raise ValueError(
            "moon_res must contain 'source' or 'source_original'."
        )

    result = moon_res.copy()
    result['source'] = result[source_column]
    if 'level' not in result.columns:
        if require_level:
            raise ValueError("moon_res must contain a 'level' column.")
        result['level'] = 0
        _log(
            'MOON: moon_res has no level column; using level 0 for '
            'legacy compatibility.'
        )

    if result['source'].isna().any():
        raise ValueError('moon_res contains missing source nodes.')
    result['score'] = pd.to_numeric(result['score'], errors='raise')
    result['level'] = pd.to_numeric(result['level'], errors='raise')
    # decoupler's normalized weighted mean can legitimately return infinite
    # scores for a small regulon. cosmosR retains those values, and they work
    # naturally with the absolute-score and sign tests below. Missing scores,
    # by contrast, cannot be assigned a consistent edge direction.
    if result['score'].isna().any():
        raise ValueError('moon_res contains missing scores.')
    if not np.isfinite(result['level']).all():
        raise ValueError('moon_res contains non-finite levels.')
    if not np.all(np.equal(result['level'], np.floor(result['level']))):
        raise ValueError('moon_res levels must be integers.')
    result['level'] = result['level'].astype(int)
    return result.drop_duplicates(subset='source', keep='first')


def meta_network_cleanup(graph):
    """Clean a signed PKN using the current ``cosmosR`` edge semantics.

    Self loops are removed. Parallel source-target edges are averaged when a
    ``MultiDiGraph`` is supplied, and only pairs with a resulting sign of
    exactly ``1`` or ``-1`` are kept. A simple ``DiGraph`` has already
    collapsed duplicate edges before this function receives it.
    """
    grouped_edges = collections.defaultdict(list)
    for source, target, attributes in _iter_edges_with_data(graph):
        if source == target:
            continue
        grouped_edges[(source, target)].append((
            _canonical_sign(attributes, f'edge {source!r}->{target!r}'),
            dict(attributes),
        ))

    cleaned = nx.DiGraph()
    for (source, target), values in grouped_edges.items():
        sign = float(np.mean([value[0] for value in values]))
        if sign not in (1.0, -1.0):
            continue
        attributes = {
            key: value
            for key, value in values[0][1].items()
            if key not in _SIGN_COLUMNS
        }
        attributes['sign'] = sign
        cleaned.add_edge(source, target, **attributes)
    return cleaned


def prepare_metab_inputs(metab_input, compartment_codes):
    """
    Prepares the metabolite inputs by adding compartment codes.

    Args:
        metab_input (dict): A dictionary containing the metabolite names and
        their corresponding values.
        compartment_codes (list): A list of compartment codes to be added to
        the metabolite names.

    Returns:
        dict: A dictionary containing the updated metabolite names with
        compartment codes.

    """
    comps = ["r", "c", "e", "x", "m", "l", "n", "g"]

    ignored = [code for code in compartment_codes if code not in comps]
    if ignored:
        _log(
            'MOON: The following compartment codes are not found in the '
            'PKN and will be ignored:'
        )
        _log(ignored)

    compartment_codes = [code for code in compartment_codes if code in comps]

    if not compartment_codes:
        _log("MOON: There are no valid compartments left. No compartment codes "
              "will be added.")
        metab_input = {
            f"Metab__{name}": value for name, value in metab_input.items()
        }

        return metab_input

    else:
        _log("MOON: Adding compartment codes.")

        metab_input_list = []

        for compartment_code in compartment_codes:
            curr_metab_input = metab_input.copy()
            curr_metab_input = {
                f"{name}_{compartment_code}": value
                for name, value in curr_metab_input.items()
            }
            curr_metab_input = {
                f"Metab__{name}": value
                for name, value in curr_metab_input.items()
            }
            metab_input_list.append(curr_metab_input)

        metab_input = {
            name: value
            for curr_metab_input in metab_input_list
            for name, value in curr_metab_input.items()
        }

        return metab_input


def _is_expressed(x, expressed_genes):
    if re.search('Metab|orphanReac', x):
        return x
    if x in expressed_genes:
        return x
    if re.search(r'^Gene[0-9]+__[A-Z0-9_]+$', x):
        genes = re.sub(r'^Gene[0-9]+__', '', x).split('_')
        return x if all(gene in expressed_genes for gene in genes) else None
    if re.search(r'^Gene[0-9]+__[A-Z0-9_]+_reverse$', x):
        genes = re.sub(r'_reverse$', '', re.sub(r'^Gene[0-9]+__', '', x))
        genes = genes.split('_')
        return x if all(gene in expressed_genes for gene in genes) else None
    if re.search(r'^Gene[0-9]+__[^_][a-z]', x):
        _log(x)
        return x
    return None


def is_expressed(x, expressed_genes_entrez):
    """Return ``x`` when it passes the current COSMOS expression predicate."""
    return _is_expressed(x, set(expressed_genes_entrez))


def filter_pkn_expressed_genes(expressed_genes_entrez, unfiltered_graph):
    """Filter PKN nodes unsupported by the supplied expressed-gene set."""
    return filter_pkn_expressed_genes_fast(
        expressed_genes_entrez, unfiltered_graph
    )


def filter_pkn_expressed_genes_fast(expressed_genes_entrez, unfiltered_graph):
    """Fast graph-native equivalent of cosmosR's vectorized PKN filter."""
    _log('MOON: removing unexpressed nodes from PKN...')
    expressed_genes = set(expressed_genes_entrez)
    graph = unfiltered_graph.copy()
    nodes_to_remove = [
        node for node in graph.nodes
        if _is_expressed(node, expressed_genes) is None
    ]
    before = graph.number_of_edges()
    graph.remove_nodes_from(nodes_to_remove)
    _log(f'MOON: {before - graph.number_of_edges()} interactions removed')
    return graph


def filter_input_nodes_not_in_pkn(data, pkn):
    """
    Filters the input nodes in the 'data' dictionary that are not present in
    the PKN.

    Args:
        data (dict): A dictionary containing the input nodes.
        pkn (nx.DiGraph): The network object representing the PKN.

    Returns:
        dict: A new dictionary containing only the input nodes that are
        present in the PKN.
    """
    new_data = {key: value for key, value in data.items() if key in pkn.nodes}

    if len(data) != len(new_data):
        removed_nodes = [
            node for node in data.keys() if node not in new_data.keys()
        ]

        _log(f"MOON: {len(removed_nodes)} input/measured nodes are not in "
              f"PKN anymore: {removed_nodes}")

    return new_data


def keep_controllable_neighbours(source_dict, graph, n_steps=None):
    """Keep descendants of input nodes, optionally within ``n_steps``.

    ``n_steps=None`` retains the historical NetworkCommons unlimited traversal.
    Supplying an integer implements the bounded behaviour of current cosmosR.
    """
    _log('MOON: filtering out nodes that are not controllable from sources...')
    nodes = _reachable_nodes(
        graph, _input_nodes(source_dict, 'source_dict'), n_steps
    )
    return graph.subgraph(nodes).copy()


def keep_observable_neighbours(target_dict, graph, n_steps=None):
    """Keep ancestors of measured nodes, optionally within ``n_steps``."""
    _log('MOON: filtering out nodes that are not observable from targets...')
    reversed_graph = graph.reverse(copy=False)
    nodes = _reachable_nodes(
        reversed_graph, _input_nodes(target_dict, 'target_dict'), n_steps
    )
    return graph.subgraph(nodes).copy()


def compress_same_children(uncompressed_graph, sig_input, metab_input):
    """
    Compresses nodes in the graph that have the same children by relabeling
    them with a common signature.

    Parameters:
    - graph (networkx.Graph): The input graph.
    - sig_input (list): List of signatures to exclude from compression.
    - metab_input (list): List of metadata signatures to exclude from
    compression.

    Returns:
    - tuple: A tuple containing the compressed subnetwork, node signatures,
    and duplicated parents.
    """
    _log("MOON: starting network compression...")
    graph = _signed_digraph(uncompressed_graph)

    parents = [node for node in graph.nodes if graph.out_degree(node) > 0]
    parents.sort()
    _log(f"MOON: {len(parents)} parents found")

    df_signature = _pkn_to_regulons(graph).sort_values(
        by=['source', 'target']
    )

    df_signature['target'] = (
        df_signature['target'].astype(str) + df_signature['sign'].astype(str)
    )

    # Create a dictionary to map each parent to its targets
    parent_to_targets = df_signature.groupby('source')['target'].apply(
        lambda targets: '_____'.join(targets)
    )

    # Generate the node signatures
    node_signatures = {
        parent: 'parent_of_' + parent_to_targets[parent]
        for parent in parents
    }

    # Count the occurrences of each signature
    filtered_signatures = {
        parent: signature
        for parent, signature in node_signatures.items()
        if parent not in sig_input and parent not in metab_input
    }

    signature_counts = collections.Counter(filtered_signatures.values())

    # Identify duplicated signatures that are not in metab_input or sig_input
    duplicated_parents = {
        node: signature for node, signature in filtered_signatures.items()
        if signature_counts[signature] > 1
    }

    _log(f"MOON: {len(duplicated_parents)} duplicated parents found")

    # Do not merge a group when a shared predecessor has conflicting signs
    # towards otherwise-identical children. Unlike the original R helper this
    # also handles duplicate root parents, which have no incoming records.
    grouped_parents = collections.defaultdict(list)
    for node, signature in duplicated_parents.items():
        grouped_parents[signature].append(node)

    excluded_nodes = set()
    potential_cases = 0
    for nodes in grouped_parents.values():
        incoming_signs = collections.defaultdict(set)
        for node in nodes:
            for parent in graph.predecessors(node):
                potential_cases += 1
                incoming_signs[parent].add(graph[parent][node]['sign'])
        if any(len(signs) > 1 for signs in incoming_signs.values()):
            excluded_nodes.update(nodes)

    _log(f"MOON: {potential_cases} potential compression cases found")
    _log(
        'MOON: '
        f'{len(excluded_nodes)} nodes excluded from compression after '
        'edge check'
    )

    new_duplicated_parents = {
        node: signature for node, signature in duplicated_parents.items()
        if node not in excluded_nodes
    }

    # Relabel the nodes in the graph based on the new duplicated signatures
    subnetwork = nx.relabel_nodes(graph, new_duplicated_parents, copy=True)

    _log(
        f'MOON: network reduced from {len(graph.nodes)} to '
        f'{len(subnetwork.nodes)} nodes after compression'
    )

    return subnetwork, node_signatures, new_duplicated_parents


def _run_decoupler(mat, regulons, statistic, n_perm):
    if statistic == 'ulm':
        estimate, _ = dc.run_ulm(
            mat=mat, net=regulons, weight='sign', min_n=1
        )
        return estimate

    # This is deliberately asymmetric: current cosmosR uses two permutations
    # for plain wmean and reserves n_perm for normalized wmean.
    times = n_perm if statistic == 'norm_wmean' else 2
    estimate, norm, _, _ = dc.run_wmean(
        mat=mat,
        net=regulons,
        times=times,
        weight='sign',
        min_n=1,
    )
    return norm if statistic == 'norm_wmean' else estimate


def run_moon_core(
        upstream_input=None,
        downstream_input=None,
        graph=None,
        n_layers=None,
        n_perm=1000,
        downstream_cutoff=0,
        statistic='ulm',
        return_levels=False,
):
    """Iteratively propagate downstream activity through a signed PKN.

    ``sign``, ``interaction``, and ``mor`` edge attributes are accepted. The
    input graph is not modified; decoupler receives a local ``sign`` column.
    ``return_levels`` is accepted for R compatibility and, like cosmosR,
    levels are always returned.
    """
    if statistic not in {'ulm', 'wmean', 'norm_wmean'}:
        raise ValueError(
            "Invalid statistic. Supported values are 'ulm', 'wmean', and "
            "'norm_wmean'."
        )
    if graph is None:
        raise ValueError('graph must be provided.')
    if n_layers is None or n_layers < 1:
        raise ValueError('n_layers must be a positive integer.')

    downstream_input = _as_named_scores(downstream_input, 'downstream_input')
    upstream_input = (
        None if upstream_input is None
        else _as_named_scores(upstream_input, 'upstream_input')
    )
    regulons = _pkn_to_regulons(graph)
    regulons = regulons.loc[
        ~regulons['source'].isin(downstream_input)
    ].copy()
    decoupler_mat = pd.DataFrame([downstream_input], index=['sample'])

    estimate = _run_decoupler(
        decoupler_mat, regulons, statistic, n_perm
    )
    n_plus_one = estimate.T
    n_plus_one.columns = ['score']
    n_plus_one['level'] = 1
    results = [n_plus_one]

    layer = 1
    while (
        len(regulons) > 1
        and regulons['target'].isin(results[layer - 1].index).sum() > 1
        and layer < n_layers
    ):
        _log(f'MOON: scoring layer {layer} from downstream nodes...')
        regulons = regulons.loc[
            ~regulons['source'].isin(results[layer - 1].index)
        ].copy()
        previous_layer = results[layer - 1].drop(columns='level').T
        estimate = _run_decoupler(
            previous_layer, regulons, statistic, n_perm
        )
        n_plus_one = estimate.T
        regulons = regulons.loc[
            ~regulons['source'].isin(n_plus_one.index)
        ].copy()
        n_plus_one.columns = ['score']
        n_plus_one['level'] = layer + 1
        results.append(n_plus_one)
        layer += 1

    moon_res = pd.concat(results)
    downstream_names = pd.DataFrame.from_dict(
        downstream_input, orient='index', columns=['score']
    )
    downstream_names = downstream_names.loc[
        downstream_names['score'].abs() > downstream_cutoff
    ]
    downstream_names['level'] = 0
    moon_res = pd.concat([moon_res, downstream_names])

    if upstream_input is not None:
        real_scores = pd.Series(upstream_input, name='real_score')
        observed_scores = moon_res.index.to_series().map(real_scores)
        coherent = observed_scores.isna() | (
            np.sign(observed_scores) == np.sign(moon_res['score'])
        )
        moon_res = moon_res.loc[coherent]

    return moon_res.reset_index().rename(columns={'index': 'source'})


def run_moon(
        network,
        sig_input,
        metab_input,
        tf_regn,
        rna_input,
        n_layers=6,
        method='ulm',
        max_iter=10,
        n_perm=1000,
        downstream_cutoff=0,
):
    """Run iterative MOON scoring and TF-target coherence filtering.

    Args:
        network: Signed directed PKN. ``sign``, ``interaction``, and ``mor``
            edge attributes are accepted.
        sig_input: Upstream node-to-score mapping.
        metab_input: Downstream node-to-score mapping.
        tf_regn: TF regulon table with ``source``, ``target``, and a signed
            ``mor``/``weight``/``sign``/``interaction`` column.
        rna_input: RNA target node-to-score mapping.
        n_layers (int, optional): The number of layers in the MOON algorithm.
            Defaults to 6.
        method (str, optional): ``'ulm'``, ``'wmean'``, or ``'norm_wmean'``.
        max_iter (int, optional): The maximum number of iterations for the
            MOON algorithm. Defaults to 10.
        n_perm (int, optional): Permutations for ``'norm_wmean'``. Plain
            ``'wmean'`` uses the two permutations used by current cosmosR.
        downstream_cutoff (float, optional): Minimum absolute downstream
            score retained as a level-0 result.

    Returns:
        tuple: A tuple containing the MOON scores and the modified network.
    """
    if max_iter < 1:
        raise ValueError('max_iter must be a positive integer.')
    _log('MOON: starting MOON scoring...')

    moon_network = network.copy()

    before = 1
    after = 0
    i = 0

    while before != after and i < max_iter:
        before = len(moon_network.edges)
        moon_res = run_moon_core(
            upstream_input=sig_input,
            downstream_input=metab_input,
            graph=moon_network,
            n_layers=n_layers,
            n_perm=n_perm,
            downstream_cutoff=downstream_cutoff,
            statistic=method,
        )

        moon_network = filter_incoherent_TF_target(
            moon_res,
            tf_regn,
            moon_network,
            rna_input,
        )

        after = len(moon_network.edges)
        i += 1
        _log(f'Optimisation iteration {i} - Before: {before}, After: {after}')

    if before == after:
        _log(f'MOON: Solution converged after {i} iterations')
    else:
        _log(
            'MOON: Maximum number of iterations reached.'
            'Solution might not have converged'
        )

    return moon_res, moon_network


def _normalise_tf_regulon(tf_reg_net):
    if not isinstance(tf_reg_net, pd.DataFrame):
        raise TypeError('TF_reg_net must be a pandas DataFrame.')
    if not {'source', 'target'}.issubset(tf_reg_net.columns):
        raise ValueError(
            "TF_reg_net must contain 'source', 'target', and a signed "
            "interaction column."
        )
    result = tf_reg_net[['source', 'target']].copy()
    result['mor'] = [
        _canonical_sign(
            row, f'TF interaction {row["source"]!r}->{row["target"]!r}'
        )
        for _, row in tf_reg_net.iterrows()
    ]
    return result


def filter_incoherent_TF_target(
        moon_res, TF_reg_net, meta_network, RNA_input
):
    """Remove TF-target edges whose score, RNA, and regulation signs conflict.

    The TF network may use current cosmosR's ``mor`` or NetworkCommons'
    historical ``weight`` column (as well as ``sign`` or ``interaction``).
    """
    if 'source' not in moon_res.columns or 'score' not in moon_res.columns:
        raise ValueError("moon_res must contain 'source' and 'score' columns.")
    filtered_meta_network = meta_network.copy()
    rna_scores = _as_named_scores(RNA_input, 'RNA_input')
    regulon = _normalise_tf_regulon(TF_reg_net)
    rna_df = pd.DataFrame.from_dict(
        rna_scores, orient='index', columns=['RNA_input']
    )
    reg_meta = moon_res.merge(regulon, on='source', how='inner')
    reg_meta = reg_meta.rename(columns={'score': 'TF_score'})
    reg_meta = reg_meta.merge(
        rna_df, left_on='target', right_index=True, how='inner'
    )
    incoherent = np.sign(
        reg_meta['TF_score'] * reg_meta['RNA_input'] * reg_meta['mor']
    ) < 0
    incoherent = incoherent.fillna(False)
    edges_to_remove = list(reg_meta.loc[
        incoherent, ['source', 'target']
    ].itertuples(index=False, name=None))
    filtered_meta_network.remove_edges_from(edges_to_remove)
    return filtered_meta_network


# Preserve the public cosmosR spelling for users moving a workflow verbatim.
filter_incohrent_TF_target = filter_incoherent_TF_target


def decompress_moon_result(
        moon_res,
        node_signatures,
        duplicated_parents=None,
        meta_network_graph=None,
):
    """Expand compressed MOON scores back to original PKN node identifiers.

    The existing four-argument Python form is retained. The current R-style
    ``(moon_res, compression_result, meta_network)`` form is also accepted,
    including the three-tuple returned by ``compress_same_children``.
    """
    _log('MOON: decompressing nodes...')
    if isinstance(node_signatures, (tuple, list)):
        if len(node_signatures) != 3:
            raise ValueError(
                'A compression result tuple must contain exactly three items.'
            )
        compression_result = node_signatures
        if meta_network_graph is None:
            meta_network_graph = duplicated_parents
        _, node_signatures, duplicated_parents = compression_result
    elif (
        isinstance(node_signatures, Mapping)
        and 'node_signatures' in node_signatures
    ):
        compression_result = node_signatures
        if meta_network_graph is None:
            meta_network_graph = duplicated_parents
        node_signatures = compression_result['node_signatures']
        duplicated_parents = compression_result.get(
            'duplicated_signatures',
            compression_result.get('duplicated_parents', {}),
        )

    if meta_network_graph is None:
        raise ValueError('meta_network_graph must be provided.')
    if 'source' not in moon_res.columns:
        raise ValueError("moon_res must contain a 'source' column.")
    node_signatures = dict(node_signatures)
    duplicated_parents = dict(duplicated_parents or {})

    records = [
        {'source': compressed, 'source_original': original}
        for original, compressed in duplicated_parents.items()
    ]
    duplicated_originals = set(duplicated_parents)
    records.extend(
        {'source': node, 'source_original': node}
        for node in node_signatures
        if node not in duplicated_originals
    )
    # Match the R edge-table semantics: a final leaf is a target which never
    # occurs as a source. Isolated NetworkX vertices are not represented in
    # the R input and must therefore not add decompression rows here.
    edge_sources = {
        source for source, _, _ in _iter_edges_with_data(meta_network_graph)
    }
    final_leaves = {
        target for _, target, _ in _iter_edges_with_data(meta_network_graph)
        if target not in edge_sources
    }
    records.extend(
        {'source': node, 'source_original': node}
        for node in final_leaves
    )
    mapping_table = pd.DataFrame(
        records, columns=['source', 'source_original']
    ).drop_duplicates()
    moon_res_dec = moon_res.merge(mapping_table, on='source', how='inner')
    _log(f'MOON: decompressed {len(moon_res_dec) - len(moon_res)} nodes')
    return moon_res_dec


def reduce_solution_network(
        moon_res,
        meta_network,
        cutoff,
        sig_input,
        rna_input=None,
        n_steps=10,
):
    """Extract the current cosmosR single-threshold MOON solution network.

    The graph-native result is ``(network, ATT)``. The returned graph is the
    SIF equivalent and contains canonical ``sign`` plus ``consistency=True``
    edge attributes. ``source_original`` is accepted for legacy decompressed
    Python results when it matches the original PKN node identifiers.
    """
    _log('MOON: reducing solution network...')
    n_steps = _validated_n_steps(n_steps)
    if n_steps is None:
        raise ValueError('n_steps must be a non-negative integer.')
    upstream_input = _as_named_scores(sig_input, 'sig_input')
    source_graph = _signed_digraph(meta_network)
    scores = _score_table(moon_res, source_graph)
    qualified = scores.loc[scores['score'].abs() > cutoff].copy()
    _log(
        f'MOON: {len(scores) - len(qualified)} nodes removed by the '
        'score cutoff'
    )

    stable_columns = ['nodes', 'score', 'level', 'RNA_input']
    if qualified.empty:
        return nx.DiGraph(), _empty_att(stable_columns)

    score_by_node = qualified.set_index('source')['score'].to_dict()
    level_by_node = qualified.set_index('source')['level'].to_dict()
    valid_nodes = set(qualified['source'])
    result = nx.DiGraph()
    result.add_nodes_from(
        (node, dict(source_graph.nodes[node]))
        for node in valid_nodes if node in source_graph
    )
    result.add_nodes_from(node for node in valid_nodes if node not in result)

    for source, target, attributes in source_graph.edges(data=True):
        if source not in valid_nodes or target not in valid_nodes:
            continue
        if (
            np.sign(score_by_node[source] * score_by_node[target])
            != attributes['sign']
        ):
            continue
        copied_attributes = dict(attributes)
        copied_attributes['consistency'] = True
        result.add_edge(source, target, **copied_attributes)

    seeds = [node for node in upstream_input if node in result]
    if not seeds:
        _log('MOON: no upstream input nodes found in the qualified network.')
        return nx.DiGraph(), _empty_att(stable_columns)

    result = result.subgraph(_reachable_nodes(result, seeds, n_steps)).copy()
    while result.number_of_nodes() > 0:
        bad_children = [
            node for node in result
            if result.out_degree(node) == 0 and level_by_node[node] != 0
        ]
        bad_parents = [
            node for node in result
            if result.in_degree(node) == 0 and node not in upstream_input
        ]
        to_remove = set(bad_children).union(bad_parents)
        if not to_remove:
            break
        result.remove_nodes_from(to_remove)

    nx.set_node_attributes(
        result,
        {node: score_by_node[node] for node in result.nodes},
        'moon_score',
    )
    nx.set_node_attributes(
        result, {node: level_by_node[node] for node in result.nodes}, 'level'
    )
    att = qualified.loc[
        qualified['source'].isin(result.nodes), ['source', 'score', 'level']
    ].rename(columns={'source': 'nodes'})
    return result, _attach_rna(att, rna_input, 'nodes')


def reduce_solution_network_double_thresh(
        moon_res,
        meta_network,
        primary_thresh,
        secondary_thresh,
        sig_input,
        rna_input=None,
):
    """Extract the current cosmosR two-threshold MOON solution network.

    The path restriction intentionally follows the present R implementation:
    it runs only after a selected sign-incoherent edge is removed. This keeps
    the current cross-language behaviour, including disconnected components
    when every initially selected edge is coherent.
    """
    upstream_input = _as_named_scores(sig_input, 'sig_input')
    source_graph = _signed_digraph(meta_network)
    scores = _score_table(moon_res, source_graph, require_level=True)
    primary_nodes = set(scores.loc[
        scores['score'].abs() > primary_thresh, 'source'
    ])
    secondary_nodes = set(scores.loc[
        scores['score'].abs() > secondary_thresh, 'source'
    ])
    score_by_node = scores.set_index('source')['score'].to_dict()

    result = nx.DiGraph()
    for source, target, attributes in source_graph.edges(data=True):
        if (
            source in secondary_nodes
            and target in secondary_nodes
            and (source in primary_nodes or target in primary_nodes)
        ):
            result.add_edge(source, target, **dict(attributes))

    while result.number_of_edges() > 0:
        incoherent_edges = [
            (source, target)
            for source, target, attributes in result.edges(data=True)
            if np.sign(score_by_node[source] * score_by_node[target])
            != attributes['sign']
        ]
        if not incoherent_edges:
            break
        result.remove_edges_from(incoherent_edges)
        if result.number_of_edges() == 0:
            # An R edge table with no rows has no vertices. ``remove_edges``
            # leaves isolated NetworkX vertices behind, so clear them here.
            result = nx.DiGraph()
            break

        seeds = [node for node in upstream_input if node in result]
        level_zero = set(scores.loc[
            scores['level'] == 0, 'source'
        ]).intersection(result.nodes)
        if not seeds or not level_zero:
            result = nx.DiGraph()
            break
        forward = _reachable_nodes(result, seeds, None)
        backward = _reachable_nodes(
            result.reverse(copy=False), level_zero, None
        )
        result = _edge_only_subgraph(result, forward.intersection(backward))

    final_nodes = set(result.nodes)
    att = scores.loc[
        scores['source'].isin(final_nodes), ['source', 'score', 'level']
    ].copy()
    att['type'] = np.where(
        att['source'].isin(upstream_input),
        'upstream_input',
        np.where(att['level'] == 0, 'level0', 'other'),
    )
    nx.set_node_attributes(
        result,
        {node: score_by_node[node] for node in result.nodes},
        'moon_score',
    )
    return result, _attach_rna(att, rna_input, 'source')


def get_moon_scoring_network(
        upstream_node,
        meta_network,
        moon_scores,
        keep_upstream_node_peers=False,
):
    """Return the score-explanation subnetwork for one upstream MOON node."""
    source_graph = _signed_digraph(meta_network)
    scores = _score_table(moon_scores, source_graph, require_level=True)
    upstream_rows = scores.loc[scores['source'] == upstream_node]
    if len(upstream_rows) != 1:
        raise ValueError(
            'upstream_node must occur exactly once in moon_scores.'
        )
    n_steps = int(upstream_rows['level'].iloc[0])
    if n_steps < 0:
        raise ValueError('The upstream node level must be non-negative.')

    if not keep_upstream_node_peers:
        scores = scores.loc[
            ~(
                (scores['level'] == n_steps)
                & (scores['source'] != upstream_node)
            )
        ].copy()

    result = keep_controllable_neighbours(
        {upstream_node: 1}, source_graph, n_steps=n_steps
    )
    result = _edge_only_subgraph(result)
    targets = {target for _, target in result.edges()}
    downstream_nodes = set(scores.loc[
        (scores['level'] == 0) & scores['source'].isin(targets), 'source'
    ])
    if not downstream_nodes:
        return nx.DiGraph(), scores.iloc[0:0].copy()

    result = keep_observable_neighbours(
        {node: 1 for node in downstream_nodes}, result, n_steps=n_steps
    )
    result = _edge_only_subgraph(result)
    scores = scores.loc[scores['source'].isin(result.nodes)].copy()
    result = _edge_only_subgraph(result, set(scores['source']))

    if n_steps > 1 and not keep_upstream_node_peers:
        for level in range(n_steps, -1, -1):
            top_nodes = set(scores.loc[
                scores['level'] == level, 'source'
            ])
            child_nodes = {
                target for source, target in result.edges()
                if source in top_nodes
            }
            scores = scores.loc[
                scores['source'].isin(child_nodes)
                | (scores['level'] != level - 1)
            ].copy()
            result = _edge_only_subgraph(result, set(scores['source']))

    score_by_node = scores.set_index('source')['score'].to_dict()
    nx.set_node_attributes(
        result,
        {node: score_by_node[node] for node in result.nodes},
        'moon_score',
    )
    return result, scores


def get_ego_graph(G, sources, depth_limit=7):
    """
    Returns the ego graph of the given network graph G, centered around the
    specified sources.

    Parameters:
        G (networkx.DiGraph): The network graph.
        sources (list): The list of source nodes.
        depth_limit (int, optional): The depth limit for collecting
        descendants. Default is 7.

    Returns:
        networkx.DiGraph: The ego graph centered around the sources.
    """
    reached_nodes = set()
    for source in sources:
        descendant_dict = nx.ego_graph(G,
                                       source,
                                       radius=depth_limit,
                                       center=True,
                                       undirected=False)
        reached_nodes.update(descendant_dict.nodes)

    return G.subgraph(reached_nodes).copy()


def _translate_label(name, mapping_dict, keep_unmapped_suffix):
    if name is None or pd.isna(name):
        return name
    translated = re.sub(r'^Metab__', '', str(name))
    translated = re.sub(r'^Gene', 'Enzyme', translated)
    suffix_match = re.search(r'_[a-z]$', translated)
    suffix = suffix_match.group() if suffix_match else ''
    translated = re.sub(r'_[a-z]$', '', translated)
    if translated in mapping_dict:
        return f'Metab__{mapping_dict[translated]}{suffix}'
    return f'{translated}{suffix}' if keep_unmapped_suffix else translated


def translate_column_HMDB(my_column, mapping_dict):
    """Translate a node column using an HMDB mapping like current cosmosR."""
    if mapping_dict is None:
        raise ValueError(
            'mapping_dict is required because NetworkCommons has no '
            'packaged HMDB map.'
        )

    translate = lambda value: _translate_label(value, mapping_dict, True)
    if isinstance(my_column, pd.Series):
        return my_column.map(translate)
    if isinstance(my_column, pd.Index):
        return pd.Index(
            [translate(value) for value in my_column], name=my_column.name
        )
    if isinstance(my_column, str):
        return translate(my_column)
    return [translate(value) for value in my_column]


def translate_res(untranslated_network, att, mapping_dict):
    """Translate a graph-native SIF equivalent and node attribute table.

    This mirrors current cosmosR's intentionally distinct SIF and ATT suffix
    handling: unmapped endpoint labels lose a compartment suffix in the graph,
    while ATT labels retain it.
    """
    _log('MOON: translating network and attribute table...')
    if mapping_dict is None:
        raise ValueError(
            'mapping_dict is required because NetworkCommons has no '
            'packaged HMDB map.'
        )
    if not isinstance(att, pd.DataFrame) or len(att.columns) == 0:
        raise ValueError('att must be a pandas DataFrame with node labels.')

    node_column = 'nodes' if 'nodes' in att.columns else att.columns[0]
    network = untranslated_network.copy()
    renamed_nodes = {
        node: _translate_label(node, mapping_dict, False)
        for node in network.nodes
    }
    network = nx.relabel_nodes(network, renamed_nodes, copy=True)
    translated_att = att.copy()
    translated_att[node_column] = translated_att[node_column].map(
        lambda value: _translate_label(value, mapping_dict, True)
    )
    _log('MOON: nodes translated')
    return network, translated_att
