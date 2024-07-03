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
Plots for omics data exploration.
"""

from __future__ import annotations

__all__  = [
    'build_volcano_plot',
    'build_ma_plot',
    'build_pca_plot',
    'build_heatmap_with_tree',
]

import lazy_import
import pandas as pd
import numpy as np
plt = lazy_import.lazy_module('matplotlib.pyplot')
sns = lazy_import.lazy_module('seaborn')
sklearn_decomp = lazy_import.lazy_module('sklearn.decomposition')


def build_volcano_plot(
        data: pd.DataFrame,
        log2fc: 'str' = 'log2FoldChange',
        pval: str = 'pvalue',
        pval_threshold: float = 0.05,
        log2fc_threshold: float = 1,
        title: str = "Volcano Plot",
        xlabel: str = "log2 Fold Change",
        ylabel: str = "-log10(p-value)",
        colors: tuple = ("gray", "red", "blue"),
        alpha: float = 0.7,
        size: int = 50,
        save: bool = False,
        output_dir: str = "."
):
    data['-log10(pval)'] = -np.log10(data[pval])
    data['significant'] = (data[pval] < pval_threshold) & (abs(data[log2fc]) >= log2fc_threshold)
    data['Upregulated'] = (data[pval] < pval_threshold) & (data[log2fc] >= log2fc_threshold)
    data['Downregulated'] = (data[pval] < pval_threshold) & (data[log2fc] <= -log2fc_threshold)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        data.loc[~data['significant'], log2fc],
        data.loc[~data['significant'], '-log10(pval)'],
        c=colors[0],
        alpha=alpha,
        s=size,
        label='Non-significant'
    )

    ax.scatter(
        data.loc[data['Upregulated'], log2fc],
        data.loc[data['Upregulated'], '-log10(pval)'],
        c=colors[1],
        alpha=alpha,
        s=size,
        label='Upregulated: ' + str(len(data.loc[data['Upregulated'], log2fc])) + ' genes'
    )

    ax.scatter(
        data.loc[data['Downregulated'], log2fc],
        data.loc[data['Downregulated'], '-log10(pval)'],
        c=colors[2],
        alpha=alpha,
        s=size,
        label= 'Downregulated: ' + str(len(data.loc[data['Downregulated'], log2fc])) + ' genes'
    )

    ax.axhline(
        -np.log10(pval_threshold),
        color="blue",
        linestyle="--"
    )

    ax.axvline(
        log2fc_threshold,
        color="blue",
        linestyle="--"
    )
    ax.axvline(
        -log2fc_threshold,
        color="blue",
        linestyle="--"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if save:
        plt.savefig(f"{output_dir}/volcano_plot.png")

    plt.show()


def build_ma_plot(
        data: pd.DataFrame,
        log2fc: str,
        mean_exp: str,
        log2fc_threshold: float = 1,
        title: str = "MA Plot",
        xlabel: str = "Mean Expression",
        ylabel: str = "log2 Fold Change",
        colors: tuple = ("gray", "red"),
        alpha: float = 0.7,
        size: int = 50,
        save: bool = False,
        output_dir: str = "."
):
    data['significant'] = abs(data[log2fc]) >= log2fc_threshold

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        data.loc[~data['significant'], mean_exp],
        data.loc[~data['significant'], log2fc],
        c=colors[0],
        alpha=alpha,
        s=size,
        label='Non-significant'
    )

    ax.scatter(
        data.loc[data['significant'], mean_exp],
        data.loc[data['significant'], log2fc],
        c=colors[1],
        alpha=alpha,
        s=size,
        label='Significant'
    )

    ax.axhline(
        0,
        color="blue",
        linestyle="--"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if save:
        plt.savefig(f"{output_dir}/ma_plot.png")

    plt.show()


def build_pca_plot(
        data: pd.DataFrame,
        title: str = "PCA Plot",
        xlabel: str = "PC1",
        ylabel: str = "PC2",
        alpha: float = 0.7,
        size: int = 50,
        save: bool = False,
        output_dir: str = "."
):
    pca = sklearn_decomp.PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        alpha=alpha,
        s=size
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if save:
        plt.savefig(f"{output_dir}/pca_plot.png")

    plt.show()


def build_heatmap_with_tree(
        data: pd.DataFrame,
        top_n: int = 50,
        value_column: str = 'log2FoldChange_condition_1',
        conditions: list[str] = None,
        title: str = "Heatmap of Top Differentially Expressed Genes",
        save: bool = False,
        output_dir: str = "."
):
    """
    Build a heatmap with hierarchical clustering for the top differentially expressed genes across multiple conditions.

    Args:
        data (pd.DataFrame): DataFrame containing RNA-seq results.
        top_n (int): Number of top differentially expressed genes to include in the heatmap.
        value_column (str): Column name for the values to rank and select the top genes.
        conditions (list[str]): List of condition columns to include in the heatmap.
        title (str): Title of the plot.
        save (bool): Whether to save the plot. Default is False.
        output_dir (str): Directory to save the plot. Default is ".".
    """
    if conditions is None:
        raise ValueError("Conditions must be provided as a list of column names.")

    # Select top differentially expressed genes
    top_genes = data.nlargest(top_n, value_column).index
    top_data = data.loc[top_genes, conditions]

    # Create the clustermap
    g = sns.clustermap(top_data, cmap="viridis", cbar=True, fmt=".2f", linewidths=.5)

    plt.title(title)
    plt.ylabel("Gene")
    plt.xlabel("Condition")

    if save:
        plt.savefig(f"{output_dir}/heatmap_with_tree.png")

    plt.show()
