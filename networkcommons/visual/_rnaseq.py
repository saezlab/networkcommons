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

This module provides functions for creating various plots to visualize and explore omics data,
including density plots, volcano plots, MA plots, PCA plots, and heatmaps with hierarchical clustering.
"""

from __future__ import annotations
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    'build_volcano_plot',
    'build_ma_plot',
    'plot_pca',
    'build_heatmap_with_tree',
    'plot_density'
]

import lazy_import
import numpy as np
from networkcommons.utils import handle_missing_values


def plot_density(df: pd.DataFrame,
                 gene_ids: list[str],
                 metadata: pd.DataFrame = None,
                 id_column: str = 'idx',
                 sample_id_column: str = 'sample_ID',
                 group_column: str = 'group',
                 quantiles: list[int] = [10, 90],
                 title: str = 'Density Plot of Intensity Values',
                 xlabel: str = 'Intensity',
                 ylabel: str = 'Density'):
    """
    Plots density of intensity values for specified genes, including mean and quantile lines.
    Distributions can be separated by groups if metadata is provided. Each gene is displayed in a separate subplot.

    Args:
        df (pd.DataFrame): Input DataFrame containing gene data.
        gene_ids (list of str): List of specific genes to highlight.
        metadata (pd.DataFrame, optional): Optional DataFrame containing metadata (sample_ID, group).
        id_column (str, optional): Column name for identifying the gene. Defaults to 'idx'.
        sample_id_column (str, optional): Column name in metadata for sample IDs. Defaults to 'sample_ID'.
        group_column (str, optional): Column name in metadata for groups. Defaults to 'group'.
        quantiles (list of int, optional): List of quantiles to plot. Defaults to [10, 90].
        title (str, optional): Title of the plot. Defaults to 'Density Plot of Intensity Values'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Intensity'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Density'.

    Returns:
        None
    """
    num_genes = len(gene_ids)
    num_cols = 3
    num_rows = (num_genes + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for idx, gene_id in enumerate(gene_ids):
        specific_gene = df[df[id_column].str.contains(gene_id, na=False)]
        if not specific_gene.empty:
            values = specific_gene.iloc[0, 1:].astype(float)
            ax = axes[idx]

            if metadata is not None:
                merged_df = pd.DataFrame(values).reset_index()
                merged_df.columns = [sample_id_column, 'intensity']
                merged_df = merged_df.merge(metadata, left_on=sample_id_column, right_on=sample_id_column)

                groups = merged_df[group_column].unique()
                for group in groups:
                    group_values = merged_df[merged_df[group_column] == group]['intensity']
                    group_values.plot(kind='density', ax=ax, label=f'Group: {group}')

                    mean_value = group_values.mean()
                    ax.axvline(mean_value, linestyle='--', linewidth=1,
                               label=f'Group {group} Mean: {np.round(mean_value, 2)}')

                    for quantile in quantiles:
                        q = np.percentile(group_values, quantile)
                        ax.axvline(q, linestyle=':', linewidth=1,
                                   label=f'Group {group} Q{quantile}: {np.round(q, 2)}')
            else:
                values.plot(kind='density', ax=ax, label='Density')

                mean_value = values.mean()
                ax.axvline(mean_value, linestyle='--', linewidth=1,
                           label=f'Mean: {np.round(mean_value, 2)}')

                for quantile in quantiles:
                    q = np.percentile(values, quantile)
                    ax.axvline(q, linestyle=':', linewidth=1,
                               label=f'Q{quantile}: {np.round(q, 2)}')

            ax.set_title(f'Gene: {gene_id}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    for i in range(len(gene_ids), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def build_volcano_plot(
        data: pd.DataFrame,
        log2fc: str = 'log2FoldChange',
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
        output_dir: str = ".",
        render: bool = False
):
    """
    Creates a volcano plot to visualize differential expression analysis results.

    Args:
        data (pd.DataFrame): DataFrame containing the data for plotting.
        log2fc (str, optional): Column name for log2 fold change. Defaults to 'log2FoldChange'.
        pval (str, optional): Column name for p-value. Defaults to 'pvalue'.
        pval_threshold (float, optional): Threshold for significance based on p-value. Defaults to 0.05.
        log2fc_threshold (float, optional): Threshold for fold change. Defaults to 1.
        title (str, optional): Title of the plot. Defaults to "Volcano Plot".
        xlabel (str, optional): Label for the x-axis. Defaults to "log2 Fold Change".
        ylabel (str, optional): Label for the y-axis. Defaults to "-log10(p-value)".
        colors (tuple, optional): Colors for the plot. Defaults to ("gray", "red", "blue").
        alpha (float, optional): Transparency level of the plot points. Defaults to 0.7.
        size (int, optional): Size of the plot points. Defaults to 50.
        save (bool, optional): Whether to save the plot. Defaults to False.
        output_dir (str, optional): Directory to save the plot if `save` is True. Defaults to ".".
        render (bool, optional): Whether to show the plot. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    data = data.copy()
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
        label='Downregulated: ' + str(len(data.loc[data['Downregulated'], log2fc])) + ' genes'
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
        plt.savefig(f"{output_dir}/volcano_plot.png", bbox_inches='tight')

    if render:
        plt.show()

    return fig


def build_ma_plot(data: pd.DataFrame,
                  log2fc: str = 'log2FoldChange',
                  avg_expr: str = 'average_expression',
                  title: str = 'MA Plot',
                  xlabel: str = 'Average Expression',
                  ylabel: str = 'log2 Fold Change',
                  threshold: float = 1,
                  color: str = 'blue',
                  alpha: float = 0.7,
                  size: int = 50,
                  save: bool = False,
                  output_dir: str = ".",
                  render: bool = False):
    """
    Creates an MA plot to visualize the relationship between mean expression and fold change.

    Args:
        data (pd.DataFrame): DataFrame containing the data for plotting.
        log2fc (str, optional): Column name for log2 fold change. Defaults to 'log2FoldChange'.
        avg_expr (str, optional): Column name for average expression. Defaults to 'average_expression'.
        title (str, optional): Title of the plot. Defaults to 'MA Plot'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Average Expression'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'log2 Fold Change'.
        threshold (float, optional): Threshold for fold change. Defaults to 1.
        color (str, optional): Color for the plot points. Defaults to 'blue'.
        alpha (float, optional): Transparency level of the plot points. Defaults to 0.7.
        size (int, optional): Size of the plot points. Defaults to 50.
        save (bool, optional): Whether to save the plot. Defaults to False.
        output_dir (str, optional): Directory to save the plot if `save` is True. Defaults to ".".
        render (bool, optional): Whether to show the plot. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        data[avg_expr],
        data[log2fc],
        c=color,
        alpha=alpha,
        s=size
    )

    ax.axhline(
        threshold,
        color="red",
        linestyle="--"
    )
    ax.axhline(
        -threshold,
        color="red",
        linestyle="--"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if save:
        plt.savefig(f"{output_dir}/ma_plot.png", bbox_inches='tight')

    if render:
        plt.show()

    return fig


def plot_pca(data: pd.DataFrame,
             metadata: pd.DataFrame = None,
             components: list[int] = [1, 2],
             color_by: str = None,
             title: str = 'PCA Plot',
             xlabel: str = 'PC1',
             ylabel: str = 'PC2',
             save: bool = False,
             output_dir: str = ".",
             render: bool = False):
    """
    Creates a PCA plot to visualize the first two principal components.

    Args:
        data (pd.DataFrame): DataFrame containing the data for PCA.
        metadata (pd.DataFrame, optional): DataFrame containing metadata for coloring. Defaults to None.
        components (list of int, optional): List of principal components to plot. Defaults to [1, 2].
        color_by (str, optional): Column name in metadata for coloring points. Defaults to None.
        title (str, optional): Title of the plot. Defaults to 'PCA Plot'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'PC1'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'PC2'.
        save (bool, optional): Whether to save the plot. Defaults to False.
        output_dir (str, optional): Directory to save the plot if `save` is True. Defaults to ".".
        render (bool, optional): Whether to show the plot. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    pca = PCA(n_components=max(components))
    pca_results = pca.fit_transform(data)

    pca_df = pd.DataFrame(pca_results, columns=[f'PC{i + 1}' for i in range(pca_results.shape[1])])

    if metadata is not None:
        pca_df = pd.concat([pca_df, metadata], axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))

    if color_by and color_by in metadata.columns:
        sns.scatterplot(
            data=pca_df,
            x=f'PC{components[0]}',
            y=f'PC{components[1]}',
            hue=color_by,
            ax=ax
        )
    else:
        sns.scatterplot(
            data=pca_df,
            x=f'PC{components[0]}',
            y=f'PC{components[1]}',
            ax=ax
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if save:
        plt.savefig(f"{output_dir}/pca_plot.png", bbox_inches='tight')

    if render:
        plt.show()

    return fig


def build_heatmap_with_tree(
        data: pd.DataFrame,
        clustering_method: str = 'ward',
        metric: str = 'euclidean',
        title: str = 'Heatmap with Hierarchical Clustering',
        xlabel: str = 'Samples',
        ylabel: str = 'Genes',
        cmap: str = 'viridis',
        save: bool = False,
        output_dir: str = ".",
        render: bool = False
):
    """
    Creates a heatmap with hierarchical clustering for rows and columns.

    Args:
        data (pd.DataFrame): DataFrame containing the data for the heatmap.
        clustering_method (str, optional): Method for hierarchical clustering. Defaults to 'ward'.
        metric (str, optional): Metric for distance calculation. Defaults to 'euclidean'.
        title (str, optional): Title of the plot. Defaults to 'Heatmap with Hierarchical Clustering'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Samples'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Genes'.
        cmap (str, optional): Colormap for the heatmap. Defaults to 'viridis'.
        save (bool, optional): Whether to save the plot. Defaults to False.
        output_dir (str, optional): Directory to save the plot if `save` is True. Defaults to ".".
        render (bool, optional): Whether to show the plot. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    # Compute the distance matrices
    row_linkage = sns.clustermap(data, method=clustering_method, metric=metric, cmap=cmap)
    col_linkage = sns.clustermap(data.T, method=clustering_method, metric=metric, cmap=cmap)

    fig = plt.figure(figsize=(10, 10))

    sns.heatmap(data, cmap=cmap, ax=fig.gca(), cbar=True, annot=False, fmt=".2f")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save:
        plt.savefig(f"{output_dir}/heatmap_with_tree.png", bbox_inches='tight')

    if render:
        plt.show()

    return fig
