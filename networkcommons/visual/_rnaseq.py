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
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


__all__  = [
    'build_volcano_plot',
    'build_ma_plot',
    'plot_pca',
    'build_heatmap_with_tree',
    'plot_density'
]

import lazy_import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition as sklearn_decomp
from networkcommons._utils import handle_missing_values


def plot_density(df,
                 ensembl_ids,
                 id_column='idx',
                 quantiles=[10, 90],
                 title='Density Plot of Intensity Values',
                 xlabel='Intensity',
                 ylabel='Density'):
    """
    Plots density of intensity values for specified genes, including mean and quantile lines.

    Args:
        df (pd.DataFrame): Input DataFrame containing gene data.
        ensembl_ids (list of str): List of specific genes to highlight.
        id_column (str): Column name for identifying the gene.
        quantiles (list of int): List of quantiles to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Returns:
        None
    """
    # Ensure all relevant columns are numeric
    numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    plt.figure(figsize=(12, 6))
    
    for ensembl_id in ensembl_ids:
        specific_gene = df[df[id_column].str.contains(ensembl_id, na=False)]
        if not specific_gene.empty:
            values = specific_gene.iloc[0, 1:].astype(float)
            values.plot(kind='density', label=f'Gene: {ensembl_id}')
            
            # Mean value line
            mean_value = values.mean()
            plt.axvline(mean_value, color='k', linestyle='--', linewidth=1, label=f'Mean: {np.round(mean_value, 2)}')
            
            for quantile in quantiles:
                q = np.percentile(values, quantile)
                plt.axvline(q, color='k', linestyle=':', linewidth=1, label=f'Q{quantile}: {np.round(q, 2)}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()



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


def plot_pca(dataframe, metadata, feature_col='idx', **kwargs):
    """
    Plots the PCA (Principal Component Analysis) of a dataframe.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing numeric columns.
        metadata (pd.DataFrame or array-like): The metadata associated with the dataframe or an array-like object representing the groups.

    Returns:
        pd.DataFrame: The dataframe with PCA results.

    Raises:
        ValueError: If the dataframe contains no numeric columns suitable for PCA.
    """

    # Check if the dataframe contains any non-numeric columns
    df = dataframe.copy()

    if df.isna().sum().sum() > 0:
        print("Warning: Missing values were found in the input data and will be filled with the handle_missing_values function.")
        df = handle_missing_values(df, **kwargs)

    numeric_df = df.set_index(feature_col).T
    if type(metadata) == pd.DataFrame:
        groups = metadata.group.values
    else:
        groups = metadata

    # Handle cases where there are no numeric columns
    if numeric_df.empty:
        raise ValueError("The dataframe contains no numeric columns suitable for PCA.")

    std_devs = numeric_df.std()
    zero_std_cols = std_devs[std_devs == 0].index
    if not zero_std_cols.empty:
        print(f"Warning: The following columns have zero standard deviation and will be dropped: {list(zero_std_cols)}")
        numeric_df.drop(columns=zero_std_cols, inplace=True)

    # Standardizing the Data
    standardized_data = (numeric_df - numeric_df.mean()) / numeric_df.std()

    
    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(standardized_data)

    # Creating a dataframe with PCA results
    pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
    pca_df['group'] = groups
    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='group', palette='viridis')
    plt.title('PCA Plot (PCA1 vs PCA2)')
    plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]*100:.2f}% of variance)')
    plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]*100:.2f}% of variance)')
    plt.grid()

    # Display the plot
    plt.show()

    return pca_df


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
