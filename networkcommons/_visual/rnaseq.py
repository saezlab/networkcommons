import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List


def build_volcano_plot(
        data: pd.DataFrame,
        log2fc: str,
        pval: str,
        pval_threshold: float = 0.05,
        log2fc_threshold: float = 1,
        title: str = "Volcano Plot",
        xlabel: str = "log2 Fold Change",
        ylabel: str = "-log10(p-value)",
        colors: tuple = ("gray", "red"),
        alpha: float = 0.7,
        size: int = 50,
        save: bool = False,
        output_dir: str = "."
):
    data['-log10(pval)'] = -np.log10(data[pval])
    data['significant'] = (data[pval] < pval_threshold) & (abs(data[log2fc]) >= log2fc_threshold)

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
        data.loc[data['significant'], log2fc],
        data.loc[data['significant'], '-log10(pval)'],
        c=colors[1],
        alpha=alpha,
        s=size,
        label='Significant'
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
    pca = PCA(n_components=2)
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
        conditions: List[str] = None,
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
        conditions (List[str]): List of condition columns to include in the heatmap.
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


# Test the functions with the generated dataset
# Example condition columns for plotting
# np.random.seed(42)
#
# # Number of genes and conditions
# num_genes = 1000
# num_conditions = 4
#
# # Generate gene names
# genes = [f"gene_{i}" for i in range(num_genes)]
#
# # Generate mean expression values for each gene
# mean_expression = np.random.uniform(1, 100, num_genes)
#
# # Generate log2 fold changes and p-values for each condition
# log2_fold_changes = {
#     f'log2FoldChange_condition_{i + 1}': np.random.randn(num_genes) for i in range(num_conditions)
# }
# p_values = {
#     f'pvalue_condition_{i + 1}': np.random.uniform(0, 1, num_genes) for i in range(num_conditions)
# }
#
# # Combine data into a DataFrame
# data = pd.DataFrame({
#     'meanExpression': mean_expression,
#     **log2_fold_changes,
#     **p_values
# }, index=genes)
#
# # Display the first few rows of the generated data
# print(data.head())
#
# # Save the data to a CSV file for later use (optional)
# data.to_csv('simulated_rnaseq_data.csv')
#
# conditions = [f'log2FC_condition_{i + 1}' for i in range(num_conditions)]
#
# # Plot the Volcano plot for the first condition
# build_volcano_plot(
#     data=data,
#     log2fc='log2FoldChange_condition_1',
#     pval='pvalue_condition_1',
#     pval_threshold=0.05,
#     log2fc_threshold=1,
#     title="Sample Volcano Plot",
#     xlabel="log2 Fold Change",
#     ylabel="-log10(p-value)",
#     colors=("gray", "red"),
#     alpha=0.7,
#     size=20,
#     save=False,
#     output_dir="."
# )
#
# # Plot the MA plot for the first condition
# build_ma_plot(
#     data=data,
#     log2fc='log2FoldChange_condition_1',
#     mean_exp='meanExpression',
#     log2fc_threshold=1,
#     title="Sample MA Plot",
#     xlabel="Mean Expression",
#     ylabel="log2 Fold Change",
#     colors=("gray", "red"),
#     alpha=0.7,
#     size=20,
#     save=False,
#     output_dir="."
# )
#
# # Plot the PCA plot for all conditions
# pca_conditions = [f'log2FoldChange_condition_{i + 1}' for i in range(num_conditions)]
# build_pca_plot(
#     data=data[pca_conditions],
#     title="Sample PCA Plot",
#     xlabel="PC1",
#     ylabel="PC2",
#     alpha=0.7,
#     size=20,
#     save=False,
#     output_dir="."
# )
#
#
# # Plot the Heatmap with hierarchical clustering for the top differentially expressed genes across all conditions
# build_heatmap_with_tree(
#     data=data,
#     top_n=50,
#     value_column='log2FoldChange_condition_1',
#     conditions=pca_conditions,
#     title="Sample Heatmap of Top Differentially Expressed Genes",
#     save=False,
#     output_dir="."
# )