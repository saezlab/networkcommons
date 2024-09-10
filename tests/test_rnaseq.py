import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import matplotlib.pyplot as plt
from networkcommons.visual import (plot_density,
                                   build_volcano_plot,
                                   build_ma_plot,
                                   plot_pca,
                                   plot_heatmap_with_tree)


@pytest.fixture
def example_dataframe():
    """Fixture for generating an example dataframe for testing."""
    data = {
        'idx': ['gene_1', 'gene_2', 'gene_3'],
        'sample_1': [10, 15, 5],
        'sample_2': [20, 18, 9],
        'sample_3': [12, 22, 8]
    }
    return pd.DataFrame(data)


@pytest.fixture
def metadata_dataframe():
    """Fixture for generating an example metadata dataframe."""
    metadata = {
        'sample_ID': ['sample_1', 'sample_2', 'sample_3'],
        'group': ['control', 'treated', 'control']
    }
    return pd.DataFrame(metadata)


def test_plot_density():
    """Test the plot_density function with valid data."""

    # Create a sample dataframe with enough data points
    example_dataframe = pd.DataFrame({
        'idx': ['gene_1', 'gene_2', 'gene_3'],
        'sample_1': [10, 15, 5],
        'sample_2': [20, 18, 9],
        'sample_3': [12, 22, 8],
        'sample_4': [14, 19, 7]  # Adding more samples to ensure enough data points
    })

    # Create metadata for grouping
    metadata_dataframe = pd.DataFrame({
        'sample_ID': ['sample_1', 'sample_2', 'sample_3', 'sample_4'],
        'group': ['control', 'treated', 'control', 'treated']
    })

    gene_ids = ['gene_1', 'gene_2']  # Make sure this has genes present in the dataframe

    # Mock plt.show to avoid blocking during the test
    with patch('matplotlib.pyplot.show'):
        plot_density(example_dataframe, gene_ids, metadata_dataframe)

    # Assert if the plot was created by checking the number of axes
    assert len(plt.gcf().get_axes()) == 2  # Should have 2 subplots for 2 genes

def test_build_volcano_plot():
    """Test the build_volcano_plot function."""
    data = pd.DataFrame({
        'log2FoldChange': [1.5, -2.0, 0.5, -0.3],
        'pvalue': [0.01, 0.04, 0.20, 0.05]
    })

    with patch('matplotlib.pyplot.show'):
        build_volcano_plot(data)

    # Assert if the plot was created
    assert len(plt.gcf().get_axes()) == 1  # Should have one main axis for the volcano plot


def test_build_ma_plot():
    """Test the build_ma_plot function."""
    data = pd.DataFrame({
        'log2FoldChange': [1.5, -2.0, 0.5, -0.3],
        'meanExpression': [10, 15, 20, 25]
    })

    with patch('matplotlib.pyplot.show'):
        build_ma_plot(data, log2fc='log2FoldChange', mean_exp='meanExpression')

    # Assert if the plot was created
    assert len(plt.gcf().get_axes()) == 1  # Should have one main axis for the MA plot


def test_plot_pca(example_dataframe, metadata_dataframe):
    """Test the plot_pca function."""
    with patch('matplotlib.pyplot.show'):
        pca_df = plot_pca(example_dataframe, metadata_dataframe)

    # Assert that the returned dataframe has the correct shape
    assert pca_df.shape[1] == 3  # Expecting PCA1, PCA2, and 'group' columns


def test_build_heatmap_with_tree():
    """Test the build_heatmap_with_tree function."""
    data = pd.DataFrame({
        'gene_1': [2.3, -1.1, 0.4],
        'gene_2': [1.2, 0.5, -0.7],
        'gene_3': [3.1, 0.9, -1.2]
    }, index=['condition_1', 'condition_2', 'condition_3'])

    with patch('matplotlib.pyplot.show'):
        fig = plot_heatmap_with_tree(
            data,
            clustering_method='ward',
            metric='euclidean',
            title='Test Heatmap',
            xlabel='Samples',
            ylabel='Genes',
            cmap='viridis',
            save=False,
            render=False
        )

    # Assert if the figure was created and contains an axes object
    assert isinstance(fig, plt.Figure)  # Check if the returned object is a matplotlib Figure
    assert len(fig.get_axes()) > 0  # Assert that axes were created in the figure