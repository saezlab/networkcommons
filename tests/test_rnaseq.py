import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import matplotlib
import matplotlib.pyplot as plt
from networkcommons.visual import (plot_density,
                                   build_volcano_plot,
                                   build_ma_plot,
                                   plot_pca,
                                   plot_heatmap_with_tree)

# Set the matplotlib backend to 'Agg' for headless environments (like GitHub Actions)
matplotlib.use('Agg')


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


### TESTING PLOT_DENSITY ###

def test_plot_density_valid_data():
    """Test plot_density with valid data."""
    example_dataframe = pd.DataFrame({
        'idx': ['gene_1', 'gene_2', 'gene_3'],
        'sample_1': [10, 15, 5],
        'sample_2': [20, 18, 9],
        'sample_3': [12, 22, 8],
        'sample_4': [14, 19, 7]
    })

    metadata_dataframe = pd.DataFrame({
        'sample_ID': ['sample_1', 'sample_2', 'sample_3', 'sample_4'],
        'group': ['control', 'treated', 'control', 'treated']
    })

    gene_ids = ['gene_1', 'gene_2']

    with patch('matplotlib.pyplot.show'):
        plot_density(example_dataframe, gene_ids, metadata_dataframe)

    assert len(plt.gcf().get_axes()) == 2


def test_plot_density_missing_data():
    """Test plot_density with missing values in the data."""
    example_dataframe = pd.DataFrame({
        'idx': ['gene_1', 'gene_2', 'gene_3'],
        'sample_1': [10, np.nan, 5],
        'sample_2': [20, 18, 9],
        'sample_3': [12, 22, np.nan],
        'sample_4': [14, 19, 7]
    })

    metadata_dataframe = pd.DataFrame({
        'sample_ID': ['sample_1', 'sample_2', 'sample_3', 'sample_4'],
        'group': ['control', 'treated', 'control', 'treated']
    })

    gene_ids = ['gene_1', 'gene_2']

    with patch('matplotlib.pyplot.show'):
        # Since missing data is present, we catch any errors that occur when plotting
        with pytest.raises(ValueError, match="`dataset` input should have multiple elements"):
            plot_density(example_dataframe, gene_ids, metadata_dataframe)

### TESTING BUILD_VOLCANO_PLOT ###

def test_build_volcano_plot_valid_data():
    """Test the build_volcano_plot function with valid data."""
    data = pd.DataFrame({
        'log2FoldChange': [1.5, -2.0, 0.5, -0.3],
        'pvalue': [0.01, 0.04, 0.20, 0.05]
    })

    with patch('matplotlib.pyplot.show'):
        build_volcano_plot(data)

    assert len(plt.gcf().get_axes()) == 1  # Should have one main axis for the volcano plot


def test_build_volcano_plot_empty_data():
    """Test build_volcano_plot with an empty dataframe."""
    data = pd.DataFrame({
        'log2FoldChange': [],
        'pvalue': []
    })

    with patch('matplotlib.pyplot.show'):
        build_volcano_plot(data)

    assert len(plt.gcf().get_axes()) == 1  # The plot should still exist even if empty


def test_build_volcano_plot_edge_cases():
    """Test build_volcano_plot with edge cases (e.g., very high/low p-values)."""
    data = pd.DataFrame({
        'log2FoldChange': [1.5, -2.0, 0.5, -0.3],
        'pvalue': [1e-300, 1e-10, 1, 0.9999]  # Extremely high and low p-values
    })

    with patch('matplotlib.pyplot.show'):
        build_volcano_plot(data)

    assert len(plt.gcf().get_axes()) == 1


### TESTING BUILD_MA_PLOT ###

def test_build_ma_plot_valid_data():
    """Test build_ma_plot with valid data."""
    data = pd.DataFrame({
        'log2FoldChange': [1.5, -2.0, 0.5, -0.3],
        'meanExpression': [10, 15, 20, 25]
    })

    with patch('matplotlib.pyplot.show'):
        build_ma_plot(data, log2fc='log2FoldChange', mean_exp='meanExpression')

    assert len(plt.gcf().get_axes()) == 1  # Should have one main axis for the MA plot


def test_build_ma_plot_empty_data():
    """Test build_ma_plot with an empty dataframe."""
    data = pd.DataFrame({
        'log2FoldChange': [],
        'meanExpression': []
    })

    with patch('matplotlib.pyplot.show'):
        build_ma_plot(data, log2fc='log2FoldChange', mean_exp='meanExpression')

    assert len(plt.gcf().get_axes()) == 1  # Plot should still exist


### TESTING PLOT_PCA ###

def test_plot_pca_valid_data(example_dataframe, metadata_dataframe):
    """Test the plot_pca function."""
    with patch('matplotlib.pyplot.show'):
        pca_df = plot_pca(example_dataframe, metadata_dataframe)

    assert pca_df.shape[1] == 3  # Expecting PCA1, PCA2, and 'group' columns


def test_plot_pca_empty_data():
    """Test plot_pca with empty dataframe."""
    empty_dataframe = pd.DataFrame({
        'idx': ['gene_1', 'gene_2', 'gene_3']
    })

    metadata = pd.DataFrame({
        'sample_ID': [],
        'group': []
    })

    with patch('matplotlib.pyplot.show'):
        with pytest.raises(ValueError, match="The dataframe contains no numeric columns suitable for PCA."):
            plot_pca(empty_dataframe, metadata)


def test_plot_pca_missing_data():
    """Test plot_pca with missing data."""
    example_dataframe = pd.DataFrame({
        'idx': ['gene_1', 'gene_2', 'gene_3'],
        'sample_1': [10, np.nan, 5],
        'sample_2': [20, 18, np.nan],
        'sample_3': [np.nan, 22, 8]
    })

    metadata = pd.DataFrame({
        'sample_ID': ['sample_1', 'sample_2', 'sample_3'],
        'group': ['control', 'treated', 'control']
    })

    with patch('matplotlib.pyplot.show'):
        with pytest.raises(ValueError, match="The dataframe contains no numeric columns suitable for PCA."):
            plot_pca(example_dataframe, metadata)


### TESTING PLOT_HEATMAP_WITH_TREE ###

def test_plot_heatmap_with_tree_valid_data():
    """Test the plot_heatmap_with_tree function with valid data."""
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

    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) > 0  # Check that axes exist in the figure


def test_plot_heatmap_with_tree_empty_data():
    """Test the plot_heatmap_with_tree function with empty dataframe."""
    empty_data = pd.DataFrame({
        'gene_1': [],
        'gene_2': [],
        'gene_3': []
    }, index=[])

    with patch('matplotlib.pyplot.show'):
        with pytest.raises(ValueError,
                           match="The number of observations cannot be determined on an empty distance matrix."):
            plot_heatmap_with_tree(
                empty_data,
                clustering_method='ward',
                metric='euclidean',
                title='Empty Heatmap',
                xlabel='Samples',
                ylabel='Genes',
                cmap='viridis',
                save=False,
                render=False
            )


def test_plot_heatmap_with_tree_missing_data():
    """Test the plot_heatmap_with_tree function with missing data (NaN values)."""
    data_with_nan = pd.DataFrame({
        'gene_1': [2.3, np.nan, 0.4],
        'gene_2': [1.2, 0.5, -0.7],
        'gene_3': [3.1, 0.9, np.nan]
    }, index=['condition_1', 'condition_2', 'condition_3'])

    with patch('matplotlib.pyplot.show'):
        with pytest.raises(ValueError, match="The condensed distance matrix must contain only finite values."):
            plot_heatmap_with_tree(
                data_with_nan,
                clustering_method='ward',
                metric='euclidean',
                title='Heatmap with Missing Data',
                xlabel='Samples',
                ylabel='Genes',
                cmap='viridis',
                save=False,
                render=False
            )