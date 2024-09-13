import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set the matplotlib backend to 'Agg' for headless environments (like GitHub Actions)
matplotlib.use('Agg')

from networkcommons.visual._network_stats import (
    plot_rank,
    plot_scatter,
    lollipop_plot,
    plot_n_nodes_edges,
    plot_n_nodes_edges_from_df,
    build_heatmap_with_tree,
    create_heatmap
)

@pytest.fixture
def setup_data():
    # Sample data setup
    df = pd.DataFrame({
        'idx': ['Gene1', 'Gene2', 'Gene3'],
        'Value1': [10, 20, 15],
        'Value2': [12, 18, 14],
        'method': ['Method1', 'Method2', 'Method3'],
        'type': ['Type1', 'Type2', 'Type1'],
        'diff_dysregulation': [0.1, 0.2, -0.1],
        'coverage': [0.8, 0.9, 0.7],
        'nodes_with_phosphoinfo': [5, 10, 15]
    })

    networks = {
        'Network1': nx.DiGraph([(1, 2), (2, 3)]),
        'Network2': nx.DiGraph([(1, 2), (3, 4)])
    }

    metrics_df = pd.DataFrame({
        'Nodes': [3, 4],
        'Edges': [2, 2]
    }, index=['Network1', 'Network2'])

    jaccard_df = pd.DataFrame(
        np.array([[0.0, 0.2, 0.5],
                  [0.2, 0.0, 0.3],
                  [0.5, 0.3, 0.0]]),
        columns=['A', 'B', 'C'],
        index=['A', 'B', 'C']
    )

    ora_results = pd.DataFrame({
        'ora_Term': ['Term1', 'Term2', 'Term1', 'Term2'],
        'network': ['Net1', 'Net1', 'Net2', 'Net2'],
        'ora_rank': [1, 2, 3, 4]
    })
    ora_terms = ['Term1', 'Term2']

    return df, networks, metrics_df, jaccard_df, ora_results, ora_terms



@patch('networkcommons.visual._network_stats.plt.savefig')
@patch('networkcommons.visual._network_stats._log')
def test_plot_rank_logs_missing_column(mock_log, mock_savefig):
    # Simulate a DataFrame without the 'idx' column
    df = pd.DataFrame({
        'Gene1': [10, 20, 30],
        'Gene2': [15, 25, 35]
    })

    # Mock savefig return value to prevent actual file creation
    mock_savefig.return_value = MagicMock()

    # Call the plot_rank function with bio_ids and a filepath
    plot_rank(df, bio_ids=['Gene1'], filepath='test.png')

    # Check if _log was called with the correct message
    mock_log.assert_called_once_with("Column 'idx' not found in the DataFrame. Using the index as the ID column.")

    # Ensure that plt.savefig was called once with 'test.png'
    mock_savefig.assert_called_once_with('test.png')


@patch('networkcommons.visual._network_stats._log')
def test_plot_rank_no_output_warning(mock_log, setup_data):
    df, _, _, _, _, _ = setup_data
    plot_rank(df, filepath=None, render=False)

    # Check if _log was called with the correct warning message about no output being specified
    mock_log.assert_called_once_with("No output specified. Returning the plot object.")


@patch('networkcommons.visual._network_stats.plt.savefig')
def test_plot_rank_saves_figure(mock_savefig, setup_data):
    df, _, _, _, _, _ = setup_data
    filepath = 'test_rank_plot.png'
    plot_rank(df, bio_ids=['Gene1'], filepath=filepath)
    mock_savefig.assert_called_once_with(filepath)


@patch('networkcommons.visual._network_stats.plt.savefig')
def test_plot_scatter_saves_figure(mock_savefig, setup_data):
    df, _, _, _, _, _ = setup_data
    filepath = 'test_scatter_plot.png'
    plot_scatter(df, filepath=filepath)
    mock_savefig.assert_called_once_with(filepath)


@patch('matplotlib.figure.Figure.savefig')
def test_lollipop_plot_saves_figure(mock_savefig):
    data = {
        'Label': ['A', 'B', 'C', 'D'],
        'Value': [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)
    filepath = 'test_lollipop_plot.png'

    # Call the lollipop_plot function
    fig = lollipop_plot(
        df=df,
        label_col='Label',
        value_col='Value',
        filepath=filepath,
        render=False
    )

    # Assert that savefig was called on the figure object with the correct filepath
    mock_savefig.assert_called_once_with(filepath)


@patch('networkcommons.visual._network_stats.plt.savefig')
@patch('networkcommons.visual._network_stats._log')
def test_plot_rank_logs_missing_column(mock_log, mock_savefig, setup_data):
    # Setup data fixture
    df, _, _, _, _, _ = setup_data
    df = df.drop(columns=['idx'])  # Simulate missing 'idx' column

    # Call the function with a valid filepath
    plot_rank(df, bio_ids=['Gene1'], filepath='test.png')

    # Check that _log was called with the correct message
    mock_log.assert_called_once_with("Column 'idx' not found in the DataFrame. Using the index as the ID column.")

    # Ensure the savefig function is called with the correct filepath
    mock_savefig.assert_called_once_with('test.png')


@patch('networkcommons.visual._network_stats.lollipop_plot')
def test_plot_n_nodes_edges(mock_lollipop_plot, setup_data):
    _, networks, _, _, _, _ = setup_data
    filepath = 'test_nodes_edges_plot.png'

    # Call the plot_n_nodes_edges function
    plot_n_nodes_edges(networks, filepath=filepath)

    # Prepare expected DataFrame passed to lollipop_plot
    expected_df = pd.DataFrame({
        'Network': ['Network1', 'Network1', 'Network2', 'Network2'],
        'Category': ['Nodes', 'Edges', 'Nodes', 'Edges'],
        'Values': [3, 2, 4, 2]
    })

    # Get the actual DataFrame that was passed to lollipop_plot
    actual_df = mock_lollipop_plot.call_args[0][0]

    # Ensure both DataFrames have the same column order and reset index before comparing
    expected_df = expected_df[['Network', 'Category', 'Values']].reset_index(drop=True)
    actual_df_ordered = actual_df[['Network', 'Category', 'Values']].reset_index(drop=True)

    # Assert that the DataFrames are equal
    pd.testing.assert_frame_equal(actual_df_ordered, expected_df, check_like=True)


@patch('networkcommons.visual._network_stats.lollipop_plot')
def test_plot_n_nodes_edges_from_df(mock_lollipop_plot, setup_data):
    _, _, metrics_df, _, _, _ = setup_data
    filepath = 'test_nodes_edges_df_plot.png'

    # Call the function to plot nodes and edges from the DataFrame
    plot_n_nodes_edges_from_df(metrics_df, ['Nodes', 'Edges'], filepath=filepath)

    # Prepare expected DataFrame passed to lollipop_plot
    expected_df = pd.DataFrame({
        'Network': ['Network1', 'Network1', 'Network2', 'Network2'],
        'Category': ['Nodes', 'Edges', 'Nodes', 'Edges'],
        'Values': [3, 2, 4, 2]
    })

    # Get the actual DataFrame that was passed to lollipop_plot
    actual_df = mock_lollipop_plot.call_args[0][0]

    # Ensure both DataFrames have the same column order and reset index before comparing
    expected_df = expected_df[['Network', 'Category', 'Values']].reset_index(drop=True)
    actual_df_ordered = actual_df[['Network', 'Category', 'Values']].reset_index(drop=True)

    # Assert that the DataFrames are equal
    pd.testing.assert_frame_equal(actual_df_ordered, expected_df, check_like=True)


@patch('networkcommons.visual._network_stats.sns.clustermap')
def test_build_heatmap_with_tree(mock_clustermap, setup_data):
    _, _, _, jaccard_df, _, _ = setup_data
    mock_fig = MagicMock()
    mock_clustermap.return_value = MagicMock(fig=mock_fig)

    output_dir = "."
    filepath = f"{output_dir}/heatmap_with_tree.png"

    # Call the function with save=True
    build_heatmap_with_tree(jaccard_df, save=True, output_dir=output_dir)

    # Assert that savefig was called correctly
    mock_fig.savefig.assert_called_once_with(filepath, bbox_inches='tight')


@patch('networkcommons.visual._network_stats.sns.clustermap')
def test_build_heatmap_with_tree_render(mock_clustermap, setup_data):
    _, _, _, jaccard_df, _, _ = setup_data
    mock_fig = MagicMock()
    mock_clustermap.return_value = MagicMock(fig=mock_fig)

    # Call the function with render=True
    build_heatmap_with_tree(jaccard_df, render=True)

    # Assert that show was called correctly
    mock_fig.show.assert_called_once()


@patch('networkcommons.visual._network_stats.plt.savefig')
def test_create_rank_heatmap_saves_figure(mock_savefig, setup_data):
    _, _, _, _, ora_results, ora_terms = setup_data
    filepath = 'test_rank_heatmap.png'
    create_heatmap(ora_results, ora_terms, filepath=filepath)
    mock_savefig.assert_called_once_with(filepath)