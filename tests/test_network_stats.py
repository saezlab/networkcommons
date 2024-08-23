import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from networkcommons.visual._network_stats import (
    plot_rank,
    plot_scatter,
    lollipop_plot,
    plot_n_nodes_edges,
    plot_n_nodes_edges_from_df,
    build_heatmap_with_tree,
    create_rank_heatmap
)


class TestPlotFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data setup
        self.df = pd.DataFrame({
            'idx': ['Gene1', 'Gene2', 'Gene3'],
            'Value1': [10, 20, 15],
            'Value2': [12, 18, 14],
            'method': ['Method1', 'Method2', 'Method3'],
            'type': ['Type1', 'Type2', 'Type1'],
            'diff_dysregulation': [0.1, 0.2, -0.1],
            'coverage': [0.8, 0.9, 0.7],
            'nodes_with_phosphoinfo': [5, 10, 15]
        })

        self.networks = {
            'Network1': nx.DiGraph([(1, 2), (2, 3)]),
            'Network2': nx.DiGraph([(1, 2), (3, 4)])
        }

        self.metrics_df = pd.DataFrame({
            'Nodes': [3, 4],
            'Edges': [2, 2]
        }, index=['Network1', 'Network2'])

        self.jaccard_df = pd.DataFrame(
            np.array([[0.0, 0.2, 0.5],
                      [0.2, 0.0, 0.3],
                      [0.5, 0.3, 0.0]]),
            columns=['A', 'B', 'C'],
            index=['A', 'B', 'C']
        )

        self.ora_results = pd.DataFrame({
            'ora_Term': ['Term1', 'Term2', 'Term1', 'Term2'],
            'network': ['Net1', 'Net1', 'Net2', 'Net2'],
            'ora_rank': [1, 2, 3, 4]
        })
        self.ora_terms = ['Term1', 'Term2']

    @patch('networkcommons.visual._network_stats.plt.savefig')
    def test_plot_rank(self, mock_savefig):
        filepath = 'test_rank_plot.png'
        plot_rank(self.df, bio_ids=['Gene1'], filepath=filepath)
        mock_savefig.assert_called_once_with(filepath)

    @patch('networkcommons.visual._network_stats.plt.savefig')
    def test_plot_scatter(self, mock_savefig):
        filepath = 'test_scatter_plot.png'
        plot_scatter(self.df, filepath=filepath)
        mock_savefig.assert_called_once_with(filepath)

    @patch('matplotlib.figure.Figure.savefig')
    def test_lollipop_plot_saves_figure(self, mock_savefig):
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

    @patch('networkcommons.visual._network_stats.lollipop_plot')
    def test_plot_n_nodes_edges(self, mock_lollipop_plot):
        filepath = 'test_nodes_edges_plot.png'

        # Call the plot_n_nodes_edges function
        plot_n_nodes_edges(self.networks, filepath=filepath)

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

        # Assert that lollipop_plot was called with the correct arguments
        mock_lollipop_plot.assert_called_once_with(
            actual_df,
            label_col='Network',
            value_col='Values',
            orientation='vertical',
            color_palette='Set2',
            size=10,
            linewidth=2,
            marker='o',
            title="Number of Nodes and Edges",
            filepath=filepath,
            render=False
        )

    @patch('networkcommons.visual._network_stats.lollipop_plot')
    def test_plot_n_nodes_edges_from_df(self, mock_lollipop_plot):
        filepath = 'test_nodes_edges_df_plot.png'

        # Call the function to plot nodes and edges from the DataFrame
        plot_n_nodes_edges_from_df(self.metrics_df, ['Nodes', 'Edges'], filepath=filepath)

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

        # Assert that lollipop_plot was called with the correct arguments
        mock_lollipop_plot.assert_called_once_with(
            # actual and expected DataFrames are the same - checked above
            actual_df,
            label_col='Network',
            value_col='Values',
            orientation='vertical',
            color_palette='Set2',
            size=10,
            linewidth=2,
            marker='o',
            title="Number of Nodes and Edges",
            filepath=filepath,
            render=False
        )

    @patch('networkcommons.visual._network_stats.sns.clustermap')
    def test_build_heatmap_with_tree(self, mock_clustermap):
        # Set up the mock return value for clustermap
        mock_fig = MagicMock()
        mock_clustermap.return_value = MagicMock(fig=mock_fig)

        output_dir = "."
        filepath = f"{output_dir}/heatmap_with_tree.png"

        # Call the function with save=True
        build_heatmap_with_tree(self.jaccard_df, save=True, output_dir=output_dir)

        # Assert that savefig was called correctly
        mock_fig.savefig.assert_called_once_with(filepath, bbox_inches='tight')

    @patch('networkcommons.visual._network_stats.sns.clustermap')
    def test_build_heatmap_with_tree_render(self, mock_clustermap):
        # Set up the mock return value for clustermap
        mock_fig = MagicMock()
        mock_clustermap.return_value = MagicMock(fig=mock_fig)

        # Call the function with render=True
        build_heatmap_with_tree(self.jaccard_df, render=True)

        # Assert that show was called correctly
        mock_fig.show.assert_called_once()

    @patch('networkcommons.visual._network_stats.plt.savefig')
    def test_create_rank_heatmap(self, mock_savefig):
        filepath = 'test_rank_heatmap.png'
        create_rank_heatmap(self.ora_results, self.ora_terms, filepath=filepath)
        mock_savefig.assert_called_once_with(filepath)

    @patch('networkcommons.visual._network_stats.logging.warning')
    @patch('networkcommons.visual._network_stats.lollipop_plot')
    def test_plot_n_nodes_edges_invalid_input(self,
                                              mock_lollipop_plot,
                                              mock_logging_warning):
        filepath = 'test_nodes_edges_plot.png'

        # Call the function with both show_nodes and show_edges set to False
        plot_n_nodes_edges(self.networks, filepath=filepath, show_nodes=False, show_edges=False)

        # Check that a warning was logged
        mock_logging_warning.assert_called_once_with(
            "Both 'show_nodes' and 'show_edges' are False. Using show nodes as default."
        )

        # Verify that the lollipop_plot was called with a DataFrame that contains the nodes data
        actual_df = mock_lollipop_plot.call_args[0][0]
        expected_df = pd.DataFrame({
            'Network': ['Network1', 'Network2'],  # Assuming 'self.networks' contains these two networks
            'Category': ['Nodes', 'Nodes'],  # Because 'show_nodes' was set to True by default
            'Values': [len(self.networks['Network1'].nodes), len(self.networks['Network2'].nodes)]
        })

        # Assert the DataFrame passed to lollipop_plot is correct
        pd.testing.assert_frame_equal(actual_df, expected_df, check_like=True)

        # Check that lollipop_plot was called with the correct parameters
        mock_lollipop_plot.assert_called_with(
            actual_df,
            label_col='Network',
            value_col='Values',
            orientation='vertical',
            color_palette='Set2',
            size=10,
            linewidth=2,
            marker='o',
            title="Number of Nodes",
            filepath=filepath,
            render=False
        )

    def test_plot_n_nodes_edges_from_df_invalid_input(self):
        with self.assertRaises(ValueError):
            plot_n_nodes_edges_from_df(self.metrics_df, [], render=False)


if __name__ == '__main__':
    unittest.main()
