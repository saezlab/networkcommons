import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import networkx as nx

from networkcommons.visual import (YFilesVisualizer)


class TestYFilesVisualizer(unittest.TestCase):

    def setUp(self):
        # Creating a simple graph for testing
        self.network = nx.DiGraph()
        self.network.add_node('Node1')
        self.network.add_node('Node2')
        self.network.add_edge('Node1', 'Node2', sign='activation')

        # Create a YFilesVisualizer instance
        self.visualizer = YFilesVisualizer(self.network)

    @patch('networkcommons.visual._vis_yfiles.yfiles.GraphWidget')
    def test_visualize(self, mock_graph_widget):
        # Test the visualize method
        mock_widget_instance = mock_graph_widget.return_value
        self.visualizer.visualize(graph_layout="Circular", directed=True)

        # Assert GraphWidget was called and nodes/edges were set correctly
        mock_graph_widget.assert_called_once()
        self.assertTrue(len(mock_widget_instance.nodes) > 0)
        self.assertTrue(len(mock_widget_instance.edges) > 0)
        mock_widget_instance.set_edge_color_mapping.assert_called_once()
        mock_widget_instance.set_node_styles_mapping.assert_called_once()
        mock_widget_instance.set_node_scale_factor_mapping.assert_called_once()
        mock_widget_instance.set_node_label_mapping.assert_called_once()
        self.assertTrue(mock_widget_instance.directed)

    @patch('networkcommons.visual._vis_yfiles.yfiles.GraphWidget')
    def test_vis_comparison(self, mock_graph_widget):
        # Create a mock comparison data
        node_comparison = pd.DataFrame({
            'node': ['Node1', 'Node2'],
            'comparison': ['Unique to Network 1', 'Unique to Network 2']
        })
        int_comparison = pd.DataFrame({
            'source': ['Node1'],
            'target': ['Node2'],
            'comparison': ['Common']
        })

        mock_widget_instance = mock_graph_widget.return_value
        self.visualizer.vis_comparison(int_comparison, node_comparison, graph_layout="Circular", directed=True)

        # Assert GraphWidget was called and nodes/edges were set correctly
        mock_graph_widget.assert_called_once()
        self.assertTrue(len(mock_widget_instance.nodes) > 0)
        self.assertTrue(len(mock_widget_instance.edges) > 0)
        mock_widget_instance.set_edge_color_mapping.assert_called_once()
        mock_widget_instance.set_node_styles_mapping.assert_called_once()
        mock_widget_instance.set_node_scale_factor_mapping.assert_called_once()
        mock_widget_instance.set_node_label_mapping.assert_called_once()
        self.assertTrue(mock_widget_instance.directed)

    def test_custom_edge_color_mapping(self):
        # Test the static method custom_edge_color_mapping
        edge = {"properties": {"color": "red"}}
        result = YFilesVisualizer.custom_edge_color_mapping(edge)
        self.assertEqual(result, "red")

    def test_custom_node_color_mapping(self):
        # Test the static method custom_node_color_mapping
        node = {"color": "blue"}
        result = YFilesVisualizer.custom_node_color_mapping(node)
        self.assertEqual(result, {"color": "blue"})

    def test_custom_factor_mapping(self):
        # Test the static method custom_factor_mapping
        node = {"color": "blue"}
        result = YFilesVisualizer.custom_factor_mapping(node)
        self.assertEqual(result, 1)

    def test_custom_label_styles_mapping(self):
        # Test the static method custom_label_styles_mapping
        node = {"properties": {"label": "TestLabel"}}
        result = YFilesVisualizer.custom_label_styles_mapping(node)
        expected_label_style = {
            'text': 'TestLabel',
            'backgroundColor': None,
            'fontSize': 12,
            'color': '#030200',
            'shape': 'round-rectangle',
            'textAlignment': 'center'
        }
        self.assertEqual(result, expected_label_style)


if __name__ == '__main__':
    unittest.main()
