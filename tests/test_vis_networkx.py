#!/usr/bin/env python

import unittest
from unittest.mock import patch, MagicMock
import networkx as nx

from networkcommons.visual import NetworkXVisualizer, NetworkVisualizerBase

class TestVisualizeGraphSimple(unittest.TestCase):

    @patch('networkcommons.visual._vis_networkx.nx.nx_agraph.to_agraph')
    def test_visualize_graph_simple(self, mock_to_agraph):
        # Create a simple network graph for testing
        network = nx.DiGraph()
        network.add_node('Node1')
        network.add_node('Node2')
        network.add_edge('Node1', 'Node2', interaction=1)

        source_dict = {'Node1': 1}
        target_dict = {'Node2': 1}

        mock_agraph = mock_to_agraph.return_value
        mock_agraph.graph_attr.__getitem__.return_value = '1.2'  # Set return value for graph_attr

        # Call the function to visualize the graph
        nwb = NetworkVisualizerBase(network)
        nwb.visualize_network_simple(source_dict, target_dict, prog='dot', is_sign_consistent=True, max_nodes=75)

        # Check that the network was converted to an AGraph object
        mock_to_agraph.assert_called_once_with(network)

        # Check that the layout and attributes were set correctly
        mock_agraph.layout.assert_called_once_with(prog='dot')
        self.assertEqual(mock_agraph.graph_attr['ratio'], '1.2')  # Assertion

        for node in mock_agraph.nodes():
            self.assertIn(node.attr['shape'], ['circle'])
            self.assertIn(node.attr['style'], ['filled'])

        for edge in mock_agraph.edges():
            self.assertIn(edge.attr['color'], ['forestgreen', 'tomato3', 'gray30'])


class TestNetworkXVisualizer(unittest.TestCase):

    def setUp(self):
        # Set up a simple network graph for testing
        self.network = nx.DiGraph()
        self.network.add_node('Node1', type='source')
        self.network.add_node('Node2', type='target')
        self.network.add_edge('Node1', 'Node2', effect=1)

        self.visualizer = NetworkXVisualizer(self.network)

    @patch('networkcommons.visual._vis_networkx.nx.nx_agraph.to_agraph')
    def test_visualize_network_default(self, mock_to_agraph):
        mock_agraph = mock_to_agraph.return_value

        source_dict = {'Node1': 1}
        target_dict = {'Node2': 1}

        # Mocking the styles to avoid AttributeError on 'items'
        with patch('networkcommons.visual._styles.get_styles') as mock_get_styles:
            mock_get_styles.return_value = {
                'default': {
                    'nodes': {
                        'sources': {
                            'color': '#0000ff',
                            'fillcolor': '#ff0000',
                        },
                        'targets': {
                            'color': '#ff0000',
                            'fillcolor': '#ff00ff',
                        },
                        'other': {
                            'color': '#cccccc',
                            'fillcolor': '#cccccc',
                        },
                    },
                    'edges': {
                        'neutral': {
                            'color': '#999999',
                        }
                    }
                }
            }

            # Call the method to visualize the network
            self.visualizer.visualize_network_default(source_dict, target_dict, prog='dot')

            # Check that the network was converted to an AGraph object
            mock_to_agraph.assert_called_once_with(self.network)

            # Check that the layout was set correctly
            mock_agraph.layout.assert_called_once_with(prog='dot')

    def test_color_nodes(self):
        # Mocking the return value of _styles.get_styles
        with patch('networkcommons.visual._styles.get_styles') as mock_get_styles:
            mock_get_styles.return_value = {
                'default': {
                    'nodes': {
                        'other': {
                            'default': {
                                'fillcolor': '#00ff00'  # Default fill color
                            }
                        },
                        'sources': {
                            'fillcolor': '#0000ff'  # Source fill color
                        },
                        'targets': {
                            'fillcolor': '#ff0000'  # Target fill color
                        }
                    }
                }
            }

            # Set up the network with test nodes
            self.network.add_node('source_node', type='source')
            self.network.add_node('target_node', type='target')
            self.network.add_node('other_node')

            # Test the color_nodes method
            self.visualizer.color_nodes()

            # Assertions
            self.assertEqual(self.network.nodes['source_node']['color'], '#0000ff')
            self.assertEqual(self.network.nodes['target_node']['color'], '#ff0000')
            self.assertEqual(self.network.nodes['other_node']['color'], '#00ff00')

    def test_color_edges(self):
        # Mocking the return value of _styles.get_styles
        with patch('networkcommons.visual._styles.get_styles') as mock_get_styles:
            mock_get_styles.return_value = {
                'default': {
                    'edges': {
                        1: '#000000',
                        'default': '#999999'
                    }
                }
            }
            # Test the color_edges method
            self.visualizer.color_edges()

            for edge in self.network.edges:
                self.assertEqual(self.network.get_edge_data(*edge)['color'], '#000000')

    @patch('networkcommons.visual._vis_networkx.tempfile.NamedTemporaryFile')
    @patch('networkcommons.visual._vis_networkx.mpimg.imread')
    @patch('networkcommons.visual._vis_networkx.plt.imshow')
    @patch('networkcommons.visual._vis_networkx.plt.show')
    @patch('networkcommons.visual._vis_networkx.plt.savefig')
    @patch('networkcommons.visual._vis_networkx.plt.close')
    @patch('networkcommons.visual._styles.get_styles')
    def test_visualize(self, mock_get_styles, mock_close, mock_savefig, mock_show, mock_imshow, mock_imread,
                       mock_tempfile):
        mock_tempfile.return_value.__enter__.return_value.name = 'temp.png'
        mock_imread.return_value = MagicMock()

        # Mocking the return value of _styles.get_styles
        mock_get_styles.return_value = {
            'default': {
                'nodes': {
                    'other': {
                        'default': {
                            'fillcolor': '#ffffff'
                        }
                    },
                    'sources': {
                        'fillcolor': '#ff0000'
                    },
                    'targets': {
                        'fillcolor': '#00ff00'
                    }
                },
                'edges': {
                    'default': {
                        'color': '#000000'
                    },
                    'neutral': {
                        'color': '#999999'
                    }
                }
            }
        }

        # Mock the AGraph object and its draw method
        with patch('networkcommons.visual._vis_networkx.nx.nx_agraph.to_agraph') as mock_to_agraph:
            mock_agraph = mock_to_agraph.return_value

            # Call the visualize method with render=True
            self.visualizer.visualize(output_file='network.png', render=True)

            mock_agraph.draw.assert_called_once_with('temp.png', format='png', prog='dot')
            mock_imread.assert_called_once_with('temp.png')
            mock_imshow.assert_called_once()
            mock_show.assert_called_once()

            # Reset the mock draw call count
            mock_agraph.draw.reset_mock()

            # Call the visualize method with render=False (saving)
            self.visualizer.visualize(output_file='network.png', render=False)

            mock_agraph.draw.assert_called_once_with('network.png', format='png', prog='dot')

if __name__ == '__main__':
    unittest.main()