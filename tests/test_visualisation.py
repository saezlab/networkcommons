import unittest
import networkx as nx
from networkcommons.visualize_network import {
    get_styles, 
    merge_styles, 
    set_style_attributes, 
    visualize_network_default, 
    visualize_network_sign_consistent, 
    visualize_network
}

class TestVisualizeNetwork(unittest.TestCase):

    def setUp(self):
        # Create a sample graph for testing
        self.G = nx.DiGraph()
        self.G.add_node("A")
        self.G.add_node("B")
        self.G.add_node("C")
        self.G.add_edge("A", "B")
        self.G.add_edge("B", "C")
        self.G.add_edge("C", "A")

        # Define source and target dictionaries
        self.source_dict = {"A": 1, "B": -1}
        self.target_dict = {"C": {"value": 1}}

    def test_get_styles(self):
        # Test that the styles dictionary contains the expected keys
        styles = get_styles()
        self.assertIn('default', styles)
        self.assertIn('sign_consistent', styles)
        self.assertIn('nodes', styles['default'])
        self.assertIn('edges', styles['default'])

    def test_merge_styles(self):
        # Test merging custom styles with default styles
        default_style = get_styles()['default']
        custom_style = {
            'nodes': {
                'sources': {
                    'color': 'red'
                }
            }
        }
        merged_style = merge_styles(default_style, custom_style)
        self.assertEqual(merged_style['nodes']['sources']['color'], 'red')
        self.assertEqual(merged_style['nodes']['sources']['shape'], 'circle')  # Default value

    def test_set_style_attributes(self):
        # Test setting style attributes on a node
        default_style = get_styles()['default']
        A = nx.nx_agraph.to_agraph(self.G)
        node = A.get_node("A")
        set_style_attributes(node, default_style['nodes']['sources'])
        self.assertEqual(node.attr['shape'], 'circle')
        self.assertEqual(node.attr['color'], 'steelblue')

    def test_visualize_network_default(self):
        # Test visualizing the network with default style
        A = visualize_network_default(self.G, self.source_dict, self.target_dict)
        self.assertTrue(A)  # Check that the function returns an AGraph object

    def test_visualize_network_sign_consistent(self):
        # Add interactions to edges for sign consistency
        self.G["A"]["B"]["interaction"] = 1
        self.G["B"]["C"]["interaction"] = -1
        self.G["C"]["A"]["interaction"] = 1
        # Test visualizing the network considering sign consistency
        A = visualize_network_sign_consistent(self.G, self.source_dict, self.target_dict)
        self.assertTrue(A)  # Check that the function returns an AGraph object

    def test_visualize_network_with_custom_style(self):
        # Define a custom style
        custom_style = {
            'nodes': {
                'sources': {
                    'shape': 'rectangle',
                    'color': 'red',
                    'style': 'filled',
                    'fillcolor': 'red',
                    'penwidth': 2
                }
            }
        }
        # Test visualizing the network with a custom style
        A = visualize_network(self.G, self.source_dict, self.target_dict, network_type='default', custom_style=custom_style)
        self.assertTrue(A)  # Check that the function returns an AGraph object

# if __name__ == '__main__':
#     unittest.main()
