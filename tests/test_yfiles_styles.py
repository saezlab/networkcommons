import unittest

from networkcommons.visual import (
    get_styles,
    get_comparison_color,
    get_edge_color,
    update_node_property,
    update_edge_property,
    apply_node_style,
    apply_edge_style,
)


class TestNetworkVisualization(unittest.TestCase):

    def setUp(self):
        # Set up any common objects or variables used in multiple tests
        self.styles = get_styles()
        self.node = {}
        self.edge = {}

    def test_get_styles(self):
        # Test if get_styles returns a dictionary with the expected keys
        styles = get_styles()
        self.assertIsInstance(styles, dict)
        self.assertIn('default', styles)
        self.assertIn('highlight', styles)
        self.assertIn('comparison', styles)
        self.assertIn('signs', styles)

    def test_apply_node_style(self):
        # Test if apply_node_style applies the correct style to a node
        style = self.styles['default']['nodes']
        node = apply_node_style(self.node, style)
        for key, value in style.items():
            self.assertEqual(node[key], value)

    def test_apply_edge_style(self):
        # Test if apply_edge_style applies the correct style to an edge
        style = self.styles['default']['edges']
        edge = apply_edge_style(self.edge, style)
        for key, value in style.items():
            self.assertEqual(edge[key], value)

    def test_update_node_property(self):
        # Test if update_node_property correctly updates a node property
        node = update_node_property(self.node, type="color", value="blue")
        self.assertEqual(node['color'], "blue")

    def test_update_edge_property(self):
        # Test if update_edge_property correctly updates an edge property
        edge = update_edge_property(self.edge, type="color", value="blue")
        self.assertEqual(edge['color'], "blue")

    def test_get_edge_color(self):
        # Test if get_edge_color returns the correct color based on the edge effect
        color = get_edge_color('activation', self.styles)
        self.assertEqual(color, 'green')
        color = get_edge_color('inhibition', self.styles)
        self.assertEqual(color, 'red')
        color = get_edge_color('undefined_effect', self.styles)
        self.assertEqual(color, 'black')  # Default color

    def test_get_comparison_color(self):
        # Test if get_comparison_color returns the correct color for a comparison category
        color = get_comparison_color('Unique to Network 1', self.styles, 'nodes')
        self.assertEqual(color, '#f5f536')
        color = get_comparison_color('Common', self.styles, 'edges')
        self.assertEqual(color, '#3643f5')
        color = get_comparison_color('Non-existent category', self.styles, 'nodes')
        self.assertEqual(color, '#cccccc')  # Default color


if __name__ == '__main__':
    unittest.main()
