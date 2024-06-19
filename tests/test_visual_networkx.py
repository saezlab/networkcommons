import pytest

import networkx as nx

import networkcommons._visual.networkx as _vis


class TestVisualizeNetwork:


    @pytest.fixture
    def minigraph(self) -> tuple[nx.DiGraph, dict, dict]:
        # Create a sample graph for testing
        G = nx.DiGraph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('C')
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')

        source_dict = {'A': 1, 'B': -1}
        target_dict = {'C': {'value': 1}}

        return G, source_dict, target_dict


    def test_get_styles(self):
        # Test that the styles dictionary contains the expected keys
        styles = _vis.get_styles()

        assert 'default' in styles
        assert 'sign_consistent' in styles
        assert 'nodes' in styles['default']
        assert 'edges' in styles['default']


    def test_merge_styles(self):

        # Test merging custom styles with default styles
        default_style = _vis.get_styles()['default']
        custom_style = {
            'nodes': {
                'sources': {
                    'color': 'red'
                }
            }
        }
        merged_style = _vis.merge_styles(default_style, custom_style)

        assert merged_style['nodes']['sources']['color'] == 'red'
        assert merged_style['nodes']['sources']['shape'] == 'circle'  # Default value


    def test_set_style_attributes(self, minigraph):
        # Test setting style attributes on a node
        default_style = _vis.get_styles()['default']
        A = nx.nx_agraph.to_agraph(minigraph[0])
        node = A.get_node('A')
        _vis.set_style_attributes(node, default_style['nodes']['sources'])

        assert node.attr['shape'] == 'circle'
        assert node.attr['color'] == 'steelblue'


    def test_visualize_network_default(self, minigraph):

        G, source_dict, target_dict = minigraph
        # Test visualizing the network with default style
        A = _vis.visualize_network_default(G, source_dict, target_dict)

        assert A  # Check that the function returns an AGraph object


    def test_visualize_network_sign_consistent(self, minigraph):

        G, source_dict, target_dict = minigraph
        # Add interactions to edges for sign consistency
        G['A']['B']['interaction'] = 1
        G['B']['C']['interaction'] = -1
        G['C']['A']['interaction'] = 1
        # Test visualizing the network considering sign consistency
        A = _vis.visualize_network_sign_consistent(G, source_dict, target_dict)

        assert A  # Check that the function returns an AGraph object


    def test_visualize_network_with_custom_style(self, minigraph):

        G, source_dict, target_dict = minigraph
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
        A = _vis.visualize_network(
            G,
            source_dict,
            target_dict,
            network_type = 'default',
            custom_style = custom_style,
        )

        assert A  # Check that the function returns an AGraph object
