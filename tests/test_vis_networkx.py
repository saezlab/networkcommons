import pytest
from unittest.mock import patch, MagicMock
import networkx as nx
from networkcommons.visual import NetworkXVisualizer, NetworkVisualizerBase
from networkcommons.visual import _styles

import matplotlib
# Set the matplotlib backend to 'Agg' for headless environments (like GitHub Actions)
matplotlib.use('Agg')


@pytest.fixture
def sample_network():
    # Create a simple directed graph with some nodes and edges
    G = nx.DiGraph()
    G.add_edge("1", "2", interaction=1)
    G.add_edge("2", "3", interaction=-1)
    G.add_edge("3", "4", interaction=-1)
    G.add_edge("4", "5", interaction=-1)
    return G


@pytest.fixture
def minimal_styles():
    # Return a simplified styles dictionary for testing
    return {
        'default': {
            'nodes': {
                'sources': {},
                'targets': {},
                'other': {'default': {}}
            },
            'edges': {
                'positive': {'color': 'green'},  # Positive edge color
                'negative': {'color': 'red'},    # Negative edge color
                'neutral': {'color': 'gray'},    # Neutral edge color
                'default': {'color': 'gray'}     # Default edge color
            }
        }
    }

@patch('networkcommons.visual._styles.get_styles')
def test_network_visualizer_base_init(mock_get_styles, sample_network):
    # Mock the default styles
    mock_get_styles.return_value = {
        'default': {
            'nodes': {
                'sources': {},
                'targets': {},
                'other': {'default': {}}
            },
            'edges': {
                'positive': {},
                'negative': {},
                'neutral': {},
                'default': {}
            }
        }
    }
    visualizer = NetworkVisualizerBase(sample_network)
    assert visualizer.network == sample_network
    assert visualizer.style == mock_get_styles()['default']


@patch('networkcommons.visual.NetworkVisualizerBase.set_custom_style')
def test_network_visualizer_base_set_custom_style(mock_set_custom_style, sample_network):
    visualizer = NetworkVisualizerBase(sample_network)
    custom_style = {'custom': 'style'}
    visualizer.set_custom_style(custom_style)
    mock_set_custom_style.assert_called_once_with(custom_style)


@patch('networkcommons.visual._styles.set_style_attributes')
@patch('networkcommons.visual._vis_networkx._log')
def test_network_visualizer_base_visualize_too_many_nodes(mock_log, mock_set_style_attributes, sample_network):
    visualizer = NetworkVisualizerBase(sample_network)
    visualizer.visualize_network_simple({}, {}, max_nodes=3)
    mock_log.assert_called_once_with(
        "The network is too large to visualize, you can increase the max_nodes parameter if needed.")
    assert mock_set_style_attributes.call_count == 0


@patch('networkcommons.visual._styles.set_style_attributes')
@patch('pygraphviz.AGraph.layout')
def test_network_visualizer_base_visualize_network_simple(mock_layout, mock_set_style_attributes, sample_network):
    visualizer = NetworkVisualizerBase(sample_network)
    source_dict = {str(1): 1}
    target_dict = {str(4): -1}
    graph = visualizer.visualize_network_simple(source_dict, target_dict)
    assert mock_set_style_attributes.call_count == len(graph.nodes()) + len(graph.edges())
    mock_layout.assert_called_once_with(prog='dot')


@patch('networkcommons.visual._styles.set_style_attributes')
@patch('pygraphviz.AGraph.layout')
def test_network_visualizer_sign_consistent(mock_layout, mock_set_style_attributes, sample_network):
    visualizer = NetworkXVisualizer(sample_network)
    source_dict = {str(1): 1}
    target_dict = {str(5): -1}

    # Call the sign-consistent visualization method
    visualizer.visualize_network_sign_consistent(source_dict, target_dict)

    # Assert that set_style_attributes was called for nodes and edges
    assert mock_set_style_attributes.call_count > 0

    # Assert that the layout function was called exactly once
    mock_layout.assert_called_once()


@patch('networkcommons.visual._vis_networkx._log')
@patch('networkcommons.visual._vis_networkx.plt.imshow')
@patch('networkcommons.visual._vis_networkx.plt.imread')  # Mock imread
@patch('networkcommons.visual._vis_networkx.plt.show')
def test_network_visualizer_render(mock_show, mock_imread, mock_imshow, mock_log, sample_network):
    # Initialize the visualizer
    visualizer = NetworkXVisualizer(sample_network)

    # Define source and target dictionaries
    source_dict = {str(1): 1}
    target_dict = {str(5): -1}

    # Mock the return value of plt.imread (since plt.imshow requires an image array)
    mock_img_data = MagicMock()
    mock_imread.return_value = mock_img_data  # Mocking imread to return mock image data

    # Call the visualize method with render=True
    visualizer.visualize(source_dict, target_dict, render=True)

    # Check if imshow was called with the mock image data
    mock_imshow.assert_called_once_with(mock_img_data)

    # Check if plt.show() was called to render the plot
    mock_show.assert_called_once()


@patch('networkcommons.visual._vis_networkx._log')
@patch('pygraphviz.AGraph.draw')
def test_network_visualizer_save_plot(mock_draw, mock_log, sample_network, ):
    visualizer = NetworkXVisualizer(sample_network)
    source_dict = {str(1): 1}
    target_dict = {str(5): -1}
    visualizer.visualize(source_dict, target_dict, output_file='test_plot.png')
    mock_draw.assert_called_once_with('test_plot.png', format='png', prog='dot')
    mock_log.assert_called_once_with("Network visualization saved to test_plot.png.")


@patch('networkcommons.visual._styles.set_style_attributes')
@patch('pygraphviz.AGraph.layout')
def test_network_visualizer_default_custom_style(mock_layout, mock_set_style_attributes, sample_network):
    visualizer = NetworkXVisualizer(sample_network)
    custom_style = {
        'nodes': {'sources': {'color': 'red'}, 'targets': {'color': 'green'}},
        'edges': {'neutral': {'color': 'blue'}}
    }
    graph = visualizer.visualize_network_default({1: 1}, {5: -1}, custom_style=custom_style)
    assert mock_set_style_attributes.call_count == len(graph.nodes()) + len(graph.edges())
    mock_layout.assert_called_once_with(prog='dot')


@patch('networkcommons.visual._styles.set_style_attributes')
def test_network_visualizer_edge_coloring(mock_set_style_attributes, sample_network):
    visualizer = NetworkXVisualizer(sample_network)
    for u, v in sample_network.edges():
        sample_network[u][v]['effect'] = 'positive'
    visualizer.color_edges()
    for u, v in sample_network.edges():
        assert sample_network[u][v]['color'] == visualizer.edge_colors['positive']['color']


def test_network_visualizer_empty_network():
    G = nx.DiGraph()
    visualizer = NetworkXVisualizer(G)
    result = visualizer.visualize_network_default({}, {})
    assert result is None

@patch('networkcommons.visual._styles.set_style_attributes')
@patch('pygraphviz.AGraph.layout')
def test_network_visualizer_layout_programs(mock_layout, mock_set_style_attributes, sample_network):
    visualizer = NetworkXVisualizer(sample_network)
    layout_programs = ['dot', 'neato', 'fdp', 'circo']
    for prog in layout_programs:
        visualizer.visualize(source_dict={str(1): 1}, target_dict={str(5): -1}, prog=prog)
        mock_layout.assert_called_with(prog=prog)
        mock_set_style_attributes.assert_called()


@patch('networkcommons.visual._styles.get_styles')
def test_network_visualizer_edge_interaction(mock_get_styles, sample_network, minimal_styles):
    # Mock the get_styles function to return the minimal styles
    mock_get_styles.return_value = minimal_styles

    # Initialize the visualizer with the sample network
    visualizer = NetworkXVisualizer(sample_network)

    # Visualize the network and return the AGraph object
    graph = visualizer.visualize_network_default({1: 1}, {5: -1}, prog='dot', custom_style=None)

    # Assert that the style (color) is set correctly for each edge based on the interaction
    for edge in graph.edges():
        u, v = int(edge[0]), int(edge[1])

        # Retrieve the edge color from the visualizer's AGraph object
        edge_color = graph.get_edge(u, v).attr['color']
        assert edge_color == 'gray', f"Edge {u}-{v} has unexpected color {edge_color}"
