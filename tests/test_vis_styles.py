import pytest
from networkcommons.visual._styles import get_styles, set_style_attributes, merge_styles


class MockItem:
    """
    A simple mock class to simulate graph items (nodes or edges) for testing.
    """

    def __init__(self):
        self.attr = {}


def test_get_styles():
    styles = get_styles()

    # Check if the returned styles are as expected
    assert isinstance(styles, dict), "Styles should be a dictionary."
    assert 'default' in styles, "'default' should be a key in styles."
    assert 'sign_consistent' in styles, "'sign_consistent' should be a key in styles."

    # Check structure of a 'nodes' style in 'default'
    assert 'nodes' in styles['default'], "'nodes' should be a key in 'default' style."
    assert 'sources' in styles['default']['nodes'], "'sources' should be a key in 'default.nodes' style."
    assert 'color' in styles['default']['nodes']['sources'], "'color' should be a key in 'default.nodes.sources'."
    assert styles['default']['nodes']['sources']['color'] == '#62a0ca', "Unexpected color for 'default.nodes.sources'."

    # Check structure of 'edges' style in 'default'
    assert 'edges' in styles['default'], "'edges' should be a key in 'default' style."
    assert 'positive' in styles['default']['edges'], "'positive' should be a key in 'default.edges'."
    assert styles['default']['edges']['positive'][
               'color'] == '#70bc6b', "Unexpected color for 'default.edges.positive'."

    # Check structure of 'sign_consistent' style
    assert 'nodes' in styles['sign_consistent'], "'nodes' should be a key in 'sign_consistent' style."
    assert 'sources' in styles['sign_consistent']['nodes'], "'sources' should be a key in 'sign_consistent.nodes'."
    assert 'color' in styles['sign_consistent']['nodes']['sources']['default'], "'color' should be a key in 'sign_consistent.nodes.sources.default'."
    assert styles['sign_consistent']['nodes']['sources']['default']['color'] == '#62a0ca', "Unexpected color for 'sign_consistent.nodes.sources.default.color'."

    # Check structure of 'edges' style in 'sign_consistent'
    assert 'edges' in styles['sign_consistent'], "'edges' should be a key in 'sign_consistent' style."
    assert 'positive' in styles['sign_consistent']['edges'], "'positive' should be a key in 'sign_consistent.edges'."
    assert styles['sign_consistent']['edges']['positive'][
               'color'] == '#33a02c', "Unexpected color for 'sign_consistent.edges.positive'."


def test_set_style_attributes():
    item = MockItem()
    base_style = {'color': 'blue', 'shape': 'circle'}
    condition_style = {'color': 'red'}

    # Apply base and condition styles to the item
    result = set_style_attributes(item, base_style, condition_style)

    # Check if the attributes were set correctly
    assert result.attr['color'] == 'red', "Condition style should override base style for 'color'."
    assert result.attr['shape'] == 'circle', "Base style 'shape' should be applied."

    # Test when condition_style is None
    item2 = MockItem()
    result2 = set_style_attributes(item2, base_style)

    assert result2.attr['color'] == 'blue', "Base style 'color' should be applied when no condition style."
    assert result2.attr['shape'] == 'circle', "Base style 'shape' should be applied when no condition style."


def test_merge_styles():
    default_style = {
        'nodes': {
            'color': 'blue',
            'shape': 'circle'
        },
        'edges': {
            'color': 'gray',
            'penwidth': 2
        }
    }

    custom_style = {
        'nodes': {
            'color': 'red'
        },
        'edges': {
            'penwidth': 3
        }
    }

    merged = merge_styles(default_style, custom_style)

    # Check if the merged style correctly reflects the custom style
    assert merged['nodes']['color'] == 'red', "Custom style should override default 'color' for 'nodes'."
    assert merged['nodes']['shape'] == 'circle', "'shape' should remain from default style."
    assert merged['edges']['color'] == 'gray', "'color' for 'edges' should remain from default style."
    assert merged['edges']['penwidth'] == 3, "Custom style should override 'penwidth' for 'edges'."

    # Test missing keys handling (logging is ignored in this test)
    custom_style_missing = {
        'nodes': {
            # Missing 'shape'
            'color': 'red'
        }
    }

    merged_missing = merge_styles(default_style, custom_style_missing)

    assert merged_missing['nodes'][
               'shape'] == 'circle', "'shape' should remain from default style when missing in custom."
    assert merged_missing['nodes']['color'] == 'red', "Custom 'color' should still override the default."