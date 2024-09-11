import pytest

import tempfile

import networkcommons as nc


@pytest.fixture(scope = 'session', autouse = True)
def tmp_cache():
    """
    Make sure the test suite is run with empty cache directories, and safely
    remove them on exit.
    """

    tempdirs = {
        key: tempfile.TemporaryDirectory()
        for key in ('cachedir', 'pickle_dir')
    }

    nc.config.setup(**{k: t.name for k, t in tempdirs.items()})

    yield

    _ = [t.cleanup() for t in tempdirs.values()]


@pytest.fixture(scope = 'function')
def nodes_list_dict():

    return [
        {'name': 'a', 'color': 'blue', 'size': 10},
        {'name': 'b', 'color': 'red', 'size': 17},
        {'name': 'c', 'color': 'green', 'size': 9},
    ]


@pytest.fixture(scope = 'function')
def nodes_list_dict_2():

    return [
        {'name': 'a', 'color': 'blue', 'size': 10},
        {'name': 'b', 'color': 'red', 'size': 17},
        {'name': 'c', 'color': 'green', 'size': 9},
        {'name': 'c', 'color': 'red'},
    ]


@pytest.fixture(scope = 'function')
def edges_list_dict():

    return [
        {'source': 'a', 'target': 'b', 'weight': 1.5, 'style': 'solid'},
        {'source': 'b', 'target': 'c', 'weight': 2.3, 'style': 'dashed'},
        {'source': 'a', 'target': 'c', 'weight': 0.9}
    ]
