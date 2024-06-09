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

    nc.config.setup(**tempdirs)

    yield

    _ = [t.cleanup() for t in tempdirs.values()]
