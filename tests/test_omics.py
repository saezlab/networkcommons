import pytest

import pandas as pd

from networkcommons._data import _omics


def test_datasets():

    dsets = _omics._common._datasets()

    assert 'baseurl' in dsets
    assert isinstance(dsets['datasets'], dict)
    assert 'decryptm' in dsets['datasets']


def test_datasets_2():

    dsets = _omics._common.datasets()

    assert 'decryptm' in dsets


def test_commons_url():

    url = _omics._common._commons_url('test', table = 'meta')

    assert 'metadata' in url


def test_download(tmp_path):

    url = _omics._common._commons_url('test', table = 'meta')
    path = tmp_path / 'test_download.tsv'
    _omics._common._download(url, path)

    assert path.exists()

    with open(path) as fp:

        line = next(fp)

    assert line.startswith('sample_ID\t')


def test_open():

    url = _omics._common._commons_url('test', table = 'meta')

    with _omics._common._open(url) as fp:

        line = next(fp)

    assert line.startswith('sample_ID\t')


def test_open_df():

    url = _omics._common._commons_url('test', table = 'meta')
    df = _omics._common._open(url, df = {'sep': '\t'})

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 2)


def test_decryptm_datasets():

    dsets = _omics._decryptm.decryptm_datasets()

    assert isinstance(dsets, pd.DataFrame)
    assert dsets.shape == (51, 3)
    assert dsets.fname.str.contains('curves').all()


@pytest.fixture
def decryptm_args():

    return 'KDAC_Inhibitors', 'Acetylome', 'curves_CUDC101.txt'


def test_decryptm(decryptm_args):

    df = _omics._decryptm.decryptm(*decryptm_args)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (18007, 65)
    assert df.EC50.dtype == 'float64'


def test_decryptm_experiment(decryptm_args):

    dfs = _omics._decryptm.decryptm_experiment(*decryptm_args[:2])

    assert isinstance(dfs, list)
    assert len(dfs) == 4
    assert all(isinstance(df, pd.DataFrame) for df in dfs)
    assert dfs[3].shape == (15993, 65)
    assert dfs[3].EC50.dtype == 'float64'
