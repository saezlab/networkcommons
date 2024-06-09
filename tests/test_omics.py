from networkcommons._data import _omics

import pandas as pd


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


def test_open_df():

    url = _omics._common._commons_url('test', table = 'meta')
    with _omics._common._open(url):

        line = next(fp)

    assert line.startswith('sample_ID\t')


def test_open_df():

    url = _omics._common._commons_url('test', table = 'meta')
    df = _omics._common._open(url, df = {'sep': '\t'})

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 2)
