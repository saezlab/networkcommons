import pytest

import pandas as pd

from networkcommons._data import omics


def test_datasets():

    dsets = omics.common._datasets()

    assert 'baseurl' in dsets
    assert isinstance(dsets['datasets'], dict)
    assert 'decryptm' in dsets['datasets']


def test_datasets_2():

    dsets = omics.common.datasets()

    assert 'decryptm' in dsets


def test_commons_url():

    url = omics.common._commons_url('test', table = 'meta')

    assert 'metadata' in url


@pytest.mark.slow
def test_download(tmp_path):

    url = omics.common._commons_url('test', table = 'meta')
    path = tmp_path / 'test_download.tsv'
    omics.common._download(url, path)

    assert path.exists()

    with open(path) as fp:

        line = next(fp)

    assert line.startswith('sample_ID\t')


@pytest.mark.slow
def test_open():

    url = omics.common._commons_url('test', table = 'meta')

    with omics.common._open(url) as fp:

        line = next(fp)

    assert line.startswith('sample_ID\t')


def test_open_df():

    url = omics.common._commons_url('test', table = 'meta')
    df = omics.common._open(url, df = {'sep': '\t'})

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 2)


@pytest.mark.slow
def test_decryptm_datasets():

    dsets = omics.decryptm.decryptm_datasets()

    assert isinstance(dsets, pd.DataFrame)
    assert dsets.shape == (51, 3)
    assert dsets.fname.str.contains('curves').all()


@pytest.fixture
def decryptm_args():

    return 'KDAC_Inhibitors', 'Acetylome', 'curves_CUDC101.txt'


@pytest.mark.slow
def test_decryptm_table(decryptm_args):

    df = omics.decryptm.decryptm_table(*decryptm_args)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (18007, 65)
    assert df.EC50.dtype == 'float64'


@pytest.mark.slow
def test_decryptm_experiment(decryptm_args):

    dfs = omics.decryptm.decryptm_experiment(*decryptm_args[:2])

    assert isinstance(dfs, list)
    assert len(dfs) == 4
    assert all(isinstance(df, pd.DataFrame) for df in dfs)
    assert dfs[3].shape == (15993, 65)
    assert dfs[3].EC50.dtype == 'float64'


@pytest.mark.slow
def test_panacea():

    dfs = omics.panacea()

    assert isinstance(dfs, tuple)
    assert len(dfs) == 2
    assert all(isinstance(df, pd.DataFrame) for df in dfs)
    assert dfs[0].shape == (24961, 1217)
    assert dfs[1].shape == (1216, 2)
    assert (dfs[0].drop('gene_symbol', axis = 1).dtypes == 'int64').all()
