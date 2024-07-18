import pytest

import pandas as pd
import anndata as ad

from networkcommons.data.omics import _common
from networkcommons.data import omics


def test_datasets():

    dsets = _common._datasets()

    assert 'baseurl' in dsets
    assert isinstance(dsets['datasets'], dict)
    assert 'decryptm' in dsets['datasets']


def test_datasets_2():

    dsets = _common.datasets()

    assert 'decryptm' in dsets
    assert 'CPTAC' in dsets


def test_commons_url():

    url = _common._commons_url('test', table = 'meta')

    assert 'metadata' in url


@pytest.mark.slow
def test_download(tmp_path):

    url = _common._commons_url('test', table = 'meta')
    path = tmp_path / 'test_download.tsv'
    _common._download(url, path)

    assert path.exists()

    with open(path) as fp:

        line = next(fp)

    assert line.startswith('sample_ID\t')


@pytest.mark.slow
def test_open():

    url = _common._commons_url('test', table = 'meta')

    with _common._open(url) as fp:

        line = next(fp)

    assert line.startswith('sample_ID\t')


def test_open_df():

    url = _common._commons_url('test', table = 'meta')
    df = _common._open(url, df = {'sep': '\t'})

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 2)


@pytest.mark.slow
def test_decryptm_datasets():

    dsets = omics.decryptm_datasets()

    assert isinstance(dsets, pd.DataFrame)
    assert dsets.shape == (51, 3)
    assert dsets.fname.str.contains('curves').all()


@pytest.fixture
def decryptm_args():

    return 'KDAC_Inhibitors', 'Acetylome', 'curves_CUDC101.txt'


@pytest.mark.slow
def test_decryptm_table(decryptm_args):

    df = omics.decryptm_table(*decryptm_args)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (18007, 65)
    assert df.EC50.dtype == 'float64'


@pytest.mark.slow
def test_decryptm_experiment(decryptm_args):

    dfs = omics.decryptm_experiment(*decryptm_args[:2])

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


@pytest.mark.slow
def test_scperturb_metadata():

    m = omics.scperturb_metadata()

    assert isinstance(m, dict)
    assert len(m['files']['entries']) == 50
    assert m['versions'] == {'index': 4, 'is_latest': True}


@pytest.mark.slow
def test_scperturb_datasets():

    example_url = (
        'https://zenodo.org/api/records/10044268/files/'
        'XuCao2023.h5ad/content'
    )
    dsets = omics.scperturb_datasets()

    assert isinstance(dsets, dict)
    assert len(dsets) == 50
    assert dsets['XuCao2023.h5ad'] == example_url


@pytest.mark.slow
def test_scperturb():

    var_cols = ('ensembl_id', 'ncounts', 'ncells')
    adata = omics.scperturb('AdamsonWeissman2016_GSM2406675_10X001.h5ad')

    assert isinstance(adata, ad.AnnData)
    assert tuple(adata.var.columns) == var_cols
    assert 'UMI count' in adata.obs.columns
    assert adata.shape == (5768, 35635)


@pytest.mark.slow
def test_cptac_cohortsize():

    expected_df = pd.DataFrame({
        "Cancer_type": ["BRCA", "CCRCC", "COAD", "GBM", "HNSCC", "LSCC", "LUAD", "OV", "PDAC", "UCEC"],
        "Tumor": [122, 103, 110, 99, 108, 108, 110, 83, 105, 95],
        "Normal": [0, 80, 100, 0, 62, 99, 101, 20, 44, 18]
    })

    output_df = omics.cptac_cohortsize()

    assert output_df.equals(expected_df)


@pytest.mark.slow
def test_cptac_fileinfo():

    fileinfo_df = omics.cptac_fileinfo()

    assert isinstance(fileinfo_df, pd.DataFrame)
    assert fileinfo_df.shape == (37, 2)
    assert fileinfo_df.columns.tolist() == ['File name', 'Description']


@pytest.mark.slow
def test_cptac_table():

    df = omics.cptac_table('BRCA', 'meta')

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (123, 201)
