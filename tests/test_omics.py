import pytest

import pandas as pd
import anndata as ad

from networkcommons.data.omics import _common
from networkcommons.data import omics

from unittest.mock import patch, MagicMock, mock_open
import zipfile
import bs4
import requests

import responses



def test_datasets():

    dsets = _common._datasets()

    assert 'baseurl' in dsets
    assert isinstance(dsets['datasets'], dict)
    assert 'decryptm' in dsets['datasets']


def test_datasets_2():

    dsets = _common.datasets()

    assert isinstance(dsets, pd.DataFrame)
    assert dsets.columns.tolist() == ['name', 'description', 'publication_link', 'detailed_description']
    assert 'decryptm' in dsets.index
    assert 'CPTAC' in dsets.index


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


@pytest.mark.slow
def test_open_df():

    url = _common._commons_url('test', table = 'meta')
    df = _common._open(url, df = {'sep': '\t'})

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 2)


def test_open_tsv():
    url = "http://example.com/test.tsv"
    with patch('networkcommons.data.omics._common._maybe_download', return_value='path/to/test.tsv'), \
         patch('builtins.open', mock_open(read_data="col1\tcol2\nval1\tval2")):
        with _common._open(url, ftype='tsv') as f:
            content = f.read()
            assert "col1\tcol2\nval1\tval2" in content


@patch('networkcommons.data.omics._common._maybe_download')
@patch('zipfile.ZipFile', autospec=True)
def test_open_zip(mock_maybe_download, mock_zipfile):
    url = 'http://example.com/file.zip'
    mock_maybe_download.return_value = 'file.zip'
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value = mock_zip_instance
    mock_zip_instance.__enter__.return_value = mock_zip_instance
    mock_zip_instance.__exit__.return_value = False
    with _common._open(url, 'zip') as f:
        assert f is mock_zip_instance
    mock_zipfile.assert_called_once_with('file.zip', 'r')


def test_open_html():
    url = "http://example.com/test.html"
    with patch('networkcommons.data.omics._common._maybe_download', return_value='path/to/test.html'), \
         patch('builtins.open', mock_open(read_data="<html><body>Test</body></html>")):
        result = _common._open(url, ftype='html')
        assert isinstance(result, bs4.BeautifulSoup)
        assert result.body.text == "Test"


def test_ls_success():
    url = "http://example.com/dir/"
    html_content = '''
    <html>
        <body>
            <a href="file1.txt">file1.txt</a>
            <a href="file2.txt">file2.txt</a>
            <a href="..">parent</a>
        </body>
    </html>
    '''

    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, body=html_content, status=200)
        result = _common._ls(url)
        assert result == ["file1.txt", "file2.txt"]


def test_ls_not_found():
    url = "http://example.com/dir/"

    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, status=404)
        with pytest.raises(FileNotFoundError, match="URL http://example.com/dir/ returned status code 404"):
            _common._ls(url)


@patch('networkcommons.data.omics._common._maybe_download')
def test_open_unknown_file_type(mock_maybe_download):
    url = 'http://example.com/file.unknown'
    mock_maybe_download.return_value = 'file.unknown'
    with pytest.raises(NotImplementedError, match='Can not open file type `unknown`.'):
        _common._open(url, 'unknown')


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


@patch('networkcommons.data.omics._common._conf.get')
@patch('pandas.read_pickle')
@patch('os.path.exists', return_value=True)
def test_get_ensembl_mappings_cached(mock_path_exists, mock_read_pickle, mock_conf_get):
    # Mock configuration and data
    mock_conf_get.return_value = '/path/to/pickle/dir'
    mock_df = pd.DataFrame({
        'gene_symbol': ['BRCA2', 'BRCA1'],
        'ensembl_id': ['ENSG00000139618', 'ENSG00000012048']
    })
    mock_read_pickle.return_value = mock_df

    # Run the function with the condition that the pickle file exists
    result_df = _common.get_ensembl_mappings()

    # Check that the result is as expected
    mock_read_pickle.assert_called_once_with('/path/to/pickle/dir/ensembl_map.pickle')


@patch('networkcommons.data.omics._common._conf.get')
@patch('os.path.exists', return_value=False)
@patch('biomart.BiomartServer')
def test_get_ensembl_mappings_download(mock_biomart_server, mock_path_exists, mock_conf_get):
    # Mock configuration and data
    mock_conf_get.return_value = '/path/to/pickle/dir'

    # Mock the biomart server and dataset
    mock_server_instance = MagicMock()
    mock_biomart_server.return_value = mock_server_instance
    mock_dataset = mock_server_instance.datasets['hsapiens_gene_ensembl']
    mock_response = MagicMock()
    mock_dataset.search.return_value = mock_response
    mock_response.raw.data.decode.return_value = (
        'ENST00000361390\tBRCA2\tENSG00000139618\tENSP00000354687\n'
        'ENST00000361453\tBRCA2\tENSG00000139618\tENSP00000354687\n'
        'ENST00000361453\tBRCA1\tENSG00000012048\tENSP00000354688\n'
    )

    with patch('pandas.DataFrame.to_pickle') as mock_to_pickle:
        result_df = _common.get_ensembl_mappings()

        expected_data = {
            'gene_symbol': ['BRCA2', 'BRCA2', 'BRCA1', 'BRCA2', 'BRCA1', 'BRCA2', 'BRCA1'],
            'ensembl_id': ['ENST00000361390', 'ENST00000361453', 'ENST00000361453',
                        'ENSG00000139618', 'ENSG00000012048', 'ENSP00000354687',
                        'ENSP00000354688']
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)
        mock_to_pickle.assert_called_once_with('/path/to/pickle/dir/ensembl_map.pickle')



def test_convert_ensembl_to_gene_symbol_max():
    dataframe = pd.DataFrame({
        'idx': ['ENSG000001.23', 'ENSG000002', 'ENSG000001.19'],
        'value': [10, 20, 15]
    })
    equivalence_df = pd.DataFrame({
        'ensembl_id': ['ENSG000001', 'ENSG000002'],
        'gene_symbol': ['GeneA', 'GeneB']
    })
    result_df = omics.convert_ensembl_to_gene_symbol(dataframe, equivalence_df, summarisation='max')
    expected_df = pd.DataFrame({
        'gene_symbol': ['GeneA', 'GeneB'],
        'value': [15, 20]
    })
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_convert_ensembl_to_gene_symbol_min():
    dataframe = pd.DataFrame({
        'idx': ['ENSG000001.28', 'ENSG000002', 'ENSG000001.23'],
        'value': [10, 20, 15]
    })
    equivalence_df = pd.DataFrame({
        'ensembl_id': ['ENSG000001', 'ENSG000002'],
        'gene_symbol': ['GeneA', 'GeneB']
    })
    result_df = omics.convert_ensembl_to_gene_symbol(dataframe, equivalence_df, summarisation='min')
    expected_df = pd.DataFrame({
        'gene_symbol': ['GeneA', 'GeneB'],
        'value': [10, 20]
    })
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_convert_ensembl_to_gene_symbol_mean():
    dataframe = pd.DataFrame({
        'idx': ['ENSG000001.29', 'ENSG000002', 'ENSG000001.48'],
        'value': [10, 20, 15]
    })
    equivalence_df = pd.DataFrame({
        'ensembl_id': ['ENSG000001', 'ENSG000002'],
        'gene_symbol': ['GeneA', 'GeneB']
    })
    result_df = omics.convert_ensembl_to_gene_symbol(dataframe, equivalence_df, summarisation='mean')
    expected_df = pd.DataFrame({
        'gene_symbol': ['GeneA', 'GeneB'],
        'value': [12.5, 20]
    })
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_convert_ensembl_to_gene_symbol_median():
    dataframe = pd.DataFrame({
        'idx': ['ENSG000001.10', 'ENSG000002', 'ENSG000001.2'],
        'value': [10, 20, 15]
    })
    equivalence_df = pd.DataFrame({
        'ensembl_id': ['ENSG000001', 'ENSG000002'],
        'gene_symbol': ['GeneA', 'GeneB']
    })
    result_df = omics.convert_ensembl_to_gene_symbol(dataframe, equivalence_df, summarisation='median')
    expected_df = pd.DataFrame({
        'gene_symbol': ['GeneA', 'GeneB'],
        'value': [12.5, 20]
    })
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_convert_ensembl_to_gene_symbol_no_match():
    dataframe = pd.DataFrame({
        'idx': ['ENSG000001.1', 'ENSG000003', 'ENSG000001.02'],
        'value': [10, 20, 15]
    })
    equivalence_df = pd.DataFrame({
        'ensembl_id': ['ENSG000001', 'ENSG000002'],
        'gene_symbol': ['GeneA', 'GeneB']
    })
    with patch('builtins.print') as mocked_print:
        result_df = omics.convert_ensembl_to_gene_symbol(dataframe, equivalence_df, summarisation='mean')
        print(mocked_print.mock_calls)
        expected_df = pd.DataFrame({
            'gene_symbol': ['GeneA'],
            'value': [12.5]
        })
        pd.testing.assert_frame_equal(result_df, expected_df)
        mocked_print.assert_any_call("Number of non-matched Ensembl IDs: 1 (33.33%)")


@patch('biomart.BiomartServer')
def test_get_ensembl_mappings(mock_biomart_server):
    # Mock the biomart server and dataset
    mock_server_instance = MagicMock()
    mock_biomart_server.return_value = mock_server_instance
    mock_dataset = mock_server_instance.datasets['hsapiens_gene_ensembl']
    mock_response = MagicMock()
    mock_dataset.search.return_value = mock_response
    mock_response.raw.data.decode.return_value = (
        'ENST00000361390\tBRCA2\tENSG00000139618\tENSP00000354687\n'
        'ENST00000361453\tBRCA2\tENSG00000139618\tENSP00000354687\n'
        'ENST00000361453\tBRCA1\tENSG00000012048\tENSP00000354688\n'
    )

    result_df = omics.get_ensembl_mappings()
    print(result_df)

    expected_data = {
        'gene_symbol': ['BRCA2', 'BRCA2', 'BRCA1', 'BRCA2', 'BRCA1', 'BRCA2', 'BRCA1'],
        'ensembl_id': ['ENST00000361390', 'ENST00000361453', 'ENST00000361453',
                       'ENSG00000139618', 'ENSG00000012048', 'ENSP00000354687',
                       'ENSP00000354688']
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)