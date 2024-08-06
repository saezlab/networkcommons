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
import os
import hashlib



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


def test_open_html():
    url = "http://example.com/test.html"
    with patch('networkcommons.data.omics._common._maybe_download', return_value='path/to/test.html'), \
         patch('builtins.open', mock_open(read_data="<html><body>Test</body></html>")):
        result = _common._open(url, ftype='html')
        assert isinstance(result, bs4.BeautifulSoup)
        assert result.body.text == "Test"


@patch('networkcommons.data.omics._common._download')
@patch('networkcommons.data.omics._common._log')
@patch('networkcommons.data.omics._common._conf.get')
@patch('os.path.exists')
@patch('hashlib.md5')
def test_maybe_download_exists(mock_md5, mock_exists, mock_conf_get, mock_log, mock_download):
    # Setup mock values
    url = 'http://example.com/file.txt'
    md5_hash = MagicMock()
    md5_hash.hexdigest.return_value = 'dummyhash'
    mock_md5.return_value = md5_hash
    mock_conf_get.return_value = '/mock/cache/dir'
    mock_exists.return_value = True

    # Call the function
    path = _common._maybe_download(url)

    # Assertions
    mock_md5.assert_called_once_with(url.encode())
    mock_conf_get.assert_called_once_with('cachedir')
    mock_exists.assert_called_once_with('/mock/cache/dir/dummyhash-file.txt')
    mock_log.assert_called_once_with('Looking up in cache: `http://example.com/file.txt` -> `/mock/cache/dir/dummyhash-file.txt`.')
    mock_download.assert_not_called()
    assert path == '/mock/cache/dir/dummyhash-file.txt'


@patch('networkcommons.data.omics._common._download')
@patch('networkcommons.data.omics._common._log')
@patch('networkcommons.data.omics._common._conf.get')
@patch('os.path.exists')
@patch('hashlib.md5')
def test_maybe_download_not_exists(mock_md5, mock_exists, mock_conf_get, mock_log, mock_download):
    # Setup mock values
    url = 'http://example.com/file.txt'
    md5_hash = MagicMock()
    md5_hash.hexdigest.return_value = 'dummyhash'
    mock_md5.return_value = md5_hash
    mock_conf_get.return_value = '/mock/cache/dir'
    mock_exists.return_value = False

    # Call the function
    path = _common._maybe_download(url)

    # Assertions
    mock_md5.assert_called_once_with(url.encode())
    mock_conf_get.assert_called_once_with('cachedir')
    mock_exists.assert_called_once_with('/mock/cache/dir/dummyhash-file.txt')
    mock_log.assert_any_call('Looking up in cache: `http://example.com/file.txt` -> `/mock/cache/dir/dummyhash-file.txt`.')
    mock_log.assert_any_call('Not found in cache, initiating download: `http://example.com/file.txt`.')
    mock_download.assert_called_once_with(url, '/mock/cache/dir/dummyhash-file.txt')
    assert path == '/mock/cache/dir/dummyhash-file.txt'


@patch('networkcommons.data.omics._common._requests_session')
@patch('networkcommons.data.omics._common._log')
@patch('networkcommons.data.omics._common._conf.get')
def test_download(mock_conf_get, mock_log, mock_requests_session, tmp_path):
    # Setup mock values
    url = 'http://example.com/file.txt'
    path = tmp_path / 'file.txt'
    timeouts = (5, 5)
    mock_conf_get.side_effect = lambda k: 5 if k in ('http_read_timout', 'http_connect_timout') else None
    mock_session = MagicMock()
    mock_requests_session.return_value = mock_session
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b'test content']
    mock_session.get.return_value.__enter__.return_value = mock_response

    # Call the function
    _common._download(url, str(path))

    # Assertions
    mock_conf_get.assert_any_call('http_read_timout')
    mock_conf_get.assert_any_call('http_connect_timout')
    mock_log.assert_any_call(f'Downloading `{url}` to `{path}`.')
    mock_log.assert_any_call(f'Finished downloading `{url}` to `{path}`.')
    mock_requests_session.assert_called_once()
    mock_session.get.assert_called_once_with(url, timeout=(5, 5), stream=True)
    mock_response.raise_for_status.assert_called_once()
    mock_response.iter_content.assert_called_once_with(chunk_size=8192)

    # Check that the file was written correctly
    with open(path, 'rb') as f:
        content = f.read()
    assert content == b'test content'


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


# FILE: omics/_decryptm.py
@pytest.fixture
def decryptm_args():
    return 'KDAC_Inhibitors', 'Acetylome', 'curves_CUDC101.txt'


@patch('networkcommons.data.omics._decryptm._common._ls')
@patch('networkcommons.data.omics._decryptm._common._baseurl', return_value='http://example.com')
@patch('pandas.read_pickle')
@patch('os.path.exists', return_value=False)
@patch('pandas.DataFrame.to_pickle')
def test_decryptm_datasets_update(mock_to_pickle, mock_path_exists, mock_read_pickle, mock_baseurl, mock_ls):
    # Mock the directory listing
    mock_ls.side_effect = [
        ['experiment1', 'experiment2'],  # First call, list experiments
        ['data_type1', 'data_type2'],  # Second call, list data types for experiment1
        ['curves_file1.txt', 'curves_file2.txt'],  # Third call, list files for experiment1/data_type1
        ['curves_file3.txt', 'curves_file4.txt'],  # Fourth call, list files for experiment1/data_type2
        ['data_type1', 'data_type2'],  # Fifth call, list data types for experiment2
        ['curves_file5.txt', 'curves_file6.txt'],  # Sixth call, list files for experiment2/data_type1
        ['curves_file7.txt', 'curves_file8.txt']   # Seventh call, list files for experiment2/data_type2
    ]

    dsets = omics.decryptm_datasets(update=True)

    assert isinstance(dsets, pd.DataFrame)
    assert dsets.shape == (8, 3)  # 4 experiments * 2 data types = 8 files
    assert dsets.columns.tolist() == ['experiment', 'data_type', 'fname']
    mock_to_pickle.assert_called_once()


@patch('pandas.read_pickle')
@patch('os.path.exists', return_value=True)
def test_decryptm_datasets_cached(mock_path_exists, mock_read_pickle):
    # Mock the cached DataFrame
    mock_df = pd.DataFrame({
        'experiment': ['experiment1', 'experiment2'],
        'data_type': ['data_type1', 'data_type2'],
        'fname': ['curves_file1.txt', 'curves_file2.txt']
    })
    mock_read_pickle.return_value = mock_df

    dsets = omics.decryptm_datasets(update=False)

    assert isinstance(dsets, pd.DataFrame)
    assert dsets.shape == (2, 3)
    assert dsets.columns.tolist() == ['experiment', 'data_type', 'fname']
    mock_read_pickle.assert_called_once()


@patch('networkcommons.data.omics._decryptm._common._open')
def test_decryptm_table(mock_open, decryptm_args):
    mock_df = pd.DataFrame({'EC50': [0.5, 1.0, 1.5]})
    mock_open.return_value = mock_df

    df = omics.decryptm_table(*decryptm_args)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 1)
    assert df.EC50.dtype == 'float64'
    mock_open.assert_called_once()


@patch('networkcommons.data.omics._decryptm.decryptm_datasets')
@patch('networkcommons.data.omics._decryptm.decryptm_table')
def test_decryptm_experiment(mock_decryptm_table, mock_decryptm_datasets, decryptm_args):
    mock_decryptm_datasets.return_value = pd.DataFrame({
        'experiment': ['KDAC_Inhibitors', 'KDAC_Inhibitors'],
        'data_type': ['Acetylome', 'Acetylome'],
        'fname': ['curves_CUDC101.txt', 'curves_other.txt']
    })
    mock_df = pd.DataFrame({'EC50': [0.5, 1.0, 1.5]})
    mock_decryptm_table.return_value = mock_df

    dfs = omics.decryptm_experiment(decryptm_args[0], decryptm_args[1])

    assert isinstance(dfs, list)
    assert len(dfs) == 2
    assert all(isinstance(df, pd.DataFrame) for df in dfs)
    assert dfs[0].shape == (3, 1)
    assert dfs[0].EC50.dtype == 'float64'
    mock_decryptm_table.assert_called()


@patch('networkcommons.data.omics._decryptm.decryptm_datasets')
def test_decryptm_experiment_no_dataset(mock_decryptm_datasets):
    mock_decryptm_datasets.return_value = pd.DataFrame({
        'experiment': ['KDAC_Inhibitors'],
        'data_type': ['Acetylome'],
        'fname': ['curves_CUDC101.txt']
    })

    with pytest.raises(ValueError, match='No such dataset in DecryptM: `Invalid_Experiment/Invalid_Type`.'):
        omics.decryptm_experiment('Invalid_Experiment', 'Invalid_Type')


# FILE: omics/_panacea.py

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


@patch('networkcommons.data.omics._cptac._conf.get')
@patch('os.path.exists', return_value=True)
@patch('pandas.read_pickle')
def test_cptac_cohortsize_cached(mock_read_pickle, mock_path_exists, mock_conf_get):
    # Mock configuration and data
    mock_conf_get.return_value = '/mock/path'
    mock_df = pd.DataFrame({
        "Cancer_type": ["BRCA", "CCRCC", "COAD", "GBM", "HNSCC", "LSCC", "LUAD", "OV", "PDAC", "UCEC"],
        "Tumor": [122, 103, 110, 99, 108, 108, 110, 83, 105, 95],
        "Normal": [0, 80, 100, 0, 62, 99, 101, 20, 44, 18]
    })
    mock_read_pickle.return_value = mock_df

    # Run the function with the condition that the pickle file exists
    result_df = omics.cptac_cohortsize()

    # Check that the result is as expected
    mock_read_pickle.assert_called_once_with('/mock/path/cptac_cohort.pickle')
    pd.testing.assert_frame_equal(result_df, mock_df)


@patch('networkcommons.data.omics._cptac._conf.get')
@patch('os.makedirs')  # Patch os.makedirs to prevent FileNotFoundError
@patch('os.path.exists', return_value=False)
@patch('pandas.read_excel')
@patch('pandas.DataFrame.to_pickle')
def test_cptac_cohortsize_download(mock_to_pickle, mock_read_excel, mock_makedirs, mock_conf_get, mock_path_exists):
    # Mock configuration and data
    mock_conf_get.return_value = '/mock/path'
    mock_df = pd.DataFrame({
        "Cancer_type": ["BRCA", "CCRCC", "COAD", "GBM", "HNSCC", "LSCC", "LUAD", "OV", "PDAC", "UCEC"],
        "Tumor": [122, 103, 110, 99, 108, 108, 110, 83, 105, 95],
        "Normal": [0, 80, 100, 0, 62, 99, 101, 20, 44, 18]
    })
    mock_read_excel.return_value = mock_df

    # Run the function with the condition that the pickle file does not exist
    result_df = omics.cptac_cohortsize(update=True)

    # Check that the result is as expected
    mock_read_excel.assert_called_once()
    mock_to_pickle.assert_called_once()
    pd.testing.assert_frame_equal(result_df, mock_df)


@patch('networkcommons.data.omics._cptac._conf.get')
@patch('os.path.exists', return_value=True)
@patch('pandas.read_pickle')
def test_cptac_fileinfo_cached(mock_read_pickle, mock_path_exists, mock_conf_get):
    # Mock configuration and data
    mock_conf_get.return_value = '/mock/path'
    mock_df = pd.DataFrame({
        "File name": ["file1.txt", "file2.txt"],
        "Description": ["Description1", "Description2"]
    })
    mock_read_pickle.return_value = mock_df

    # Run the function with the condition that the pickle file exists
    result_df = omics.cptac_fileinfo()

    # Check that the result is as expected
    mock_read_pickle.assert_called_once_with('/mock/path/cptac_info.pickle')
    pd.testing.assert_frame_equal(result_df, mock_df)


@patch('networkcommons.data.omics._cptac._conf.get')
@patch('os.makedirs')  # Patch os.makedirs to prevent FileNotFoundError
@patch('os.path.exists', return_value=False)
@patch('pandas.read_excel')
@patch('pandas.DataFrame.to_pickle')
def test_cptac_fileinfo_download(mock_to_pickle, mock_read_excel, mock_makedirs, mock_conf_get, mock_path_exists):
    # Mock configuration and data
    mock_conf_get.return_value = '/mock/path'
    mock_df = pd.DataFrame({
        "File name": ["file1.txt", "file2.txt"],
        "Description": ["Description1", "Description2"]
    })
    mock_read_excel.return_value = mock_df

    # Run the function with the condition that the pickle file does not exist
    result_df = omics.cptac_fileinfo(update=True)

    # Check that the result is as expected
    mock_read_excel.assert_called_once()
    mock_to_pickle.assert_called_once()
    pd.testing.assert_frame_equal(result_df, mock_df)


@patch('networkcommons.data.omics._cptac._common._ls')
@patch('networkcommons.data.omics._cptac._common._baseurl', return_value='http://example.com/')
def test_cptac_datatypes(mock_baseurl, mock_ls):
    # Mock the return value of _ls to simulate the directory listing
    mock_ls.return_value = [
        'directory1',
        'directory2',
        'CPTAC_pancancer_data_freeze_cohort_size.xlsx',
        'CPTAC_pancancer_data_freeze_file_description.xlsx'
    ]

    expected_directories = ['directory1', 'directory2']

    # Call the function
    directories = omics.cptac_datatypes()

    # Check if the returned directories match the expected directories
    assert directories == expected_directories


@patch('networkcommons.data.omics._common._open')
def test_cptac_table(mock_open):
    mock_df = pd.DataFrame({
        "sample_ID": ["sample1", "sample2"],
        "value": [123, 456]
    })
    mock_open.return_value = mock_df

    df = omics.cptac_table('proteomics', 'BRCA', 'file.tsv')

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    mock_open.assert_called_once_with(
        _common._commons_url('CPTAC', data_type='proteomics', cancer_type='BRCA', fname='file.tsv'),
        df={'sep': '\t'}
    )


def test_cptac_extend_dataframe():
    df = pd.DataFrame({
        "idx": ["sample1", "sample2", "sample3"],
        "Tumor": ["Yes", "No", "Yes"],
        "Normal": ["No", "Yes", "No"]
    })

    extended_df = omics.cptac_extend_dataframe(df)

    print(extended_df)

    expected_df = pd.DataFrame({
        "sample_ID": ["sample1_tumor", "sample3_tumor", "sample2_ctrl"]
    })

    pd.testing.assert_frame_equal(extended_df, expected_df)


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