import pandas as pd
from unittest.mock import patch, MagicMock
from networkcommons.data.network._moon import get_cosmos_pkn
from networkcommons.data.network._liana import get_lianaplus
from networkcommons.data.network._omnipath import get_omnipath, get_phosphositeplus
import os


def test_get_lianaplus():
    # Create a mock DataFrame to be returned by the mocked select_resource function
    mock_data = pd.DataFrame({
        'source': ['gene1', 'gene2', 'gene3'],
        'target': ['gene4', 'gene5', 'gene6']
    })

    with patch('liana.resource.select_resource', return_value=mock_data) as mock_select_resource:
        result = get_lianaplus('Consensus')

        # Check that the select_resource function was called with the correct argument
        mock_select_resource.assert_called_once_with('Consensus')

        # Check the result DataFrame
        expected_result = mock_data.copy()
        expected_result.columns = ['source', 'target']
        expected_result['sign'] = 1

        pd.testing.assert_frame_equal(result, expected_result)


def test_get_cosmos_pkn_file_exists():
    path = os.path.join('dummy_path', 'metapkn.pickle')
    mock_df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=True), \
         patch('pandas.read_pickle', return_value=mock_df) as mock_read_pickle:

        result = get_cosmos_pkn(update=False)

        mock_read_pickle.assert_called_once_with(path)
        pd.testing.assert_frame_equal(result, mock_df)


def test_get_cosmos_pkn_file_not_exists_or_update():
    path = os.path.join('dummy_path', 'metapkn.pickle')
    mock_df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=False), \
         patch('networkcommons.data.omics._common._baseurl', return_value='http://dummy_url'), \
         patch('pandas.read_csv', return_value=mock_df) as mock_read_csv, \
         patch('pandas.DataFrame.to_pickle') as mock_to_pickle:

        result = get_cosmos_pkn(update=False)

        mock_read_csv.assert_called_once_with('http://dummy_url/prior_knowledge/meta_network.sif', sep='\t')
        mock_to_pickle.assert_called_once_with(path)
        pd.testing.assert_frame_equal(result, mock_df)


def test_get_cosmos_pkn_update():
    path = os.path.join('dummy_path', 'metapkn.pickle')
    mock_df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=True), \
         patch('networkcommons.data.omics._common._baseurl', return_value='http://dummy_url'), \
         patch('pandas.read_csv', return_value=mock_df) as mock_read_csv, \
         patch('pandas.DataFrame.to_pickle') as mock_to_pickle:

        result = get_cosmos_pkn(update=True)

        mock_read_csv.assert_called_once_with('http://dummy_url/prior_knowledge/meta_network.sif', sep='\t')
        mock_to_pickle.assert_called_once_with(path)
        pd.testing.assert_frame_equal(result, mock_df)


def test_get_omnipath():
    mock_data = pd.DataFrame({
        'source': ['P12345', 'P23456'],
        'target': ['P34567', 'P45678'],
        'source_genesymbol': ['GeneA', 'GeneB'],
        'target_genesymbol': ['GeneC', 'GeneD'],
        'consensus_direction': [True, True],
        'consensus_stimulation': [True, False],
        'consensus_inhibition': [False, True],
        'curation_effort': [3, 2]
    })

    with patch('omnipath.interactions.AllInteractions.get', return_value=mock_data):
        result = get_omnipath(genesymbols=True, directed_signed=True)

        expected_result = pd.DataFrame({
            'source': ['GeneA', 'GeneB'],
            'target': ['GeneC', 'GeneD'],
            'sign': [1, -1]
        })

        pd.testing.assert_frame_equal(result, expected_result)


def test_get_omnipath_no_filter():
    mock_data = pd.DataFrame({
        'source': ['P12345', 'P23456'],
        'target': ['P34567', 'P45678'],
        'source_genesymbol': ['GeneA', 'GeneB'],
        'target_genesymbol': ['GeneC', 'GeneD'],
        'consensus_direction': [True, True],
        'consensus_stimulation': [True, False],
        'consensus_inhibition': [False, True],
        'curation_effort': [3, 2]
    })

    with patch('omnipath.interactions.AllInteractions.get', return_value=mock_data):
        result = get_omnipath(genesymbols=True, directed_signed=False)

        expected_result = pd.DataFrame({
            'source': ['GeneA', 'GeneB'],
            'target': ['GeneC', 'GeneD'],
            'sign': [1, -1]
        })

        pd.testing.assert_frame_equal(result, expected_result)


def test_get_phosphositeplus_file_exists():
    path = os.path.join('dummy_path', 'phosphositeplus.pickle')
    mock_df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=True), \
         patch('pandas.read_pickle', return_value=mock_df) as mock_read_pickle:

        result = get_phosphositeplus(update=False)

        mock_read_pickle.assert_called_once_with(path)
        pd.testing.assert_frame_equal(result, mock_df)


def test_get_phosphositeplus_file_not_exists_or_update():
    path = os.path.join('dummy_path', 'phosphositeplus.pickle')
    mock_df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=False), \
         patch('networkcommons.data.omics._common._baseurl', return_value='http://dummy_url'), \
         patch('pandas.read_csv', return_value=mock_df) as mock_read_csv, \
         patch('pandas.DataFrame.to_pickle') as mock_to_pickle:

        result = get_phosphositeplus(update=False)

        mock_read_csv.assert_called_once_with('http://dummy_url/prior_knowledge/kinase-substrate.tsv', sep='\t')
        mock_to_pickle.assert_called_once_with(path)
        pd.testing.assert_frame_equal(result, mock_df)


def test_get_phosphositeplus_update():
    path = os.path.join('dummy_path', 'phosphositeplus.pickle')
    mock_df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=True), \
         patch('networkcommons.data.omics._common._baseurl', return_value='http://dummy_url'), \
         patch('pandas.read_csv', return_value=mock_df) as mock_read_csv, \
         patch('pandas.DataFrame.to_pickle') as mock_to_pickle:

        result = get_phosphositeplus(update=True)

        mock_read_csv.assert_called_once_with('http://dummy_url/prior_knowledge/kinase-substrate.tsv', sep='\t')
        mock_to_pickle.assert_called_once_with(path)
        pd.testing.assert_frame_equal(result, mock_df)