import pandas as pd
from unittest.mock import patch, MagicMock
from networkcommons.data.network._moon import get_cosmos_pkn
from networkcommons.data.network._liana import get_lianaplus
from networkcommons.data.network._omnipath import get_omnipath, get_phosphositeplus
import os


def test_get_liana_pkn_file_exists():
    path = os.path.join('dummy_path', 'lianaplus.pickle')
    mock_df = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=True), \
         patch('pandas.read_pickle', return_value=mock_df) as mock_read_pickle:

        result = get_lianaplus(update=False)

        mock_read_pickle.assert_called_once_with(path)
        pd.testing.assert_frame_equal(result, mock_df)


def test_get_liana_pkn_file_not_exists_or_update():
    path = os.path.join('dummy_path', 'lianaplus.pickle')
    mock_df = pd.DataFrame({'source_genesymbol': ['a'], 'target_genesymbol': ['b'], 'resource': ['consensus']})
    expected_result = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=False), \
         patch('networkcommons.data.omics._common._baseurl', return_value='http://dummy_url'), \
         patch('pandas.read_csv', return_value=mock_df) as mock_read_csv, \
         patch('networkcommons.data.network._liana._log' ) as mock_log, \
         patch('pandas.DataFrame.to_pickle') as mock_to_pickle:

        result = get_lianaplus(resource='consensus', update=False)

        mock_read_csv.assert_called_with('http://dummy_url/prior_knowledge/liana_ligrec.csv', sep=',')
        mock_to_pickle.assert_called_once_with(path)
        pd.testing.assert_frame_equal(result, expected_result)

        result = get_lianaplus(resource='fakeresource', update=False)
        mock_log.assert_called_with('LIANA+: No data found for resource fakeresource')


def test_get_liana_pkn_update():
    path = os.path.join('dummy_path', 'lianaplus.pickle')
    mock_df = pd.DataFrame({'source_genesymbol': ['a'], 'target_genesymbol': ['b'], 'resource': ['consensus']})
    expected_result = pd.DataFrame({'source': ['a'], 'target': ['b'], 'sign': [1]})

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=True), \
         patch('networkcommons.data.omics._common._baseurl', return_value='http://dummy_url'), \
         patch('pandas.read_csv', return_value=mock_df) as mock_read_csv, \
         patch('pandas.DataFrame.to_pickle') as mock_to_pickle:

        result = get_lianaplus(update=True)

        mock_read_csv.assert_called_once_with('http://dummy_url/prior_knowledge/liana_ligrec.csv', sep=',')
        mock_to_pickle.assert_called_once_with(path)
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