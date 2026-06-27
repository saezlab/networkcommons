import pandas as pd
from unittest.mock import patch, MagicMock, call
from networkcommons.data.network._moon import get_cosmos_pkn, get_hmdb_mapper
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


def _make_resolve_response(hmdb_ids, names):
    """Build a minimal entities/resolve response for the given id→name pairs."""
    matches = []
    for hmdb_id, name in zip(hmdb_ids, names):
        matches.append({
            'identifier': hmdb_id,
            'candidates': [{
                'identifiers': [
                    {'identifier': name, 'identifierType': 'Iupac Traditional Name:OM:0211'},
                ]
            }]
        })
    return {'matches': matches}


def test_get_hmdb_mapper_file_exists():
    path = os.path.join('dummy_path', 'hmdb_mapper.pickle')
    mock_mapper = {'HMDB0000122': 'glucose', 'HMDB0000190': 'ethanol'}

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=True), \
         patch('pandas.read_pickle', return_value=mock_mapper) as mock_read_pickle:

        result = get_hmdb_mapper(update=False)

        mock_read_pickle.assert_called_once_with(path)
        assert result == mock_mapper


def test_get_hmdb_mapper_file_not_exists():
    path = os.path.join('dummy_path', 'hmdb_mapper.pickle')
    mock_pkn = pd.DataFrame({
        'source': ['Metab__HMDB0000122_c', 'GeneA'],
        'target': ['GeneB', 'Metab__HMDB0000190_r'],
        'sign': [1, -1],
    })
    resolve_response = _make_resolve_response(
        ['HMDB0000122', 'HMDB0000190'],
        ['glucose', 'ethanol'],
    )
    mock_client = MagicMock()
    mock_client._fetch.return_value = resolve_response

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=False), \
         patch('networkcommons.data.network._moon.get_cosmos_pkn', return_value=mock_pkn), \
         patch('omnipath_client.OmniPath', return_value=mock_client), \
         patch('pandas.to_pickle') as mock_to_pickle:

        result = get_hmdb_mapper(update=False)

        mock_client._fetch.assert_called_once()
        _, fetch_kwargs = mock_client._fetch.call_args
        assert fetch_kwargs['identifiers'] == sorted(['HMDB0000122', 'HMDB0000190']) or \
               set(fetch_kwargs['identifiers']) == {'HMDB0000122', 'HMDB0000190'}
        mock_to_pickle.assert_called_once_with({'HMDB0000122': 'glucose', 'HMDB0000190': 'ethanol'}, path)
        assert result == {'HMDB0000122': 'glucose', 'HMDB0000190': 'ethanol'}


def test_get_hmdb_mapper_update():
    path = os.path.join('dummy_path', 'hmdb_mapper.pickle')
    mock_pkn = pd.DataFrame({
        'source': ['Metab__HMDB0000122_c'],
        'target': ['GeneA'],
        'sign': [1],
    })
    resolve_response = _make_resolve_response(['HMDB0000122'], ['glucose'])
    mock_client = MagicMock()
    mock_client._fetch.return_value = resolve_response

    with patch('networkcommons._conf.get', return_value='dummy_path'), \
         patch('os.path.exists', return_value=True), \
         patch('networkcommons.data.network._moon.get_cosmos_pkn', return_value=mock_pkn), \
         patch('omnipath_client.OmniPath', return_value=mock_client), \
         patch('pandas.to_pickle') as mock_to_pickle:

        result = get_hmdb_mapper(update=True)

        mock_client._fetch.assert_called_once()
        assert result == {'HMDB0000122': 'glucose'}


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