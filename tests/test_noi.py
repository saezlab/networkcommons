from networkcommons import _noi


SAMPLE_PROTEINS_HUMAN = {
    'EGFR': {
        'identifier': 'EGFR',
        'label': 'EGFR',
        'id_type': 'genesymbol',
        'organism': 9606,
        'entity_type': 'protein',
        'original_id': 'EGFR',
        'original_id_type': 'genesymbol',
    },
}


def test_noi_str():

    noi = _noi.Noi('EGFR')

    assert len(noi) == 1
    assert set(noi.keys()) == {'noi'}
    assert noi['noi'][0].asdict() == SAMPLE_PROTEINS_HUMAN['EGFR']