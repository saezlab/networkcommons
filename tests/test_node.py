import pytest

from networkcommons._noi import _node


SAMPLE_HUMAN_PROTEINS = {
    'P00533': {
        'identifier': 'P00533',
        'label': 'EGFR',
        'id_type': 'uniprot',
        'organism': 9606,
        'entity_type': 'protein',
        'original_id': 'P00533',
        'original_id_type': 'uniprot',
    },
    'ULK1': {
        'identifier': 'ULK1',
        'label': 'ULK1',
        'id_type': 'genesymbol',
        'organism': 9606,
        'entity_type': 'protein',
        'original_id': 'ULK1',
        'original_id_type': 'genesymbol',
    },
    'ENSG00000109424': {
        'identifier': 'ENSG00000109424',
        'label': 'UCP1',
        'id_type': 'ensg',
        'organism': 9606,
        'entity_type': 'protein',
        'original_id': 'ENSG00000109424',
        'original_id_type': 'ensg',
    },
}


SAMPLE_MOUSE_PROTEINS = {
    'Egfr': {
        'entity_type': 'protein',
        'id_type': 'genesymbol',
        'identifier': 'Egfr',
        'label': 'Egfr',
        'organism': 10090,
        'original_id': 'Egfr',
        'original_id_type': 'genesymbol',
    },
}


@pytest.fixture
def human_protein_ids() -> list[str]:

    return list(SAMPLE_HUMAN_PROTEINS.keys())


def test_node_human_protein(human_protein_ids):

    for _id in human_protein_ids:

        node = _node.Node(_id)

        assert node.asdict() == SAMPLE_HUMAN_PROTEINS[_id]


def test_node_id_translation(human_protein_ids):

    upc1_ensg = human_protein_ids[2]
    upc1_uniprot_attrs = SAMPLE_HUMAN_PROTEINS[upc1_ensg].copy()
    upc1_uniprot_attrs['identifier'] = 'P25874'
    upc1_uniprot_attrs['id_type'] = 'uniprot'
    upc1 = _node.Node(human_protein_ids[2])

    assert list(upc1.as_idtype('uniprot'))[0].asdict() == upc1_uniprot_attrs


def test_node_mouse_protein():

    egfr_m = _node.Node('Egfr', organism = 'mouse')

    assert egfr_m.asdict() == SAMPLE_MOUSE_PROTEINS['Egfr']


def test_node_orthology_translation():

    egfr_m_o_attrs = SAMPLE_MOUSE_PROTEINS['Egfr'].copy()
    egfr_m_o_attrs['original_id'] = 'EGFR'
    egfr_m = _node.Node('Egfr', organism = 'mouse')
    egfr_h = _node.Node('EGFR', organism = 'human')
    egfr_m_o = list(egfr_h.as_organism('mouse'))[0]

    assert egfr_m_o.asdict() == egfr_m_o_attrs
