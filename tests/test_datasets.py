#!/usr/bin/env python

#
# This file is part of the `networkcommons` Python module
#
# Copyright 2024
# Heidelberg University Hospital
#
# File author(s): Saez Lab (omnipathdb@gmail.com)
#
# Distributed under the GPLv3 license
# See the file `LICENSE` or read a copy at
# https://www.gnu.org/licenses/gpl-3.0.txt
#

"""
Tests for the dataset layer and its first dataset, PANACEA.

The tests run offline: artifacts are seeded into a temporary cache and served
by Pooch from there.
"""

import ast
import gzip
import hashlib
import re
from collections import Counter
from pathlib import Path

import pytest
import responses

from networkcommons.datasets import _data, panacea

COUNTS_FILE = 'panacea/counts/GSE186341_ASPC_dream_counts.csv.gz'
SERIES_FILE = 'panacea/matrix/GSE186341_series_matrix.txt.gz'

COUNTS_CSV = (
    b'"","DMSO__24","DMSO_0_24","ICOTINIB_7.5_24"\n'
    b'"100287102",0,1,5\n'
    b'"1",2,3,4\n'
)

SERIES_MATRIX = '\n'.join([
    '!Sample_title\t"aspc_A2_DMSO_0_24 ASPCPS1AA02_CGTTGTCA"\t'
    '"aspc_C2_DMSO_0_24 ASPCPS1AC02_TTTTACCG"\t'
    '"aspc_F2_Icotinib_7.5_24 ASPCPS1AF02_CAGGCGTA"',
    '!Sample_geo_accession\t"GSM5644146"\t"GSM5644148"\t"GSM5644150"',
    '!Sample_organism_ch1\t"Homo sapiens"\t"Homo sapiens"\t"Homo sapiens"',
    '!Sample_characteristics_ch1\t"treatment: DMSO"\t"treatment: DMSO"\t'
    '"treatment: Icotinib"',
    '!Sample_characteristics_ch1\t"cell line: AsPC-1"\t"cell line: AsPC-1"\t'
    '"cell line: AsPC-1"',
    '!Sample_characteristics_ch1\t"provider: ATCC"\t"provider: ATCC"\t'
    '"provider: ATCC"',
    '',
]).encode('utf-8')


def _hash(content: bytes) -> str:

    return 'sha256:' + hashlib.sha256(content).hexdigest()


def _entry(filename: str, content: bytes) -> tuple:

    return (
        filename,
        _hash(content),
        'https://example.org/' + Path(filename).name,
    )


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """
    Use a dedicated cache inside the test temporary directory.
    """

    root = (tmp_path / 'cache')
    root.mkdir()
    monkeypatch.setattr(_data, '_cache_root', lambda: root.resolve())

    return root


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """
    Replace the package registry with a test one.
    """

    def _write(entries) -> None:

        path = tmp_path / '_registry.txt'
        lines = [
            f'{filename} {checksum} {url}'
            for filename, checksum, url in entries
        ]
        path.write_text('\n'.join(lines) + '\n')
        monkeypatch.setattr(_data, '_REGISTRY_PATH', path)

    return _write


def _seed(cache_root: Path, filename: str, content: bytes) -> Path:

    path = cache_root / filename
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(content)

    return path


def test_registry_integrity():

    cache = _data._cache()
    declared = set(panacea._COUNTS.values()) | {panacea._SERIES_MATRIX}

    assert declared == set(cache.registry)

    for name, checksum in cache.registry.items():

        assert not Path(name).is_absolute(), name
        assert '..' not in Path(name).parts, name
        assert re.fullmatch(r'sha256:[0-9a-f]{64}', checksum), name
        assert cache.get_url(name).startswith('https://'), name


def test_fetch_uses_cache_offline(cache_root, registry):

    filename = 'tests/offline.bin'
    content = b'cached payload'
    registry([_entry(filename, content)])
    _seed(cache_root, filename, content)

    with responses.RequestsMock():

        path = _data.fetch(filename)

    assert path.read_bytes() == content


def test_fetch_without_cache_raises(cache_root, registry):

    registry([_entry('tests/missing.bin', b'payload')])

    with responses.RequestsMock():

        with pytest.raises(_data.DataError):

            _data.fetch('tests/missing.bin')


def test_fetch_unknown_artifact_raises(cache_root, registry):

    registry([])

    with pytest.raises(_data.DataError, match = 'Unknown artifact'):

        _data.fetch('tests/unknown.bin')


def test_fetch_rejects_unsafe_cache_path(cache_root, registry):

    victim = cache_root.parent / 'victim.txt'
    victim.write_text('keep me')
    registry([('../victim.txt', _hash(b'x'), 'https://example.org/evil.bin')])

    with pytest.raises(_data.DataError, match = 'Unsafe cache path'):

        _data.fetch('../victim.txt', refresh = True)

    assert victim.read_text() == 'keep me'


def test_registry_rejects_malformed_entry(tmp_path, monkeypatch):

    path = tmp_path / '_registry.txt'
    path.write_text('only-one-element\n')
    monkeypatch.setattr(_data, '_REGISTRY_PATH', path)

    with pytest.raises(_data.DataError, match = 'Could not read'):

        _data._cache()


def test_fetch_rejects_invalid_checksum(cache_root, registry):

    registry([('tests/badhash.bin', 'not-a-hash', 'https://example.org/x.bin')])

    with pytest.raises(_data.DataError):

        _data.fetch('tests/badhash.bin')


def test_fetch_redownloads_on_hash_mismatch(cache_root, registry):

    filename = 'tests/mismatch.bin'
    content = b'good payload'
    registry([_entry(filename, content)])
    _seed(cache_root, filename, b'corrupt payload')

    with responses.RequestsMock() as mocked:

        mocked.add(
            responses.GET,
            'https://example.org/mismatch.bin',
            body = content,
        )
        path = _data.fetch(filename)

    assert path.read_bytes() == content


def test_fetch_refresh_downloads_again(cache_root, registry):

    filename = 'tests/refresh.bin'
    content = b'fresh payload'
    registry([_entry(filename, content)])
    _seed(cache_root, filename, b'stale payload')

    with responses.RequestsMock() as mocked:

        mocked.add(
            responses.GET,
            'https://example.org/refresh.bin',
            body = content,
        )
        path = _data.fetch(filename, refresh = True)

        assert len(mocked.calls) == 1

    assert path.read_bytes() == content


def test_cell_lines_and_labels():

    lines = panacea.cell_lines()

    assert len(lines) == 11
    assert 'ASPC' in lines
    assert panacea.cell_line_label('ASPC') == 'AsPC-1'
    assert panacea.cell_line_label('H1793') == 'NCI-H1793'

    with pytest.raises(ValueError, match = 'Unknown PANACEA cell line'):

        panacea.cell_line_label('NOPE')


@pytest.fixture
def panacea_registry(cache_root, registry):

    counts = gzip.compress(COUNTS_CSV)
    series = gzip.compress(SERIES_MATRIX)
    registry([
        _entry(COUNTS_FILE, counts),
        _entry(SERIES_FILE, series),
    ])
    _seed(cache_root, COUNTS_FILE, counts)
    _seed(cache_root, SERIES_FILE, series)

    return cache_root


def test_counts_normalizes_gene_column(panacea_registry):

    with responses.RequestsMock():

        table = panacea.counts('ASPC')

    assert list(table.columns) == [
        'gene_id',
        'DMSO__24',
        'DMSO__24.1',
        'ICOTINIB_7.5_24',
    ]
    assert table['gene_id'].tolist() == [100287102, 1]


def test_counts_unknown_cell_line(panacea_registry):

    with pytest.raises(ValueError, match = 'Unknown PANACEA cell line'):

        panacea.counts('NOPE')


def test_metadata_columns_and_groups(panacea_registry):

    with responses.RequestsMock():

        table = panacea.metadata()

    assert list(table['sample_id']) == [
        'aspc_A2_DMSO_0_24',
        'aspc_C2_DMSO_0_24',
        'aspc_F2_Icotinib_7.5_24',
    ]
    assert list(table['group']) == ['DMSO__24', 'DMSO__24', 'ICOTINIB_7.5_24']
    assert list(table['cell_line']) == ['AsPC-1', 'AsPC-1', 'AsPC-1']
    assert list(table['geo_accession']) == [
        'GSM5644146',
        'GSM5644148',
        'GSM5644150',
    ]
    assert list(table['provider']) == ['ATCC', 'ATCC', 'ATCC']


def test_counts_columns_join_metadata_groups(panacea_registry):

    with responses.RequestsMock():

        counts = panacea.counts('ASPC')
        meta = panacea.samples('ASPC')

    columns = Counter(
        re.sub(r'\.\d+$', '', column)
        for column in counts.columns
        if column != 'gene_id'
    )
    groups = Counter(meta['group'])

    assert columns == groups
    assert set(meta['sample_id']).isdisjoint(columns)


def test_samples_filters_cell_line(panacea_registry):

    with responses.RequestsMock():

        rows = panacea.samples('ASPC')

    assert len(rows) == 3
    assert list(rows['cell_line'].unique()) == ['AsPC-1']


def test_samples_unknown_cell_line(panacea_registry):

    with pytest.raises(ValueError, match = 'Unknown PANACEA cell line'):

        panacea.samples('NOPE')


def test_no_pandas_readers_outside_reader_module():

    package = Path(_data.__file__).parent

    for module in sorted(package.glob('*.py')):

        if module.name == '_io.py':

            continue

        tree = ast.parse(module.read_text())

        for node in ast.walk(tree):

            if not isinstance(node, ast.Call):

                continue

            function = node.func

            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == 'pd'
            ):

                assert not function.attr.startswith('read_'), (
                    f'{module.name}: pandas reader {function.attr} outside _io.py'
                )
