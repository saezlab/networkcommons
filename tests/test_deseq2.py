import pytest

import pandas as pd

from networkcommons.data import omics

@pytest.mark.slow
def test_deseq2():

    # Create dummy dataset for testing, samples as colnames, genes as rownames
    counts = pd.DataFrame({
        'gene_symbol': ['Gene1', 'Gene2', 'Gene3'],
        'Sample1': [90, 150, 10],
        'Sample2': [80, 60, 9],
        'Sample3': [100, 80, 12],
        'Sample4': [100, 120, 17]
    })

    metadata = pd.DataFrame({
        'sample_ID': ['Sample1', 'Sample2', 'Sample3', 'Sample4'],
        'group': ['Control', 'Treatment', 'Treatment', 'Control']
    })

    result = omics.deseq2(
        counts,
        metadata,
        ref_group='Control',
        test_group='Treatment',
    )

    assert isinstance(result, pd.DataFrame)
    cols_expected = {'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj'}
    assert cols_expected.issubset(result.columns)

    data = {
        'baseMean': [93.233027, 101.285704, 11.793541],
        'log2FoldChange': [0.218173, -0.682184, -0.052951],
        'lfcSE': [0.328029, 0.352410, 0.521688],
        'stat': [0.665101, -1.935768, -0.101500],
        'pvalue': [0.505986, 0.052896, 0.919154],
        'padj': [0.758979, 0.158688, 0.919154]
    }

    expected_result = pd.DataFrame(data, index=['Gene1', 'Gene2', 'Gene3'])
    expected_result.index.name = 'gene_symbol'
    pd.testing.assert_frame_equal(result, expected_result, check_exact=False)
