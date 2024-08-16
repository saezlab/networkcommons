import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from networkcommons.data import omics

# Here th nternal DESeq2 components are mocked to isolate the test from the actual pyDESeq2 implementation.
@patch('networkcommons.data.omics._deseq2._log')
@patch('networkcommons.data.omics._deseq2._conf.get', return_value=1)
@patch('networkcommons.data.omics._deseq2._deseq2_ds.DeseqStats')
@patch('networkcommons.data.omics._deseq2._deseq2_dds.DeseqDataSet')
@patch('networkcommons.data.omics._deseq2._deseq2_default_inference.DefaultInference')
def test_deseq2(mock_inference, mock_dds, mock_stats, mock_conf_get, mock_log):
    counts = pd.DataFrame({
        'gene_symbol': ['Gene1', 'Gene2', 'Gene3'],
        'Sample1': [90, 150, 10],
        'Sample2': [80, 60, 9],
        'Sample3': [100, 80, 12],
        'Sample4': [100, 120, 17]
    })

    metadata = pd.DataFrame({
        'sample_ID': ['Sample1', 'Sample2', 'Sample3', 'Sample4'],
        'group': ['Control_Group', 'Treatment_Group', 'Treatment_Group', 'Control_Group']
    })

    mock_dds_instance = MagicMock()
    mock_dds.return_value = mock_dds_instance

    mock_stats_instance = MagicMock()
    mock_stats_instance.results_df = pd.DataFrame({
        'baseMean': [93.233027, 101.285704, 11.793541],
        'log2FoldChange': [0.218173, -0.682184, -0.052951],
        'lfcSE': [0.328029, 0.352410, 0.521688],
        'stat': [0.665101, -1.935768, -0.101500],
        'pvalue': [0.505986, 0.052896, 0.919154],
        'padj': [0.758979, 0.158688, 0.919154]
    }, index=['Gene1', 'Gene2', 'Gene3'])
    mock_stats_instance.results_df.index.name = 'gene_symbol'

    mock_stats.return_value = mock_stats_instance

    result = omics.deseq2(
        counts,
        metadata,
        ref_group='Control_Group',
        test_group='Treatment_Group',
    )
    # now without haifens
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

    expected_result = pd.DataFrame({
        'baseMean': [93.233027, 101.285704, 11.793541],
        'log2FoldChange': [0.218173, -0.682184, -0.052951],
        'lfcSE': [0.328029, 0.352410, 0.521688],
        'stat': [0.665101, -1.935768, -0.101500],
        'pvalue': [0.505986, 0.052896, 0.919154],
        'padj': [0.758979, 0.158688, 0.919154]
    }, index=['Gene1', 'Gene2', 'Gene3'])
    expected_result.index.name = 'gene_symbol'
    
    pd.testing.assert_frame_equal(result, expected_result, check_exact=False)
    mock_log.assert_called_with('Finished running DESeq2.')
