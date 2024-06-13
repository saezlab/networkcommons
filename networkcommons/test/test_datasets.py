from networkcommons.datasets import (
    get_available_datasets,
    download_dataset,
    run_deseq2_analysis
)
import pandas as pd


# Test get_available_datasets
def test_get_available_datasets():
    # Call the get_available_datasets function
    datasets = get_available_datasets()

    # Assert that the returned value is a list
    assert isinstance(datasets, list)

    # Assert that the returned list is not empty
    assert len(datasets) > 0


def test_download_dataset():
    # Call the download_dataset function with a specific dataset
    dataset = 'unit_test'
    data = download_dataset(dataset)

    # Assert that the returned value is a list
    assert isinstance(data, dict)
    assert 'test__countdata' in data
    assert 'test__metadata' in data
    assert isinstance(data['test__countdata'], pd.DataFrame)
    assert isinstance(data['test__metadata'], pd.DataFrame)

    # Assert that the returned list is not empty
    assert len(data) > 0


def test_run_deseq2_analysis():
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

    # Call the deseq2_analysis function
    result = run_deseq2_analysis(counts,
                                 metadata,
                                 ref_group='Control',
                                 test_group='Treatment')

    # Assert that the returned value is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert that the DataFrame has the expected columns
    assert 'log2FoldChange' in result.columns
    assert 'lfcSE' in result.columns
    assert 'stat' in result.columns
    assert 'pvalue' in result.columns
    assert 'padj' in result.columns

    # Assert that the DataFrame has the expected content
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
