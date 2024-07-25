import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import networkcommons.visual as visual
from unittest.mock import patch
import pytest

@pytest.mark.slow
def test_pca_with_metadata_df():
    df = pd.DataFrame({
        'idx': ['A', 'B', 'C'],
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    })
    metadata_df = pd.DataFrame({
        'sample': ['A', 'B', 'C'],
        'group': ['control', 'treated', 'control']
    })
    result_df = visual.plot_pca(df, metadata_df)
    assert isinstance(result_df, pd.DataFrame)
    assert 'PCA1' in result_df.columns
    assert 'PCA2' in result_df.columns
    assert 'group' in result_df.columns


@pytest.mark.slow
def test_pca_with_metadata_array():
    df = pd.DataFrame({
        'idx': ['A', 'B', 'C'],
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    })
    metadata_arr = np.array(['control', 'treated', 'control'])
    result_df = visual.plot_pca(df, metadata_arr)
    assert isinstance(result_df, pd.DataFrame)
    assert 'PCA1' in result_df.columns
    assert 'PCA2' in result_df.columns
    assert 'group' in result_df.columns


@pytest.mark.slow
def test_pca_no_numeric_columns():
    df = pd.DataFrame({'idx': ['A', 'B', 'C']})
    metadata_df = pd.DataFrame({
        'sample': ['A', 'B', 'C'],
        'group': ['control', 'treated', 'control']
    })
    try:
        visual.plot_pca(df, metadata_df)
    except ValueError as e:
        assert str(e) == "The dataframe contains no numeric columns suitable for PCA."


@pytest.mark.slow
def test_pca_zero_std_columns():
    df_with_zero_std = pd.DataFrame({
        'idx': ['feature1', 'feature2', 'feature3'],
        'A': [1, 1, 1],
        'B': [1, 5, 6],
        'C': [1, 8, 9]
    })
    metadata_df = pd.DataFrame({
        'sample': ['A', 'B', 'C'],
        'group': ['control', 'treated', 'control']
    })
    with patch('builtins.print') as mocked_print:
        result_df = visual.plot_pca(df_with_zero_std, metadata_df)
        print(mocked_print.mock_calls)  # Print the captured print calls for debugging
        mocked_print.assert_any_call("Warning: The following columns have zero standard deviation and will be dropped: ['feature1']")
