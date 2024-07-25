import networkcommons._utils as utils
import pandas as pd
import numpy as np


def test_fill_and_drop():
    df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [3, 2, np.nan], 'C': [np.nan, 7, 8]})
    result = utils.handle_missing_values(df, 0.5)
    expected = pd.DataFrame({'index': [0, 1], 'A': [1.0, 2.0], 'B': [3.0, 2.0], 'C': [2.0, 7.0]}).astype({'index': 'int64'})
    pd.testing.assert_frame_equal(result, expected)


def test_all_rows_dropped():
    df = pd.DataFrame({'A': [1, np.nan, np.nan], 'B': [np.nan, np.nan, np.nan], 'C': [np.nan, np.nan, 8]})
    result = utils.handle_missing_values(df, 0.1)
    expected = pd.DataFrame({'index': [], 'A': [], 'B': [], 'C': []}).astype({'index': 'int64'})
    pd.testing.assert_frame_equal(result, expected)


def test_non_numeric_column():
    df = pd.DataFrame({'id': ['a', 'b', 'c'], 'A': [1, 2, np.nan], 'B': [3, 2, np.nan], 'C': [np.nan, 7, 8]})
    result = utils.handle_missing_values(df, 0.5)
    expected = pd.DataFrame({'id': ['a', 'b'], 'A': [1.0, 2.0], 'B': [3.0, 2.0], 'C': [2.0, 7.0]})
    pd.testing.assert_frame_equal(result, expected)


def test_more_than_one_non_numeric_column():
    df = pd.DataFrame({'id1': ['a', 'b', 'c'], 'id2': ['x', 'y', 'z'], 'A': [1, 2, np.nan], 'B': [3, 2, np.nan]})
    try:
        utils.handle_missing_values(df, 0.5)
    except ValueError as e:
        assert str(e) == "More than one non-numeric column found: Index(['id1', 'id2'], dtype='object')"


def test_no_missing_values():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = utils.handle_missing_values(df, 0.5)
    expected = df.reset_index().rename(columns={'index': 'index'})
    expected = expected.astype({'index': 'int64'})
    pd.testing.assert_frame_equal(result, expected)
