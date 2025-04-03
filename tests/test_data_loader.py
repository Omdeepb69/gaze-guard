```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Assuming data_loader.py is in the same directory or PYTHONPATH
try:
    from data_loader import load_csv_data, handle_missing_values, normalize_data, encode_categorical
except ImportError:
    # Fallback for environments where data_loader might not be directly available
    # This allows the test structure to be parsed, but tests will fail if module not found
    print("Warning: data_loader module not found. Tests requiring it will fail.", file=sys.stderr)
    # Define dummy functions to avoid NameError during collection if needed,
    # but tests using them should ideally fail or be skipped.
    def load_csv_data(filepath): raise ImportError("data_loader.load_csv_data not found")
    def handle_missing_values(dataframe, strategy='mean', columns=None): raise ImportError("data_loader.handle_missing_values not found")
    def normalize_data(dataframe, columns): raise ImportError("data_loader.normalize_data not found")
    def encode_categorical(dataframe, columns): raise ImportError("data_loader.encode_categorical not found")


@pytest.fixture(scope="module")
def temp_data_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("data")
    return temp_dir

@pytest.fixture(scope="module")
def sample_csv_path(temp_data_dir):
    csv_content = """id,feature1,feature2,category,value
1,10,5.5,A,100
2,20,,B,150
3,,8.8,A,120
4,40,10.1,C,
5,50,12.3,B,180
"""
    csv_path = temp_data_dir / "sample_data.csv"
    csv_path.write_text(csv_content)
    return csv_path

@pytest.fixture(scope="module")
def sample_dataframe():
    data = {
        'id': [1, 2, 3, 4, 5],
        'feature1': [10.0, 20.0, np.nan, 40.0, 50.0],
        'feature2': [5.5, np.nan, 8.8, 10.1, 12.3],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'value': [100.0, 150.0, 120.0, np.nan, 180.0]
    }
    return pd.DataFrame(data)


def test_load_csv_data_success(sample_csv_path):
    df = load_csv_data(sample_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (5, 5)
    assert list(df.columns) == ['id', 'feature1', 'feature2', 'category', 'value']
    assert df.loc[0, 'id'] == 1
    assert df.loc[1, 'category'] == 'B'
    assert pd.isna(df.loc[1, 'feature2'])

def test_load_csv_data_file_not_found(temp_data_dir):
    non_existent_path = temp_data_dir / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_csv_data(non_existent_path)

def test_load_csv_data_invalid_csv(temp_data_dir):
    invalid_csv_path = temp_data_dir / "invalid_data.csv"
    invalid_csv_path.write_text("id,feature1\n1,10\n2,20,extra")
    with pytest.raises(Exception): # Catch generic Exception or specific pandas error
         load_csv_data(invalid_csv_path)


def test_handle_missing_values_mean(sample_dataframe):
    df_processed = handle_missing_values(sample_dataframe.copy(), strategy='mean')
    numeric_cols = sample_dataframe.select_dtypes(include=np.number).columns
    assert not df_processed[numeric_cols].isnull().any().any()
    assert df_processed.loc[2, 'feature1'] == pytest.approx(sample_dataframe['feature1'].mean())
    assert df_processed.loc[1, 'feature2'] == pytest.approx(sample_dataframe['feature2'].mean())
    assert df_processed.loc[3, 'value'] == pytest.approx(sample_dataframe['value'].mean())
    assert sample_dataframe['category'].equals(df_processed['category'])
    assert sample_dataframe['id'].equals(df_processed['id'])


def test_handle_missing_values_median(sample_dataframe):
    df_processed = handle_missing_values(sample_dataframe.copy(), strategy='median')
    numeric_cols = sample_dataframe.select_dtypes(include=np.number).columns
    assert not df_processed[numeric_cols].isnull().any().any()
    assert df_processed.loc[2, 'feature1'] == pytest.approx(sample_dataframe['feature1'].median())
    assert df_processed.loc[1, 'feature2'] == pytest.approx(sample_dataframe['feature2'].median())
    assert df_processed.loc[3, 'value'] == pytest.approx(sample_dataframe['value'].median())

def test_handle_missing_values_mode(sample_dataframe):
    data_mode = {
        'numeric': [1.0, 2.0, np.nan, 2.0, 3.0],
        'category': ['A', 'B', 'A', np.nan, 'B']
    }
    df_mode = pd.DataFrame(data_mode)
    df_processed = handle_missing_values(df_mode.copy(), strategy='mode', columns=['numeric', 'category'])

    assert not df_processed['numeric'].isnull().any()
    assert not df_processed['category'].isnull().any()
    assert df_processed.loc[2, 'numeric'] == 2.0
    expected_mode_cat = df_mode['category'].mode()[0]
    assert df_processed.loc[3, 'category'] == expected_mode_cat


def test_handle_missing_values_drop(sample_dataframe):
    df_processed_f1 = handle_missing_values(sample_dataframe.copy(), strategy='drop', columns=['feature1'])
    assert df_processed_f1.shape[0] == 4
    assert 2 not in df_processed_f1.index

    df_processed_val = handle_missing_values(sample_dataframe.copy(), strategy='drop', columns=['value'])
    assert df_processed_val.shape[0] == 4
    assert 3 not in df_processed_val.index

    df_processed_multi = handle_missing_values(sample_dataframe.copy(), strategy='drop', columns=['feature1', 'feature2', 'value'])
    assert df_processed_multi.shape[0] == 2
    assert list(df_processed_multi.index) == [0, 4]


def test_handle_missing_values_invalid_strategy(sample_dataframe):
    with pytest.raises(ValueError):
        handle_missing_values(sample_dataframe.copy(), strategy='unknown')

def test_handle_missing_values_no_nan(sample_dataframe):
    df_no_nan = sample_dataframe.dropna().copy()
    df_processed = handle_missing_values(df_no_nan.copy(), strategy='mean')
    pd.testing.assert_frame_equal(df_no_nan, df_processed)


def test_normalize_data_success(sample_dataframe):
    df_filled = handle_missing_values(sample_dataframe.copy(), strategy='mean')
    cols_to_normalize = ['feature1', 'value']
    df_normalized = normalize_data(df_filled.copy(), columns=cols_to_normalize)

    for col in cols_to_normalize:
        assert df_normalized[col].min() >= 0.0
        assert df_normalized[col].max() <= 1.0 + 1e-9 # Allow for float precision

    f1_min = df_filled['feature1'].min()
    f1_max = df_filled['feature1'].max()
    val_min = df_filled['value'].min()
    val_max = df_filled['value'].max()

    expected_f1_norm_0 = (df_filled.loc[0, 'feature1'] - f1_min) / (f1_max - f1_min)
    expected_val_norm_0 = (df_filled.loc[0, 'value'] - val_min) / (val_max - val_min)
    assert df_normalized.loc[0, 'feature1'] == pytest.approx(expected_f1_norm_0)
    assert df_normalized.loc[0, 'value'] == pytest.approx(expected_val_norm_0)

    assert df_filled['id'].equals(df_normalized['id'])
    assert df_filled['feature2'].equals(