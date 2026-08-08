import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datakit import DataKit

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        'department': ['IT', 'HR', 'Finance', 'Marketing', 'Sales'],
        'active': [True, False, True, True, False]
    })

@pytest.fixture
def sample_datakit(sample_df):
    return DataKit(sample_df)

@pytest.fixture
def insurance_csv_path():
    return Path(__file__).parent / "fixtures" / "insurance.csv"

@pytest.fixture
def insurance_datakit(insurance_csv_path):
    return DataKit(insurance_csv_path)

@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        'col_0_missing': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'col_10_missing': [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
        'col_40_missing': [1, 2, 3, 4, 5, 6, None, None, None, None],
        'col_90_missing': [1, None, None, None, None, None, None, None, None, None]
    })

@pytest.fixture
def df_with_duplicates():
    return pd.DataFrame({
        'id': [1, 2, 3, 1, 2, 6],
        'value': ['a', 'b', 'c', 'a', 'b', 'f']
    }) # duplicate rows 0,3 and 1,4

@pytest.fixture
def df_with_constant_column():
    return pd.DataFrame({
        'normal': range(10),
        'constant': [1] * 10,
        'near_constant': ['a'] * 9 + ['b']
    })

@pytest.fixture
def df_empty():
    return pd.DataFrame()

@pytest.fixture
def df_single_row():
    return pd.DataFrame({'a': [1], 'b': ['test']})

@pytest.fixture
def df_all_null():
    return pd.DataFrame({'a': [1, 2], 'b': [None, None]})

@pytest.fixture
def df_mixed_types():
    return pd.DataFrame({
        'real_numeric': [1, 2, 3],
        'numeric_as_string': ["10", "20", "30"],
        'mixed_content': ["1", "two", "3"]
    })

@pytest.fixture
def df_with_id_column():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'user_id': range(1, 21),
        'value': rng.normal(size=20),
        'category': rng.choice(['A', 'B', 'C'], size=20)
    })
