# tests/test_student_eda.py
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np
from src.student_eda import load_and_cache_dataset, clean_and_typecast_data, perform_feature_engineering

@pytest.fixture
def dummy_student_data():
    """Generates mock survey observations to validate transformations."""
    return pd.DataFrame({
        'Year': [3.0, 5.0, 4.0],
        'Gender': [1, 2, 1],
        'GPA': [3, 4, 1],
        'Time_Studying': [2, 4, 1],
        'Time_Friends': [3, 2, 5],
        'Time_SocicalMedia': [1, 5, 3],
        'Adapt_Learning_Uni': [4, 5, 2],
        'Poor_Stu': [2, 1, 2],
        'Policy_Stu': [2, 2, 1],
        'Minority_Stu': [2, 1, 2],
        'SupportOf_Uni': [4, 5, 3],
        'SupportOf_Lec': [5, 4, 2],
        'Facilitie_Uni': [3, 4, 5],
        'Quality_Lecturer': [4, 5, 3],
        'TrainingCurriculum': [5, 4, 2],
        'Study_Methods': [4, 5, 3],
        'Competitive_Class': [3, 2, 4],
        'InfuenceF_Friends': [4, 3, 5]
    })

@patch('os.path.exists', return_value=True)
@patch('pandas.read_pickle')
def test_load_and_cache_dataset_from_cache(mock_read_pickle, mock_exists, dummy_student_data):
    """Verifies that data is correctly loaded from the pickle cache when it exists."""
    mock_read_pickle.return_value = dummy_student_data
    
    df = load_and_cache_dataset("data/fake_raw.xlsx", "data/fake_cache.pkl")
    
    assert df.shape == (3, 18)
    mock_read_pickle.assert_called_once_with("data/fake_cache.pkl")

@patch('os.path.exists')
@patch('pandas.read_excel')
@patch('pandas.DataFrame.to_pickle')
@patch('os.makedirs')
def test_load_dataset_first_try_success(mock_makedirs, mock_to_pickle, mock_read_excel, mock_exists, dummy_student_data):
    """Tests the first-run scenario where the cache is missing but the raw Excel file exists."""
    # Side effect: cache path does NOT exist (False), but raw Excel path DOES exist (True)
    mock_exists.side_effect = lambda path: path == "data/fake_raw.xlsx"
    mock_read_excel.return_value = dummy_student_data
    
    df = load_and_cache_dataset("data/fake_raw.xlsx", "data/fake_cache.pkl")
    
    assert df.shape == (3, 18)
    mock_read_excel.assert_called_once_with("data/fake_raw.xlsx")
    mock_to_pickle.assert_called_once_with("data/fake_cache.pkl")

@patch('os.path.exists', return_value=False)
def test_load_dataset_missing_file_raises_error(mock_exists):
    """Verifies that a FileNotFoundError is raised if both cache and raw files are missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_and_cache_dataset("data/missing_raw.xlsx", "data/missing_cache.pkl")
        
    assert "Missing raw data file at: data/missing_raw.xlsx" in str(exc_info.value)

def test_clean_and_typecast_data(dummy_student_data):
    """Checks that floated survey columns are accurately cast into structural integers."""
    processed_df = clean_and_typecast_data(dummy_student_data)
    
    assert processed_df['Year'].dtype == np.int64
    assert processed_df['GPA'].dtype == np.int64
    assert processed_df['Year'].iloc[0] == 3

def test_perform_feature_engineering(dummy_student_data):
    """Validates that discrete survey integers successfully match to literal textual strings."""
    processed_df = clean_and_typecast_data(dummy_student_data)
    final_df = perform_feature_engineering(processed_df)
    
    assert final_df['GPA_Label'].iloc[0] == 'Fair'
    assert final_df['GPA_Label'].iloc[2] == 'Poor'
    assert final_df['Gender_Label'].iloc[1] == 'Female'
    assert final_df['Poor_Stu_Label'].iloc[1] == 'Yes (Poor)'
    assert final_df['Policy_Stu_Label'].iloc[0] == 'No (Not Supported)'
