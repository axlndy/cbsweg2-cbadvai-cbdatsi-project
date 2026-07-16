# tests/test_student_eda.py

import pytest
import pandas as pd
import numpy as np
import os
import tempfile

from src.student_eda import (
    load_and_cache_dataset,
    validate_dataset,
    clean_and_typecast_data,
    perform_feature_engineering
)

# ==========================================
#                  FIXTURES
# ==========================================

@pytest.fixture
def dummy_student_data():
    """
    Generates a valid mock student dataset for preprocessing tests.
    Contains all 22 required columns mapped to valid categorical integer ranges.
    """
    return pd.DataFrame({
        "Year": [3, 5, 4],
        "Gender": [1, 2, 1],
        "Policy_Stu": [2, 2, 1],
        "Minority_Stu": [2, 1, 2],
        "Poor_Stu": [2, 1, 2],
        "Father_Edu": [3, 4, 5],
        "Mother_Edu": [2, 3, 4],
        "Father_Occupation": [2, 3, 1],
        "Mother_Occupation": [1, 2, 3],
        "Time_Friends": [3, 2, 5],
        "Time_SocicalMedia": [1, 5, 3],
        "Time_Studying": [2, 4, 1],
        #"GPA": [3, 4, 1],
        "Adapt_Learning_Uni": [4, 5, 2],
        "Study_Methods": [4, 5, 3],
        "SupportOf_Uni": [4, 5, 3],
        "SupportOf_Lec": [5, 4, 2],
        "Facilitie_Uni": [3, 4, 5],
        "Quality_Lecturer": [4, 5, 3],
        "TrainingCurriculum": [5, 4, 2],
        "Competitive_Class": [3, 2, 4],
        "InfuenceF_Friends": [4, 3, 5]
    })


# ==========================================
#         MODULE 1: LOAD & CACHE
# ==========================================

def test_load_and_cache_dataset_first_run(dummy_student_data):
    """
    Verifies that the raw Excel dataset is loaded and a cache file
    is created when no cache exists.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(temp_dir, "sample.xlsx")
        cache_path = os.path.join(temp_dir, "dataset_cache.pkl")

        # Create a temporary Excel dataset
        dummy_student_data.to_excel(raw_path, index=False)

        # Execute
        loaded_df = load_and_cache_dataset(raw_path, cache_path)

        # Verify
        assert os.path.exists(cache_path)
        pd.testing.assert_frame_equal(loaded_df, dummy_student_data)


def test_load_and_cache_dataset_from_cache(dummy_student_data):
    """
    Verifies that an existing cache is loaded instead of reading
    the Excel file again.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(temp_dir, "sample.xlsx")
        cache_path = os.path.join(temp_dir, "dataset_cache.pkl")

        # Create both files
        dummy_student_data.to_excel(raw_path, index=False)
        dummy_student_data.to_pickle(cache_path)

        loaded_df = load_and_cache_dataset(raw_path, cache_path)

        pd.testing.assert_frame_equal(loaded_df, dummy_student_data)


def test_load_and_cache_dataset_missing_file():
    """
    Verifies that loading fails when the raw dataset does not exist.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(temp_dir, "missing.xlsx")
        cache_path = os.path.join(temp_dir, "dataset_cache.pkl")

        with pytest.raises(FileNotFoundError):
            load_and_cache_dataset(raw_path, cache_path)


# ==========================================
#         MODULE 2: VALIDATION
# ==========================================

def test_validate_dataset(dummy_student_data):
    """
    Verifies that a valid dataset successfully passes validation.
    """
    assert validate_dataset(dummy_student_data) is True


def test_validate_dataset_missing_column(dummy_student_data):
    """
    Verifies that validation fails when a required column is missing.
    """
    invalid_df = dummy_student_data.drop(columns=["Gender"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(invalid_df)


def test_validate_dataset_invalid_value(dummy_student_data):
    """
    Verifies that validation fails when a survey variable contains an invalid encoded value.
    """
    invalid_df = dummy_student_data.copy()
    invalid_df.loc[0, "Gender"] = 99

    with pytest.raises(ValueError, match="Invalid value"):
        validate_dataset(invalid_df)


def test_validate_dataset_empty():
    """
    Verifies that validation fails safely if an entirely empty dataset is passed.
    Because the dataframe is empty, it will trigger the missing columns check first.
    """
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="Missing required columns"): 
        validate_dataset(empty_df)


# ==========================================
#       MODULE 3: CLEAN & TYPECAST
# ==========================================

def test_clean_and_typecast_data(dummy_student_data):
    """
    Verifies that numeric survey columns are converted to integer data types.
    """
    processed_df = clean_and_typecast_data(dummy_student_data)

    assert processed_df["Year"].dtype == np.int64
    assert processed_df["GPA"].dtype == np.int64
    assert processed_df["Year"].iloc[0] == 3


def test_clean_and_typecast_data_type_corruption(dummy_student_data):
    """
    Assures that if categorical text values are corrupted (e.g., a string instead of an int), 
    the pandas astype() function catches it and fails safely.
    """
    corrupted_df = dummy_student_data.copy()

    corrupted_df["GPA"] = corrupted_df["GPA"].astype(object)
    
    corrupted_df.loc[0, "GPA"] = "High" 

    with pytest.raises((TypeError, ValueError)): 
        clean_and_typecast_data(corrupted_df)

# ==========================================
#     MODULE 4: FEATURE ENGINEERING
# ==========================================

def test_perform_feature_engineering(dummy_student_data):
    """
    Verifies that encoded survey responses are correctly mapped to descriptive labels.
    """
    processed_df = clean_and_typecast_data(dummy_student_data)
    final_df = perform_feature_engineering(processed_df)

    # Validate mapping accuracy based on our specific mapping  dictionary
    assert final_df["GPA_Label"].iloc[0] == "Fair"   # GPA 3 -> Fair
    assert final_df["GPA_Label"].iloc[2] == "Poor"   # GPA 1 -> Poor
    
    assert final_df["Gender_Label"].iloc[1] == "Female" # Gender 2 -> Female
    
    assert final_df["Poor_Stu_Label"].iloc[1] == "Yes (Poor)" # Poor_Stu 1 -> Yes (Poor)
    
    assert final_df["Policy_Stu_Label"].iloc[0] == "No (Not Supported)" # Policy_Stu 2 -> No
