# tests/test_student_eda.py

import pytest
import pandas as pd
import numpy as np

from src.student_eda import (
    validate_dataset,
    clean_and_typecast_data,
    perform_feature_engineering
)


@pytest.fixture
def dummy_student_data():
    """
    Generates a valid mock student dataset for preprocessing tests.
    """
    return pd.DataFrame({
        "Year": [3.0, 5.0, 4.0],
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
        "GPA": [3, 4, 1],
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


def test_clean_and_typecast_data(dummy_student_data):
    """
    Verifies that numeric survey columns are converted to integer data types.
    """
    processed_df = clean_and_typecast_data(dummy_student_data)

    assert processed_df["Year"].dtype == np.int64
    assert processed_df["GPA"].dtype == np.int64
    assert processed_df["Year"].iloc[0] == 3


def test_perform_feature_engineering(dummy_student_data):
    """
    Verifies that encoded survey responses are correctly mapped to descriptive labels.
    """
    processed_df = clean_and_typecast_data(dummy_student_data)
    final_df = perform_feature_engineering(processed_df)

    assert final_df["GPA_Label"].iloc[0] == "Fair"
    assert final_df["GPA_Label"].iloc[2] == "Poor"

    assert final_df["Gender_Label"].iloc[1] == "Female"

    assert final_df["Poor_Stu_Label"].iloc[1] == "Yes (Poor)"

    assert final_df["Policy_Stu_Label"].iloc[0] == "No (Not Supported)"