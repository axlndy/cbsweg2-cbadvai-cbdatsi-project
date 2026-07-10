# tests/test_student_eda.py
import pytest
import pandas as pd
import numpy as np
from src.student_eda import clean_and_typecast_data, perform_feature_engineering

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
    
    # Assert descriptive targets mapped correctly
    assert final_df['GPA_Label'].iloc[0] == 'Fair'
    assert final_df['GPA_Label'].iloc[2] == 'Poor'
    assert final_df['Gender_Label'].iloc[1] == 'Female'
    assert final_df['Poor_Stu_Label'].iloc[1] == 'Yes (Poor)'
    assert final_df['Policy_Stu_Label'].iloc[0] == 'No (Not Supported)'