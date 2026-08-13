# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

from src.cbadvai.preprocessing import SELECTED_FEATURES


@pytest.fixture
def dummy_student_data():
    """
    Generates a valid mock student dataset (CBDATSI module contract).

    The fixture contains all 22 required dataset columns and uses
    valid categorical/ordinal codes based on the CBDATSI data dictionary.
    """
    return pd.DataFrame({
        "Year": [3, 5, 4, 1, 2],
        "Gender": [1, 2, 1, 2, 1],
        "Policy_Stu": [2, 2, 1, 1, 2],
        "Minority_Stu": [2, 1, 2, 2, 1],
        "Poor_Stu": [2, 1, 2, 1, 2],

        "Father_Edu": [3, 4, 5, 2, 1],
        "Mother_Edu": [2, 3, 4, 1, 2],

        "Father_Occupation": [2, 3, 1, 4, 2],
        "Mother_Occupation": [1, 2, 3, 4, 1],

        "Time_Friends": [3, 2, 5, 1, 4],
        "Time_SocicalMedia": [1, 5, 3, 2, 4],
        "Time_Studying": [2, 4, 1, 5, 3],

        "GPA": [3, 4, 1, 5, 2],

        "Adapt_Learning_Uni": [4, 5, 2, 3, 1],
        "Study_Methods": [4, 5, 3, 2, 1],

        "SupportOf_Uni": [4, 5, 3, 2, 1],
        "SupportOf_Lec": [5, 4, 2, 3, 1],
        "Facilitie_Uni": [3, 4, 5, 1, 2],
        "Quality_Lecturer": [4, 5, 3, 2, 1],
        "TrainingCurriculum": [5, 4, 2, 3, 1],

        "Competitive_Class": [3, 2, 4, 1, 5],
        "InfuenceF_Friends": [4, 3, 5, 2, 1]
    })


@pytest.fixture
def synthetic_survey_df():
    """
    Generates a 100-row synthetic student survey dataset for CBADVAI model training & evaluation tests.
    Includes all SELECTED_FEATURES and GPA target classes (1-5).
    """
    np.random.seed(42)
    n_samples = 100

    data = {col: np.random.randint(1, 6, size=n_samples) for col in SELECTED_FEATURES}
    data['Policy_Stu'] = np.random.choice([0, 1], size=n_samples)
    data['GPA'] = np.random.choice([1, 2, 3, 4, 5], size=n_samples)

    return pd.DataFrame(data)
