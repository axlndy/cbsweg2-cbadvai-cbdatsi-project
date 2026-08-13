import pytest
import pandas as pd
import numpy as np
from src.cbadvai.preprocessing import SELECTED_FEATURES

@pytest.fixture
def synthetic_survey_df():
    """
    Generates a synthetic survey dataset ensuring at least 15 samples per class
    so SMOTE and cross-validation folds never crash on minority bounds.
    """
    np.random.seed(42)
    n_per_class = 20  # 20 samples * 5 classes = 100 total samples
    
    # Ensure every target GPA class (1 to 5) has exactly 20 samples
    gpa = np.hstack([[c] * n_per_class for c in [1, 2, 3, 4, 5]])
    np.random.shuffle(gpa)
    
    data = {}
    for feature in SELECTED_FEATURES:
        if feature == 'Policy_Stu':
            # Raw binary survey values: 1 (Yes) or 2 (No)
            data[feature] = np.random.choice([1, 2], size=100)
        else:
            # 1 to 5 Likert scale
            data[feature] = np.random.randint(1, 6, size=100)
            
    data['GPA'] = gpa
    return pd.DataFrame(data)
