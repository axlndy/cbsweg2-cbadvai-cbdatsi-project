import pytest
import pandas as pd
import numpy as np
from src.cbadvai.preprocessing import SELECTED_FEATURES

# Strict baseline contract of expected survey features
EXPECTED_FEATURES = [
    'Study_Methods', 
    'Time_Studying', 
    'Time_Friends', 
    'Time_SocicalMedia', 
    'Adapt_Learning_Uni', 
    'Policy_Stu', 
    'SupportOf_Uni', 
    'SupportOf_Lec', 
    'Facilitie_Uni', 
    'Quality_Lecturer', 
    'TrainingCurriculum'
]

@pytest.fixture
def synthetic_survey_df():
    """
    Generates a fully compliant, in-memory synthetic dataset matching 
    the exact 11 features and GPA target schema from Database paper.xlsx.
    
    Guarantees exactly 20 samples per GPA class (100 total samples) so 
    SMOTE resampling and cross-validation splits never crash on minority class bounds.
    """
    np.random.seed(42)
    n_per_class = 20  # 20 samples * 5 classes = 100 total samples
    
    # Ensure every target GPA class (1 to 5) has exactly 20 samples
    gpa = np.hstack([[c] * n_per_class for c in [1, 2, 3, 4, 5]])
    np.random.shuffle(gpa)
    
    data = {}
    for feature in EXPECTED_FEATURES:
        if feature == 'Policy_Stu':
            # Raw binary survey values: 1 (Yes) or 2 (No)
            data[feature] = np.random.choice([1, 2], size=100)
        else:
            # 1 to 5 Likert scale (upper bound 6 is exclusive -> yields integers 1,2,3,4,5)
            data[feature] = np.random.randint(1, 6, size=100)
            
    data['GPA'] = gpa
    return pd.DataFrame(data)
