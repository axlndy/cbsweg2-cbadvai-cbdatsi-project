import pytest
import pandas as pd
import numpy as np
from src.cbadvai.preprocessing import SELECTED_FEATURES

@pytest.fixture
def synthetic_survey_df():
    """
    Dynamically generates an in-memory synthetic dataset matching 
    SELECTED_FEATURES and the GPA target schema.
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {}
    for feature in SELECTED_FEATURES:
        if feature == 'Policy_Stu':
            # Raw survey binary choices: 1 (Yes) or 2 (No)
            data[feature] = np.random.choice([1, 2], size=n_samples)
        else:
            data[feature] = np.random.randint(1, 6, size=n_samples)
            
    data['GPA'] = np.random.choice([1, 2, 3, 4, 5], size=n_samples)
    
    return pd.DataFrame(data)
