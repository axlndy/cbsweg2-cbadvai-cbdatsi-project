# ==========================================
# Authors: Africa, Akisha Jeneille; Andaya, Axl Roel; Galang, Rienzel Kristian
# Project: CBSWEG2 MCO4 - Student ML Pipelines
# ==========================================

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.model_pipeline import (
    train_logistic_regression,
    train_neural_network,
    generate_predictions
)

# ==========================================
#                  FIXTURES
# ==========================================

@pytest.fixture
def dummy_model_data():
    """
    Generates a mock dataset mimicking the 22 ordinal features and the GPA target.
    Returns 10 rows of sample data.
    """
    np.random.seed(42)
    # Generate 10 rows of 22 features (values 1-5 to mimic the Likert scale)
    X = pd.DataFrame(
        np.random.randint(1, 6, size=(10, 22)), 
        columns=[f"feature_{i}" for i in range(22)]
    )
    # Generate 10 rows of target GPA brackets
    y = pd.Series(np.random.randint(1, 6, size=10), name="GPA")
    return X, y

# ==========================================
#         MODULE 1: MODEL TRAINING
# ==========================================

def test_train_logistic_regression(dummy_model_data):
    """
    Verifies that the Logistic Regression model initializes and fits to the data.
    """
    X, y = dummy_model_data
    model = train_logistic_regression(X, y)
    
    # Verify the object is the correct sklearn model
    assert isinstance(model, LogisticRegression)
    # The 'classes_' attribute only exists if the model was successfully fitted
    assert hasattr(model, 'classes_')

def test_train_neural_network(dummy_model_data):
    """
    Verifies that the MLP Neural Network initializes and fits to the data.
    """
    X, y = dummy_model_data
    model = train_neural_network(X, y)
    
    assert isinstance(model, MLPClassifier)
    assert hasattr(model, 'classes_')

# ==========================================
#         MODULE 2: PREDICTIONS
# ==========================================

def test_generate_predictions_lr(dummy_model_data):
    """
    Verifies that predictions return an array matching the number of input rows.
    """
    X, y = dummy_model_data
    model = train_logistic_regression(X, y)
    
    predictions = generate_predictions(model, X)
    
    # If we pass 10 rows in, we should get 10 predictions out
    assert len(predictions) == len(X)

def test_generate_predictions_nn(dummy_model_data):
    """
    Verifies that the neural network predictions output the correct shape.
    """
    X, y = dummy_model_data
    model = train_neural_network(X, y)
    
    predictions = generate_predictions(model, X)
    
    assert len(predictions) == len(X)