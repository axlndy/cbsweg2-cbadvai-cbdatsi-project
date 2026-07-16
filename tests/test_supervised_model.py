# tests/test_supervised_model.py
import pytest
import numpy as np
import pandas as pd
from src.train_supervised import train_ordinal_model, SingularMatrixWarning

# =====================================================================
#                         AUTOMATED UNIT TESTS
# =====================================================================

@pytest.fixture
def dummy_training_set():
    """
    Generates balanced baseline training dataset wrapper containing
    exactly 5 ordered target classes for the Ordinal Model.
    """
    return pd.DataFrame({
        'col_A': [1, 2, 3, 4, 5, 6],
        'col_B': [5, 4, 3, 2, 1, 0],
        'col_C': [2, 4, 1, 5, 3, 2],
        'GPA': [1, 2, 3, 4, 5, 3] # Evaluates exactly 5 ordinal boundaries (1 to 5)
    })


def test_EDP_UT_018_successful_model_training(dummy_training_set):
    """
    EDP-UT-018: Verifies that the model trains completely and returns a 
    valid statsmodels results wrapper object.
    """
    features = ['col_A', 'col_C']
    target = 'GPA'
    
    model_object = train_ordinal_model(dummy_training_set, features, target)
    assert model_object is not None
    assert hasattr(model_object, 'mle_retvals')


def test_EDP_UT_019_model_convergence_verification(dummy_training_set):
    """
    EDP-UT-019: Confirms optimizer optimization successfully terminates within boundaries.
    """
    features = ['col_A', 'col_C']
    target = 'GPA'
    
    model_object = train_ordinal_model(dummy_training_set, features, target)
    assert model_object.mle_retvals["converged"] is True
    assert isinstance(model_object.llf, float)


def test_EDP_UT_020_prediction_generation(dummy_training_set):
    """
    EDP-UT-020: Validates that a matching 1D matrix of discrete indices is output
    matching the input feature count.
    """
    features = ['col_A', 'col_C']
    target = 'GPA'
    
    model_res = train_ordinal_model(dummy_training_set, features, target)
    prob_distributions = model_res.model.predict(model_res.params, exog=dummy_training_set[features])
    predicted_classes = np.argmax(prob_distributions, axis=1) + 1
    
    assert len(predicted_classes) == len(dummy_training_set)
    assert isinstance(predicted_classes, np.ndarray)


def test_EDP_UT_021_probability_validation(dummy_training_set):
    """
    EDP-UT-021: Confirms class allocation vectors total strictly to 1.0 distribution precision.
    """
    features = ['col_A', 'col_C']
    target = 'GPA'
    
    model_res = train_ordinal_model(dummy_training_set, features, target)
    prob_distributions = model_res.model.predict(model_res.params, exog=dummy_training_set[features])
    
    # 1. Check that the probability rows sum to 1.0
    row_sums = np.sum(prob_distributions, axis=1)
    assert np.allclose(row_sums, 1.0)
    
    # 2. Check the output dimensions (6 samples, 5 classes)
    assert prob_distributions.shape == (6, 5)


def test_EDP_UT_022_perfect_multicollinearity_case():
    """
    EDP-UT-022: Verifies singular array exceptions are emitted when column values match perfectly.
    """
    collinear_dataframe = pd.DataFrame({
        'col_A': [1, 2, 3, 4, 5, 6],
        'col_B': [1, 2, 3, 4, 5, 6],  # Exact duplicate of col_A to force 100% correlation
        'GPA': [1, 2, 3, 4, 5, 3]
    })
    features = ['col_A', 'col_B']
    target = 'GPA'
    
    # Catch both the warnings and the underlying linear algebra exception from our rank check
    with pytest.warns(SingularMatrixWarning) as record:
        with pytest.raises(np.linalg.LinAlgError) as exc_info:
            train_ordinal_model(collinear_dataframe, features, target)
            
    assert "Matrix is singular to working precision." in str(record[0].message)
    assert "Singular matrix error in regression modeling." in str(exc_info.value)
