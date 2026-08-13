import pytest
import numpy as np
import pandas as pd

# Adjust imports based on your exact file structure
from src.cbadvai.metrics import evaluate_ordinal_model, evaluate_dummy_baseline

def test_evaluate_ordinal_model():
    """Validates that the custom ordinal evaluation metrics compute accurately."""
    # Dummy arrays: 5 samples, 1 mistake (predicted 5 instead of 4)
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 5, 5]) 
    
    # Run the evaluation
    results = evaluate_ordinal_model(y_true, y_pred, model_name="Test Model")
    
    # Verify the dictionary contains all expected keys
    expected_keys = ['Macro_F1', 'MAE', 'QWK', 'Confusion_Matrix']
    for key in expected_keys:
        assert key in results, f"Missing '{key}' in evaluation output dictionary."
    
    # Verify MAE calculation (1 total absolute error / 5 samples = 0.2)
    assert results['MAE'] == 0.2, f"Expected MAE of 0.2, but got {results['MAE']}"
    
    # Verify Confusion Matrix shape (should be a 5x5 matrix for 5 classes)
    # The confusion matrix function in scikit-learn adapts to the labels present,
    # but since both 1 through 5 are present in true/pred combined, it will be 5x5.
    assert results['Confusion_Matrix'].shape == (5, 5), "Confusion Matrix dimensions are incorrect."

def test_evaluate_dummy_baseline():
    """Validates that the dummy baseline generates metrics successfully using the most frequent class."""
    # Create mock training data where class '3' is the most frequent
    X_train = pd.DataFrame({'feature1': [1, 2, 1, 1, 2]})
    y_train = pd.Series([3, 3, 3, 4, 5]) 
    
    # Create mock testing data
    X_test = pd.DataFrame({'feature1': [1, 2]})
    y_test = pd.Series([3, 4])
    
    # Run the dummy baseline evaluation
    results = evaluate_dummy_baseline(
        X_train, y_train, X_test, y_test, strategy="most_frequent"
    )
    
    # The dummy classifier will predict '3' for both test samples.
    # True: [3, 4] | Pred: [3, 3]
    # MAE Calculation: (|3-3| + |4-3|) / 2 = (0 + 1) / 2 = 0.5
    assert results['MAE'] == 0.5, f"Expected Dummy MAE of 0.5, but got {results['MAE']}"
    assert results['Macro_F1'] > 0, "Macro F1 should compute successfully and be > 0"